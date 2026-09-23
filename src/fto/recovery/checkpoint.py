import os
from pathlib import Path
import subprocess
import tempfile


_FTO_EXCLUDE_PATTERNS = [
    'execution_logs.json',
    'fto_execution_logs.json',
    'fto.log',
    'node_outputs.yaml',
    'workflow_summary.yaml',
    'token_usage_*.json',
    'traces.jsonl',
    '.fto/',
]

# Diff files live under the repo's own worktree rather than the system temp
# dir: a restarted node's file tools resolve paths inside the workspace the
# MAS sandbox gives them, not arbitrary absolute paths like /tmp/*, so a diff
# handed back as an out-of-tree path is one the node can never actually open.
_DIFF_SUBDIR = Path('.fto') / 'diffs'


class Checkpoint:
    def __init__(self) -> None:
        pass

    def save_baseline(self) -> None:
        pass

    def save(self, node_id: str) -> None:
        pass

    def restore(self, node_id: str) -> None:
        pass

    def diff(self, idx: int, node_id: str) -> None:
        pass

class GitBranchCheckpoint(Checkpoint):
    def __init__(self, repo_path: Path, run_id: str) -> None:
        super().__init__()
        self.repo_path = repo_path
        self.run_id = run_id
        self.branch_prefix = f'FTO-{self.run_id}'

        if not self._is_git_repo():
            self._git('init')

        self._write_local_excludes()

    def _git(self, *args) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                ['git', *args],
                cwd=self.repo_path,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            stderr = e.stderr.strip() if e.stderr else ''
            raise RuntimeError(
                f"git {' '.join(str(a) for a in args)} failed "
                f"(exit {e.returncode})"
                + (f': {stderr}' if stderr else '')
            ) from None

    def _write_local_excludes(self) -> None:
        """Write FTO framework files to .git/info/exclude so they are never committed."""
        exclude_file = self.repo_path / '.git' / 'info' / 'exclude'
        exclude_file.parent.mkdir(parents=True, exist_ok=True)
        existing = exclude_file.read_text(encoding='utf-8') if exclude_file.exists() else ''
        additions = [p for p in _FTO_EXCLUDE_PATTERNS if p not in existing]
        if additions:
            with exclude_file.open('a', encoding='utf-8') as f:
                f.write('\n# FTO framework files\n')
                f.write('\n'.join(additions) + '\n')

    def _is_git_repo(self) -> bool:
        try:
            self._git('rev-parse', '--is-inside-work-tree')
            return True
        except (subprocess.CalledProcessError, RuntimeError):
            return False

    def _is_dirty(self) -> bool:
        result = subprocess.run(
            ['git', 'status', '--porcelain'],
            cwd=self.repo_path,
            capture_output=True,
            text=True,
        )
        return bool(result.stdout.strip())

    def _commit_if_dirty(self, msg: str) -> bool:
        if not self._is_dirty():
            return
        self._git('add', '-A')
        # `git status --porcelain` can flag things (e.g. submodule content
        # changes) that `git add -A` doesn't actually stage, leaving nothing
        # to commit. Re-check the index instead of assuming staging worked.
        staged = subprocess.run(
            ['git', 'diff', '--cached', '--quiet'], cwd=self.repo_path
        )
        if staged.returncode == 0:
            return
        self._git('commit', '-m', msg)

    def _ref(self, node_id: str) -> str:
        return f'{self.branch_prefix}-{node_id}'

    def save_baseline(self) -> None:
        self._commit_if_dirty('[FTO] baseline')
        self.baseline_ref = f'{self.branch_prefix}-baseline'
        self._git('branch', '-f', self.baseline_ref, 'HEAD')

    def save(self, node_id: str) -> None:
        self._commit_if_dirty(f'[FTO] pre-{node_id} exec')
        self._git('branch', '-f', self._ref(node_id), 'HEAD')

    def restore(self, node_id: str) -> None:
        self._git('restore', '--source', self._ref(node_id), '--worktree', '--', ':/')

    def diff(self, idx: int, node_id: str) -> Path | None:
        if idx is not None and idx > 4:
            raise ValueError(
                f'idx_step {idx} is not supported; the injection index '
                f'cannot be higher than 4.'
            )
        ref = self._ref(node_id) if idx == 4 else self.baseline_ref
        result = self._git('diff', ref)
        if not result.stdout.strip():
            return None
        diff_dir = self.repo_path / _DIFF_SUBDIR
        diff_dir.mkdir(parents=True, exist_ok=True)
        fd, path = tempfile.mkstemp(
            prefix=f'{self.branch_prefix}-diff-', suffix='.patch', dir=diff_dir
        )
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(result.stdout)
        return Path(path).relative_to(self.repo_path)
