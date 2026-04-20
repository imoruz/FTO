from pathlib import Path
import subprocess


class Checkpoint:
    def __init__(self):
        pass
    
    def save_baseline(self):
        pass

    def save(self, node_id: str):
        pass

    def restore(self, node_id: str):
        pass


class GitBranchCheckpoint(Checkpoint):
    def __init__(self, repo_path: Path, run_id: str):
        super().__init__()
        self.repo_path = repo_path
        self.run_id = run_id
        self.branch_prefix = f"FTO-{self.run_id}"

    def _git(self, *args):
        try:
            return subprocess.run(
                ["git", *args],
                cwd=self.repo_path,
                check=True,
                capture_output=True,
                text=True
            )
        except subprocess.CalledProcessError as e:
            raise subprocess.CalledProcessError(
                e.returncode, e.cmd, e.output,
                f"stderr: {e.stderr.strip()}"
            ) from None
    
    def _is_dirty(self) -> bool:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=self.repo_path,
            capture_output=True,
            text=True,
        )
        return bool(result.stdout.strip())

    def _commit_if_dirty(self, msg: str) -> bool:
        if self._is_dirty():
            self._git("add", "-A")
            self._git("commit", "-m", msg)

    def _ref(self, node_id: str) -> str:
        return f"{self.branch_prefix}-{node_id}"

    def save_baseline(self):
        self._commit_if_dirty("[FTO] baseline")
        self.baseline_ref = f"{self.branch_prefix}-baseline"
        self._git("branch", "-f", self.baseline_ref, "HEAD")

    def save(self, node_id: str):
        self._commit_if_dirty(f"[FTO] pre-{node_id} exec")
        self._git("branch", "-f", self._ref(node_id), "HEAD")

    def restore(self, node_id: str):
        self._git("restore", "--source", self._ref(node_id), "--worktree", "--", ".")