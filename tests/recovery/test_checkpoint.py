import subprocess

import pytest

from fto.recovery.checkpoint import Checkpoint, GitBranchCheckpoint


def _git(repo_path, *args):
    return subprocess.run(
        ['git', *args], cwd=repo_path, check=True, capture_output=True, text=True,
    )


def _init_repo_with_identity(repo_path):
    """Pre-init the repo with a committer identity so GitBranchCheckpoint's own
    `git init` is skipped and `git commit` has something to attribute commits to."""
    _git(repo_path, 'init')
    _git(repo_path, 'config', 'user.email', 'test@example.com')
    _git(repo_path, 'config', 'user.name', 'Test User')


def _branches(repo_path):
    result = _git(repo_path, 'branch', '--list')
    return {line.strip().lstrip('* ').strip() for line in result.stdout.splitlines() if line.strip()}


class TestCheckpointBase:
    def test_methods_are_noops(self):
        checkpoint = Checkpoint()
        assert checkpoint.save_baseline() is None
        assert checkpoint.save('node-1') is None
        assert checkpoint.restore('node-1') is None


class TestGitBranchCheckpointInit:
    def test_initializes_git_repo_when_not_already_one(self, tmp_path):
        assert not (tmp_path / '.git').exists()

        GitBranchCheckpoint(repo_path=tmp_path, run_id='run1')

        assert (tmp_path / '.git').is_dir()

    def test_does_not_reinit_an_existing_repo(self, tmp_path):
        _init_repo_with_identity(tmp_path)
        (tmp_path / 'a.txt').write_text('hello')
        _git(tmp_path, 'add', '-A')
        _git(tmp_path, 'commit', '-m', 'initial')
        head_before = _git(tmp_path, 'rev-parse', 'HEAD').stdout.strip()

        GitBranchCheckpoint(repo_path=tmp_path, run_id='run1')

        head_after = _git(tmp_path, 'rev-parse', 'HEAD').stdout.strip()
        assert head_before == head_after

    def test_writes_fto_exclude_patterns(self, tmp_path):
        GitBranchCheckpoint(repo_path=tmp_path, run_id='run1')

        exclude_file = tmp_path / '.git' / 'info' / 'exclude'
        content = exclude_file.read_text(encoding='utf-8')
        assert 'execution_logs.json' in content
        assert 'traces.jsonl' in content

    def test_writing_excludes_twice_does_not_duplicate_entries(self, tmp_path):
        GitBranchCheckpoint(repo_path=tmp_path, run_id='run1')
        exclude_file = tmp_path / '.git' / 'info' / 'exclude'
        content_after_first = exclude_file.read_text(encoding='utf-8')

        GitBranchCheckpoint(repo_path=tmp_path, run_id='run2')

        content_after_second = exclude_file.read_text(encoding='utf-8')
        assert content_after_second == content_after_first

    def test_branch_prefix_includes_run_id(self, tmp_path):
        checkpoint = GitBranchCheckpoint(repo_path=tmp_path, run_id='abc123')
        assert checkpoint.branch_prefix == 'FTO-abc123'


class TestGitBranchCheckpointGitHelper:
    def test_git_raises_runtime_error_on_failure(self, tmp_path):
        _init_repo_with_identity(tmp_path)
        checkpoint = GitBranchCheckpoint(repo_path=tmp_path, run_id='run1')

        with pytest.raises(RuntimeError, match='git .*failed'):
            checkpoint._git('not-a-real-git-subcommand')

    def test_is_git_repo_true_for_initialized_repo(self, tmp_path):
        _init_repo_with_identity(tmp_path)
        checkpoint = GitBranchCheckpoint(repo_path=tmp_path, run_id='run1')
        assert checkpoint._is_git_repo() is True

    def test_is_dirty_reflects_working_tree_state(self, tmp_path):
        _init_repo_with_identity(tmp_path)
        checkpoint = GitBranchCheckpoint(repo_path=tmp_path, run_id='run1')
        assert checkpoint._is_dirty() is False

        (tmp_path / 'a.txt').write_text('hello')

        assert checkpoint._is_dirty() is True


class TestGitBranchCheckpointSaveRestore:
    def _checkpoint(self, tmp_path, run_id='run1'):
        _init_repo_with_identity(tmp_path)
        return GitBranchCheckpoint(repo_path=tmp_path, run_id=run_id)

    def test_save_baseline_commits_and_creates_baseline_branch(self, tmp_path):
        checkpoint = self._checkpoint(tmp_path)
        (tmp_path / 'a.txt').write_text('v1')

        checkpoint.save_baseline()

        assert checkpoint.baseline_ref == 'FTO-run1-baseline'
        assert 'FTO-run1-baseline' in _branches(tmp_path)
        assert checkpoint._is_dirty() is False
        log = _git(tmp_path, 'log', '-1', '--pretty=%s').stdout.strip()
        assert log == '[FTO] baseline'

    def test_save_baseline_is_a_noop_commit_when_clean(self, tmp_path):
        checkpoint = self._checkpoint(tmp_path)
        (tmp_path / 'a.txt').write_text('v1')
        checkpoint.save_baseline()
        head_after_baseline = _git(tmp_path, 'rev-parse', 'HEAD').stdout.strip()

        checkpoint.save_baseline()

        assert _git(tmp_path, 'rev-parse', 'HEAD').stdout.strip() == head_after_baseline

    def test_save_commits_and_creates_node_branch(self, tmp_path):
        checkpoint = self._checkpoint(tmp_path)
        (tmp_path / 'a.txt').write_text('v1')
        checkpoint.save_baseline()

        (tmp_path / 'a.txt').write_text('v2')
        checkpoint.save('node-1')

        assert 'FTO-run1-node-1' in _branches(tmp_path)
        log = _git(tmp_path, 'log', '-1', '--pretty=%s').stdout.strip()
        assert log == '[FTO] pre-node-1 exec'

    def test_restore_resets_worktree_to_saved_state(self, tmp_path):
        checkpoint = self._checkpoint(tmp_path)
        (tmp_path / 'a.txt').write_text('v1')
        checkpoint.save_baseline()

        (tmp_path / 'a.txt').write_text('v2')
        checkpoint.save('node-1')

        (tmp_path / 'a.txt').write_text('v3-uncommitted')

        checkpoint.restore('node-1')

        assert (tmp_path / 'a.txt').read_text() == 'v2'

    def test_restore_can_go_back_to_an_earlier_checkpoint(self, tmp_path):
        checkpoint = self._checkpoint(tmp_path)
        (tmp_path / 'a.txt').write_text('v1')
        checkpoint.save_baseline()

        (tmp_path / 'a.txt').write_text('v2')
        checkpoint.save('node-1')

        (tmp_path / 'a.txt').write_text('v3')
        checkpoint.save('node-2')

        checkpoint.restore('node-1')

        assert (tmp_path / 'a.txt').read_text() == 'v2'

    def test_multiple_run_ids_keep_separate_branch_namespaces(self, tmp_path):
        checkpoint_a = self._checkpoint(tmp_path, run_id='runA')
        (tmp_path / 'a.txt').write_text('v1')
        checkpoint_a.save_baseline()

        checkpoint_b = GitBranchCheckpoint(repo_path=tmp_path, run_id='runB')
        (tmp_path / 'a.txt').write_text('v2')
        checkpoint_b.save_baseline()

        branches = _branches(tmp_path)
        assert 'FTO-runA-baseline' in branches
        assert 'FTO-runB-baseline' in branches
