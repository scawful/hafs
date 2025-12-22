import shutil
import subprocess
from pathlib import Path
from typing import List, Optional

from .base import SourceControlProvider, VCSFileStatus, VCSStatusType


class GitProvider(SourceControlProvider):
    @property
    def name(self) -> str:
        return "git"

    def is_available(self) -> bool:
        return shutil.which("git") is not None

    def _run_git(self, args: List[str], cwd: Path) -> str:
        try:
            result = subprocess.run(
                ["git"] + args,
                cwd=str(cwd),
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return ""

    def is_repository(self, path: Path) -> bool:
        if not path.exists():
            return False
        # git rev-parse --is-inside-work-tree
        try:
            subprocess.run(
                ["git", "rev-parse", "--is-inside-work-tree"],
                cwd=str(path),
                capture_output=True,
                check=True
            )
            return True
        except subprocess.CalledProcessError:
            return False

    def get_root(self, path: Path) -> Optional[Path]:
        if not self.is_repository(path):
            return None
        root_str = self._run_git(["rev-parse", "--show-toplevel"], path)
        return Path(root_str) if root_str else None

    def get_branch(self, path: Path) -> Optional[str]:
        if not self.is_repository(path):
            return None
        return self._run_git(["rev-parse", "--abbrev-ref", "HEAD"], path)

    def get_status(self, path: Path) -> List[VCSFileStatus]:
        if not self.is_repository(path):
            return []

        root = self.get_root(path)
        if not root:
            return []

        # git status --porcelain
        output = self._run_git(["status", "--porcelain"], root)
        statuses = []

        for line in output.splitlines():
            if len(line) < 4:
                continue

            x = line[0]
            y = line[1]
            file_path_str = line[3:].strip()
            # Handle quoted paths if necessary, but git status --porcelain usually
            # behaves okay without quotes for simple paths.
            # If filenames have spaces, they might be quoted.
            if file_path_str.startswith('"') and file_path_str.endswith('"'):
                 file_path_str = file_path_str[1:-1]

            full_path = root / file_path_str

            # Determine status
            # X          Y     Meaning
            # -------------------------------------------------
            #          [AMD]   not updated
            # M        [ MD]   updated in index
            # A        [ MD]   added to index
            # D                deleted from index
            # R        [ MD]   renamed in index
            # C        [ MD]   copied in index
            # [MARC]           index and work tree matches
            # [ MARC]     M    work tree changed since index
            # [ MARC]     D    deleted in work tree
            # [ D]        R    renamed in work tree
            # [ D]        C    copied in work tree
            # -------------------------------------------------
            # D           D    unmerged, both deleted
            # A           U    unmerged, added by us
            # U           D    unmerged, deleted by them
            # U           A    unmerged, added by them
            # D           U    unmerged, deleted by us
            # A           A    unmerged, both added
            # U           U    unmerged, both modified
            # -------------------------------------------------
            # ?           ?    untracked
            # !           !    ignored
            # -------------------------------------------------

            status_type = VCSStatusType.UNKNOWN
            is_staged = False

            if x in 'MADRC':
                is_staged = True

            if x == '?' and y == '?':
                status_type = VCSStatusType.UNTRACKED
            elif x == '!' and y == '!':
                status_type = VCSStatusType.IGNORED
            elif x == 'M' or y == 'M':
                status_type = VCSStatusType.MODIFIED
            elif x == 'A' or y == 'A':
                status_type = VCSStatusType.ADDED
            elif x == 'D' or y == 'D':
                status_type = VCSStatusType.DELETED
            elif x == 'R' or y == 'R':
                status_type = VCSStatusType.RENAMED

            statuses.append(VCSFileStatus(path=full_path, status=status_type, staged=is_staged))

        return statuses

    def stage_file(self, file_path: Path) -> bool:
        root = self.get_root(file_path.parent)
        if not root:
            return False
        # git add <file>
        try:
            subprocess.run(
                ["git", "add", str(file_path)],
                cwd=str(root), check=True, capture_output=True
            )
            return True
        except subprocess.CalledProcessError:
            return False

    def unstage_file(self, file_path: Path) -> bool:
        root = self.get_root(file_path.parent)
        if not root:
            return False
        # git restore --staged <file>
        try:
            subprocess.run(
                ["git", "restore", "--staged", str(file_path)],
                cwd=str(root), check=True, capture_output=True
            )
            return True
        except subprocess.CalledProcessError:
            return False

    def commit(self, path: Path, message: str) -> bool:
        root = self.get_root(path)
        if not root:
            return False
        try:
            subprocess.run(
                ["git", "commit", "-m", message],
                cwd=str(root), check=True, capture_output=True
            )
            return True
        except subprocess.CalledProcessError:
            return False

    def get_diff(self, file_path: Path, staged: bool = False) -> str:
        root = self.get_root(file_path.parent)
        if not root:
            return ""

        args = ["diff"]
        if staged:
            args.append("--staged")
        args.append(str(file_path))

        return self._run_git(args, root)
