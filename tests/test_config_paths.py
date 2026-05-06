from pathlib import Path

from src.config import _default_workspace_root


def test_default_workspace_root_is_non_empty() -> None:
    workspace_root = _default_workspace_root()

    assert isinstance(workspace_root, Path)
    assert workspace_root.is_absolute()
    assert workspace_root.exists()
