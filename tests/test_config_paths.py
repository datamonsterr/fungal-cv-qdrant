from pathlib import Path

from src.config import _default_workspace_root


def test_default_workspace_root_resolves_monorepo_root() -> None:
    workspace_root = _default_workspace_root()

    assert workspace_root == Path("/home/dat/dev/mycoai")
