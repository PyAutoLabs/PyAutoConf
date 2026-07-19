import os
from pathlib import Path


def setup_notebook():
    """
    Set up a Jupyter notebook to run from the workspace root.

    Finds the workspace root by walking up from the current directory
    looking for a ``config`` directory (the marker for a PyAuto workspace),
    changes to it, reconfigures autonerves paths, and enables inline plotting.
    """
    root = _find_workspace_root()
    os.chdir(root)

    from autonerves import conf

    conf.instance.push(
        new_path=root / "config",
        output_path=root / "output",
    )

    try:
        ipy = get_ipython()
        ipy.run_line_magic("matplotlib", "inline")
    except NameError:
        pass

    print(f"Working Directory has been set to `{root}`")


def _find_workspace_root(start=None):
    """Walk up from *start* (default: cwd) looking for a ``config/`` dir."""
    current = Path(start or os.getcwd()).resolve()
    for parent in [current, *current.parents]:
        if (parent / "config").is_dir():
            return parent
    raise FileNotFoundError(
        "Could not find workspace root (no 'config/' directory found "
        f"in {current} or any parent)."
    )
