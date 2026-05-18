import os
import warnings
from pathlib import Path

from autoconf import exc


class WorkspaceVersionMismatchError(exc.ConfigException):
    pass


_BYPASS_ENV_VAR = "PYAUTO_SKIP_WORKSPACE_VERSION_CHECK"


def _read_general_yaml(workspace_root):
    """
    Return the parsed ``config/general.yaml`` dict for ``workspace_root``.

    Returns an empty dict on any failure (missing file, missing yaml module,
    unreadable YAML) so the caller can fall through to legacy ``version.txt``
    handling without crashing the user's script on import.
    """
    try:
        import yaml

        config_path = workspace_root / "config" / "general.yaml"
        with config_path.open("r") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _yaml_bypass_set(general_yaml):
    return general_yaml.get("version", {}).get("workspace_version_check") is False


def _yaml_workspace_version(general_yaml):
    value = general_yaml.get("version", {}).get("workspace_version")
    if value is None:
        return None
    return str(value).strip()


def _missing_version_warning(workspace_root, library_version):
    return (
        f"Cannot verify the workspace at {workspace_root} matches the "
        f"installed library version ({library_version}): no "
        f"`version.workspace_version` key in config/general.yaml and no "
        f"version.txt at the workspace root.\n\n"
        f"If you cloned the workspace from `main` rather than a release tag, "
        f"set `version.workspace_version_check: False` in "
        f"config/general.yaml to silence this warning. The `main` branch "
        f"updates more frequently than library releases, so version "
        f"mismatches are expected and not actionable for `main`-branch users.\n\n"
        f"You can also set the environment variable "
        f"{_BYPASS_ENV_VAR}=1 to disable temporarily."
    )


def _parse_version(version_string):
    try:
        return tuple(int(part) for part in version_string.split("."))
    except (AttributeError, ValueError):
        return None


def _library_name_from_workspace(workspace_root):
    name = workspace_root.name
    suffix = "_workspace"
    if name.endswith(suffix) and len(name) > len(suffix):
        return name[: -len(suffix)]
    return None


def _update_library_block(library_name, target_version):
    package = library_name if library_name else "<library>"
    return (
        f"Your workspace is newer than your installed library. Update the "
        f"library to match:\n\n"
        f"    pip install --upgrade {package}=={target_version}"
    )


def _update_workspace_block(workspace_root):
    return (
        f"Your installed library is newer than your workspace clone. Pull "
        f"the latest workspace `main`:\n\n"
        f"    cd {workspace_root} && git pull origin main"
    )


def _mismatch_message(workspace_version, library_version, workspace_root):
    library_name = _library_name_from_workspace(workspace_root)
    ws_parsed = _parse_version(workspace_version)
    lib_parsed = _parse_version(library_version)

    if ws_parsed is not None and lib_parsed is not None and ws_parsed > lib_parsed:
        advice = _update_library_block(library_name, workspace_version)
    elif ws_parsed is not None and lib_parsed is not None and lib_parsed > ws_parsed:
        advice = _update_workspace_block(workspace_root)
    else:
        advice = (
            f"{_update_library_block(library_name, workspace_version)}\n\n"
            f"Or, if your workspace is the side that is out of date:\n\n"
            f"{_update_workspace_block(workspace_root)}"
        )

    return (
        f"Workspace version ({workspace_version}) at {workspace_root} does "
        f"not match the installed library version ({library_version}).\n\n"
        f"{advice}\n\n"
        f"To bypass this check, edit config/general.yaml:\n\n"
        f"    version:\n"
        f"      workspace_version_check: False\n\n"
        f"IMPORTANT: If you cloned the workspace from `main` rather than a "
        f"release tag, you should set `workspace_version_check: False`. The "
        f"`main` branch updates much more frequently than library releases, "
        f"so version mismatches are expected and not actionable for "
        f"`main`-branch users.\n\n"
        f"You can also set the environment variable "
        f"{_BYPASS_ENV_VAR}=1 to disable temporarily."
    )


def check_version(library_version, workspace_root=None):
    """
    Verify that the workspace at ``workspace_root`` matches ``library_version``.

    Resolves the workspace version with the following precedence:

    1. ``config/general.yaml`` — ``version.workspace_version`` key, written by
       the release pipeline. Travels with the user's config directory even
       when scripts are copy-pasted out of the workspace root.
    2. ``version.txt`` at the workspace root — legacy fallback for clones
       that pre-date the YAML key.

    If neither source is found, a warning is emitted and the check is
    skipped. If both sources exist but disagree, the YAML value wins
    (release pipeline writes both atomically; YAML is the configured
    source-of-truth on the user's machine).

    The check can be disabled in two ways:

    * Set ``version.workspace_version_check: False`` in
      ``config/general.yaml`` — the recommended path for users on
      ``main``-branch workspace clones, where mismatches are expected
      because ``main`` updates more frequently than library releases.
    * Set ``PYAUTO_SKIP_WORKSPACE_VERSION_CHECK=1`` — intended for
      developers running source checkouts where workspace and library
      versions intentionally diverge.

    Defaults ``workspace_root`` to the current working directory, which is
    where users run workspace scripts from.
    """
    if os.environ.get(_BYPASS_ENV_VAR) == "1":
        return

    root = Path(workspace_root) if workspace_root else Path.cwd()

    general_yaml = _read_general_yaml(root)

    if _yaml_bypass_set(general_yaml):
        return

    workspace_version = _yaml_workspace_version(general_yaml)

    if workspace_version is None:
        version_file = root / "version.txt"
        if version_file.exists():
            workspace_version = version_file.read_text().strip()

    if workspace_version is None or workspace_version == "":
        warnings.warn(_missing_version_warning(root, library_version))
        return

    if workspace_version != library_version:
        raise WorkspaceVersionMismatchError(
            _mismatch_message(workspace_version, library_version, root)
        )
