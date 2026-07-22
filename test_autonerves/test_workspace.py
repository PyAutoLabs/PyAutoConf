import textwrap
import warnings

import pytest

from autonerves.workspace import check_version, WorkspaceVersionMismatchError


def _write_general_yaml(tmp_path, body):
    config_dir = tmp_path / "config"
    config_dir.mkdir(exist_ok=True)
    (config_dir / "general.yaml").write_text(textwrap.dedent(body).lstrip())


def test_match_via_version_txt(tmp_path):
    (tmp_path / "version.txt").write_text("2026.7.22.1\n")
    check_version("2026.7.22.1", workspace_root=tmp_path)


def test_mismatch_via_version_txt_raises(tmp_path):
    (tmp_path / "version.txt").write_text("2026.7.22.1\n")
    with pytest.raises(WorkspaceVersionMismatchError) as info:
        check_version("2025.1.1.1", workspace_root=tmp_path)
    msg = str(info.value)
    assert "2026.7.22.1" in msg
    assert "2025.1.1.1" in msg
    assert "workspace_version_check: False" in msg
    assert "main" in msg


def test_missing_sources_warns(tmp_path):
    with pytest.warns(UserWarning, match="workspace_version_check: False"):
        check_version("2026.7.22.1", workspace_root=tmp_path)


def test_env_override_skips_mismatch(tmp_path, monkeypatch):
    monkeypatch.setenv("PYAUTO_SKIP_WORKSPACE_VERSION_CHECK", "1")
    (tmp_path / "version.txt").write_text("2025.1.1.1\n")
    check_version("2026.7.22.1", workspace_root=tmp_path)


def test_env_override_skips_missing_file(tmp_path, monkeypatch):
    monkeypatch.setenv("PYAUTO_SKIP_WORKSPACE_VERSION_CHECK", "1")
    check_version("2026.7.22.1", workspace_root=tmp_path)


def test_default_root_is_cwd(tmp_path, monkeypatch):
    (tmp_path / "version.txt").write_text("2026.7.22.1\n")
    monkeypatch.chdir(tmp_path)
    check_version("2026.7.22.1")


def test_match_via_general_yaml(tmp_path):
    _write_general_yaml(
        tmp_path,
        """
        version:
          workspace_version: 2026.7.22.1
          workspace_version_check: True
        """,
    )
    check_version("2026.7.22.1", workspace_root=tmp_path)


def test_mismatch_via_general_yaml_raises(tmp_path):
    _write_general_yaml(
        tmp_path,
        """
        version:
          workspace_version: 2026.7.22.1
          workspace_version_check: True
        """,
    )
    with pytest.raises(WorkspaceVersionMismatchError):
        check_version("2025.1.1.1", workspace_root=tmp_path)


def test_yaml_bypass_skips_mismatch(tmp_path):
    _write_general_yaml(
        tmp_path,
        """
        version:
          workspace_version: 2026.7.22.1
          workspace_version_check: False
        """,
    )
    check_version("2025.1.1.1", workspace_root=tmp_path)


def test_yaml_bypass_skips_missing_version_key(tmp_path):
    _write_general_yaml(
        tmp_path,
        """
        version:
          workspace_version_check: False
        """,
    )
    check_version("2026.7.22.1", workspace_root=tmp_path)


def test_yaml_overrides_version_txt_when_both_present(tmp_path):
    _write_general_yaml(
        tmp_path,
        """
        version:
          workspace_version: 2026.7.22.1
        """,
    )
    (tmp_path / "version.txt").write_text("2025.1.1.1\n")
    check_version("2026.7.22.1", workspace_root=tmp_path)


def test_version_txt_used_when_yaml_lacks_workspace_version(tmp_path):
    _write_general_yaml(
        tmp_path,
        """
        version:
          python_version_check: True
        """,
    )
    (tmp_path / "version.txt").write_text("2026.7.22.1\n")
    check_version("2026.7.22.1", workspace_root=tmp_path)


def test_yaml_without_version_key_falls_through_to_warning(tmp_path):
    _write_general_yaml(
        tmp_path,
        """
        updates:
          iterations_per_quick_update: 1
        """,
    )
    with pytest.warns(UserWarning):
        check_version("2026.7.22.1", workspace_root=tmp_path)


def test_unparseable_yaml_falls_back_to_version_txt(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "general.yaml").write_text("::: not valid yaml :::")
    (tmp_path / "version.txt").write_text("2026.7.22.1\n")
    check_version("2026.7.22.1", workspace_root=tmp_path)


def test_newer_library_within_window_passes_silently(tmp_path):
    (tmp_path / "version.txt").write_text("2026.7.22.1\n")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_version("2026.7.29.1", workspace_root=tmp_path)


def test_newer_library_beyond_window_warns_stale(tmp_path):
    (tmp_path / "version.txt").write_text("2026.7.22.1\n")
    with pytest.warns(UserWarning, match="git pull"):
        check_version("2026.9.9.1", workspace_root=tmp_path)


def test_minimum_library_version_key_is_the_floor(tmp_path):
    _write_general_yaml(
        tmp_path,
        """
        version:
          minimum_library_version: 2026.7.22.1
          workspace_version: 2027.1.1.1
        """,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_version("2026.7.29.1", workspace_root=tmp_path)


def test_below_minimum_library_version_raises_with_upgrade_advice(tmp_path):
    workspace_root = tmp_path / "autolens_workspace"
    workspace_root.mkdir()
    _write_general_yaml(
        workspace_root,
        """
        version:
          minimum_library_version: 2026.7.22.1
        """,
    )
    with pytest.raises(WorkspaceVersionMismatchError) as info:
        check_version("2025.1.1.1", workspace_root=workspace_root)
    msg = str(info.value)
    assert "pip install --upgrade autolens" in msg
    assert "==" not in msg


def test_unparseable_library_version_warns_not_raises(tmp_path):
    (tmp_path / "version.txt").write_text("2026.7.22.1\n")
    with pytest.warns(UserWarning, match="cannot be compared"):
        check_version("1.0.dev0", workspace_root=tmp_path)


def test_unparseable_workspace_record_warns_not_raises(tmp_path):
    (tmp_path / "version.txt").write_text("not-a-version\n")
    with pytest.warns(UserWarning, match="cannot be compared"):
        check_version("2026.7.22.1", workspace_root=tmp_path)
