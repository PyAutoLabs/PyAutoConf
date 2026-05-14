"""
autoconf — configuration, serialization, and I/O helpers for the PyAuto ecosystem.

Text-format I/O surfaces:

- :mod:`autoconf.dictable`  — JSON (``output_to_json`` / ``from_json``)
- :mod:`autoconf.fitsable`  — FITS (``output_to_fits`` / ``ndarray_via_fits_from``)
- :mod:`autoconf.csvable`   — CSV  (``output_to_csv`` / ``list_from_csv``)
"""
import sys
import warnings
from pathlib import Path


def _python_version_check_bypassed():
    """
    Return True iff the user's workspace config disables the Python version check.

    Reads ``<cwd>/config/general.yaml`` and looks for ``version.python_version_check``.
    Any failure (missing file, unreadable YAML, missing key, missing yaml module) is
    treated as "not bypassed" so the default check still fires.
    """
    try:
        import yaml

        config_path = Path.cwd() / "config" / "general.yaml"
        with config_path.open("r") as f:
            data = yaml.safe_load(f) or {}
        return data.get("version", {}).get("python_version_check") is False
    except Exception:
        return False


_RECOMMENDED_PYTHON_VERSIONS = {(3, 12), (3, 13)}


def _emit_python_version_warning():
    current = sys.version_info[:2]
    if current in _RECOMMENDED_PYTHON_VERSIONS:
        return
    if _python_version_check_bypassed():
        return

    py = f"{current[0]}.{current[1]}"
    lines = [
        f"PyAuto: Python {py} detected -- first-class support is 3.12 and 3.13.",
        "",
        f"Things will probably work fine on {py}, but it is not the recommended",
        "version and you may hit edge cases.",
    ]
    if current < (3, 11):
        lines.extend(
            [
                "",
                "Note: JAX acceleration is not available on Python <3.11. Models",
                "that pass use_jax=True will error.",
            ]
        )
    lines.extend(
        [
            "",
            "Recommended: install Python 3.12 or 3.13.",
            "To silence this warning, add to <cwd>/config/general.yaml:",
            "",
            "    version:",
            "      python_version_check: False",
        ]
    )

    inner_width = max(len(line) for line in lines)
    border = "+" + "-" * (inner_width + 4) + "+"
    framed = [border]
    for line in lines:
        framed.append("|  " + line.ljust(inner_width) + "  |")
    framed.append(border)

    print("\n".join(framed), file=sys.stderr)
    warnings.warn(
        f"PyAuto: running on Python {py}; first-class support is 3.12/3.13. "
        f"Suppress this warning via 'version.python_version_check: False' in "
        f"config/general.yaml.",
        UserWarning,
        stacklevel=2,
    )


_emit_python_version_warning()

from . import jax_wrapper
from . import exc
from .tools.decorators import cached_property
from .conf import Config
from .conf import instance
from .json_prior.config import default_prior
from .json_prior.config import make_config_for_class
from .json_prior.config import path_for_class
from .json_prior.config import JSONPriorConfig

from .setup_colab import for_autolens
from .setup_notebook import setup_notebook
from .test_mode import test_mode_level, is_test_mode, skip_fit_output, skip_visualization, skip_checks
from .workspace import check_version, WorkspaceVersionMismatchError


__version__ = "2026.5.14.2"
