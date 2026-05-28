import os
from pathlib import Path


def test_mode_level():
    """
    Return the current test mode level.

    0 = off (normal operation)
    1 = reduce sampler iterations to minimum (existing behavior)
    2 = bypass sampler entirely, call likelihood once
    3 = bypass sampler entirely, skip likelihood call
    """
    return int(os.environ.get("PYAUTO_TEST_MODE", "0"))


def is_test_mode():
    """
    Return True if any test mode is active.
    """
    return test_mode_level() > 0


def skip_fit_output():
    """
    Return True if fit I/O should be skipped.

    Controls: pre/post-fit output, VRAM profiling, result info text,
    likelihood function checks.
    """
    return os.environ.get("PYAUTO_SKIP_FIT_OUTPUT", "0") == "1"


def skip_visualization():
    """
    Return True if fit visualization should be skipped.

    Controls: Visualizer.should_visualize, plot decorators,
    quantity visualizers.
    """
    return os.environ.get("PYAUTO_SKIP_VISUALIZATION", "0") == "1"


def skip_checks():
    """
    Return True if validation checks should be skipped.

    Controls: mesh pixel validation (hilbert), position resampling,
    inversion position exceptions, sample weight thresholds.
    """
    return os.environ.get("PYAUTO_SKIP_CHECKS", "0") == "1"


def skip_latents():
    """
    Return True if latent variable computation should be skipped.

    Auto-enabled when any ``PYAUTO_TEST_MODE`` level is active (test-mode
    fits mock the underlying samples, so latent values are not meaningful)
    and can also be triggered independently via ``PYAUTO_SKIP_LATENTS=1``
    for real-mode fits where the user wants to bypass the post-fit
    ``compute_latent_samples`` pass.
    """
    return is_test_mode() or os.environ.get("PYAUTO_SKIP_LATENTS", "0") == "1"


def with_test_mode_segment(base: Path) -> Path:
    """
    Return ``base`` with a ``test_mode`` segment appended when
    ``PYAUTO_TEST_MODE`` is active, else return ``base`` unchanged.

    Workspace scripts that compose their own output paths (e.g. the
    ``guides/results/aggregator/`` tutorials in ``autolens_workspace`` and
    ``autogalaxy_workspace``) must agree with PyAutoFit's internal
    ``_test_mode_segment`` (``autofit/non_linear/paths/abstract.py``), which
    inserts ``output/test_mode/...`` whenever ``PYAUTO_TEST_MODE`` is set.
    This helper exposes the same namespacing rule so workspace path
    composition stays consistent without duplicating the env-var check.

    The name avoids a leading ``test_`` so pytest does not try to
    collect callsites in test modules as test functions.

    Examples
    --------
    >>> # PYAUTO_TEST_MODE unset
    >>> with_test_mode_segment(Path("output")) / "results_folder"
    PosixPath('output/results_folder')

    >>> # PYAUTO_TEST_MODE=2
    >>> with_test_mode_segment(Path("output")) / "results_folder"
    PosixPath('output/test_mode/results_folder')
    """
    return base / "test_mode" if is_test_mode() else base
