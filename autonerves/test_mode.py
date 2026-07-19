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


def test_mode_samples():
    """
    Return the number of fake samples the test-mode sampler bypass writes.

    Controlled by ``PYAUTO_TEST_MODE_SAMPLES`` (default 4, the historical
    bypass sample count — unset behaviour is unchanged). Larger values make
    a ``PYAUTO_TEST_MODE=2``/``3`` bypass run write a ``samples.csv`` whose
    row count and byte size are representative of a production sampler run
    (N ~ 10k-100k), so resume/load timings measured against the output are
    honest while the fit itself completes in seconds.

    Values below 4 raise: 4 is the minimum that preserves the bypass
    sample structure downstream code is tested against.
    """
    samples = int(os.environ.get("PYAUTO_TEST_MODE_SAMPLES", "4"))
    if samples < 4:
        raise ValueError(
            f"PYAUTO_TEST_MODE_SAMPLES must be >= 4 (got {samples}) — 4 is "
            f"the minimum that preserves the test-mode bypass sample "
            f"structure."
        )
    return samples


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


def latent_nan_inject_spec():
    """
    Return the ``PYAUTO_LATENT_NAN_INJECT`` spec string, or ``None``.

    This is a **test-only** knob used by the ``*_workspace_test`` integration
    suites to deliberately poison latent-variable values with NaNs in an
    arbitrary per-sample pattern, reproducing the failure mode where
    ``compute_latent_samples`` produced ``Sample`` objects with inconsistent
    key sets (see ``autofit/non_linear/analysis/analysis.py``). It is
    ``None`` (a no-op) in every normal/production run because the env var
    is unset.

    Supported spec, used by :func:`inject_latent_nans`:

    - ``"stride:N"`` — set NaN on column 0 of every sample whose **absolute**
      index (across the full sample list, not the per-batch index) is a
      non-zero multiple of ``N``. Using absolute indices makes the NaN
      pattern straddle batch boundaries, which is the exact condition
      needed to surface per-batch (rather than global) masking bugs.

      Index 0 is deliberately **never** poisoned: with the latent batch
      size ``B`` chosen so that ``N >= B``, batch 0 (absolute indices
      ``0 .. B-1``) is left fully finite, so it produces samples with the
      *complete* latent key set and seeds the latent ``Samples`` model with
      all keys. A later batch then loses column 0 (under the buggy per-batch
      JAX column mask), giving it a *reduced* key set — the mismatch that
      raises ``KeyError`` in ``Samples.summary()``. Were index 0 poisoned,
      batch 0 would itself be reduced and the mismatch would not surface.
    """
    return os.environ.get("PYAUTO_LATENT_NAN_INJECT") or None


def inject_latent_nans(values_2d, start_index):
    """
    Apply :func:`latent_nan_inject_spec` to a materialised
    ``(n_samples, n_latents)`` latent-value array.

    Parameters
    ----------
    values_2d
        A NumPy or JAX array of latent values for one batch, shape
        ``(n_samples_in_batch, n_latents)``.
    start_index
        The absolute index of row 0 of ``values_2d`` within the full sample
        list (i.e. the batch offset). NaNs are placed using
        ``start_index + local_row`` so the pattern is consistent across
        batch boundaries.

    Returns
    -------
    The array with NaNs injected per the active spec. When the spec is unset
    or unrecognised the input is returned unchanged (true no-op).

    Notes
    -----
    Handles both NumPy and JAX arrays. For JAX the functional ``.at[...].set``
    update is used (``jax.numpy`` imported locally, matching the library's
    convention of never importing JAX at module scope).
    """
    spec = latent_nan_inject_spec()
    if not spec:
        return values_2d

    if not spec.startswith("stride:"):
        return values_2d

    try:
        stride = int(spec.split(":", 1)[1])
    except (ValueError, IndexError):
        return values_2d
    if stride <= 0:
        return values_2d

    n_rows = values_2d.shape[0]
    nan_rows = [
        r
        for r in range(n_rows)
        if (start_index + r) != 0 and (start_index + r) % stride == 0
    ]
    if not nan_rows:
        return values_2d

    # JAX arrays expose ``.at`` for functional updates; NumPy arrays do not.
    if hasattr(values_2d, "at"):
        import jax.numpy as jnp

        out = values_2d
        for r in nan_rows:
            out = out.at[r, 0].set(jnp.nan)
        return out

    import numpy as np

    out = np.array(values_2d, copy=True)
    for r in nan_rows:
        out[r, 0] = np.nan
    return out


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
