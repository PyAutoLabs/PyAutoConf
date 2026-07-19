"""Tests for autonerves.test_mode helpers — focused on
``with_test_mode_segment`` since the other helpers (``is_test_mode``,
``skip_fit_output``, etc.) are exercised by PyAutoFit/PyAutoArray
integration tests downstream."""

import os
from pathlib import Path

import pytest

from autonerves.test_mode import (
    is_test_mode,
    test_mode_level,
    test_mode_samples,
    with_test_mode_segment,
)


@pytest.fixture(autouse=True)
def _restore_test_mode_env():
    """Save/restore PYAUTO_TEST_MODE around each test so a failure
    can't leak state to its neighbours."""

    saved = os.environ.get("PYAUTO_TEST_MODE")
    yield
    if saved is None:
        os.environ.pop("PYAUTO_TEST_MODE", None)
    else:
        os.environ["PYAUTO_TEST_MODE"] = saved


def test_with_test_mode_segment__env_unset_returns_base_unchanged():
    os.environ.pop("PYAUTO_TEST_MODE", None)
    assert with_test_mode_segment(Path("output")) == Path("output")


def test_with_test_mode_segment__env_zero_returns_base_unchanged():
    """``PYAUTO_TEST_MODE=0`` is the documented off state, so the
    helper must treat it the same as unset (no test_mode segment)."""
    os.environ["PYAUTO_TEST_MODE"] = "0"
    assert is_test_mode() is False
    assert with_test_mode_segment(Path("output")) == Path("output")


@pytest.mark.parametrize("level", ["1", "2", "3"])
def test_with_test_mode_segment__env_set_inserts_segment(level):
    os.environ["PYAUTO_TEST_MODE"] = level
    assert with_test_mode_segment(Path("output")) == Path("output") / "test_mode"


def test_with_test_mode_segment__chains_with_pathlib_concat():
    """The helper's return value must compose with further ``/`` ops so
    workspace scripts can write ``with_test_mode_segment(base) / 'name'``
    in a single line."""
    os.environ["PYAUTO_TEST_MODE"] = "2"
    composed = with_test_mode_segment(Path("output")) / "results_folder"
    assert composed == Path("output") / "test_mode" / "results_folder"


class TestTestModeSamples:
    @pytest.fixture(autouse=True)
    def _restore_samples_env(self):
        saved = os.environ.get("PYAUTO_TEST_MODE_SAMPLES")
        yield
        if saved is None:
            os.environ.pop("PYAUTO_TEST_MODE_SAMPLES", None)
        else:
            os.environ["PYAUTO_TEST_MODE_SAMPLES"] = saved

    def test__env_unset_returns_historical_default_of_four(self):
        os.environ.pop("PYAUTO_TEST_MODE_SAMPLES", None)
        assert test_mode_samples() == 4

    def test__env_set_returns_value(self):
        os.environ["PYAUTO_TEST_MODE_SAMPLES"] = "50000"
        assert test_mode_samples() == 50000

    def test__values_below_four_raise(self):
        os.environ["PYAUTO_TEST_MODE_SAMPLES"] = "3"
        with pytest.raises(ValueError):
            test_mode_samples()
