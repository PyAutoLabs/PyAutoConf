"""Tests for autonerves.tools.decorators.cached_property_names — the
MRO-walking helper used by PyAutoFit and PyAutoArray to extend their
__dict__-iteration filters with a forward-compat guard against
cached_property pytree/dict leaks."""

import functools

import pytest

from autonerves.tools.decorators import (
    CachedProperty,
    cached_property,
    cached_property_names,
)


def test_empty_class_returns_empty_frozenset():
    class Empty:
        pass

    result = cached_property_names(Empty)
    assert result == frozenset()
    assert isinstance(result, frozenset)


def test_stdlib_cached_property_is_recognised():
    class Container:
        @functools.cached_property
        def derived(self):
            return "computed"

    assert cached_property_names(Container) == frozenset({"derived"})


def test_autonerves_cached_property_is_recognised():
    class Container:
        @CachedProperty
        def derived(self):
            return "computed"

    assert cached_property_names(Container) == frozenset({"derived"})


def test_autonerves_cached_property_alias_is_recognised():
    """``cached_property`` re-exported from autonerves.tools.decorators is
    the same object as ``CachedProperty`` — verifies both spellings work."""

    class Container:
        @cached_property
        def derived(self):
            return "computed"

    assert cached_property_names(Container) == frozenset({"derived"})


def test_mro_walk_picks_up_base_class_descriptors():
    class Base:
        @functools.cached_property
        def base_value(self):
            return 1

    class Mid(Base):
        @functools.cached_property
        def mid_value(self):
            return 2

    class Leaf(Mid):
        @functools.cached_property
        def leaf_value(self):
            return 3

    assert cached_property_names(Leaf) == frozenset(
        {"base_value", "mid_value", "leaf_value"}
    )
    assert cached_property_names(Mid) == frozenset({"base_value", "mid_value"})
    assert cached_property_names(Base) == frozenset({"base_value"})


def test_mixed_stdlib_and_autonerves_descriptors_both_recognised():
    class Container:
        @functools.cached_property
        def stdlib_value(self):
            return 1

        @CachedProperty
        def autonerves_value(self):
            return 2

    assert cached_property_names(Container) == frozenset(
        {"stdlib_value", "autonerves_value"}
    )


def test_regular_property_is_ignored():
    class Container:
        @property
        def regular(self):
            return "live"

        @functools.cached_property
        def cached(self):
            return "memoised"

    assert cached_property_names(Container) == frozenset({"cached"})


def test_result_is_memoised_on_the_class():
    class Container:
        @functools.cached_property
        def derived(self):
            return "computed"

    first = cached_property_names(Container)
    # The cache lives on the class itself, not a parent
    assert "__cached_property_names_cache__" in Container.__dict__
    cached = Container.__dict__["__cached_property_names_cache__"]
    assert cached is first

    # Subsequent calls return the same object identity
    second = cached_property_names(Container)
    assert second is first


def test_subclass_gets_its_own_cache_entry():
    class Base:
        @functools.cached_property
        def base_value(self):
            return 1

    class Sub(Base):
        @functools.cached_property
        def sub_value(self):
            return 2

    base_result = cached_property_names(Base)
    sub_result = cached_property_names(Sub)

    assert base_result == frozenset({"base_value"})
    assert sub_result == frozenset({"base_value", "sub_value"})

    # Each class owns its own cache entry — subclass cache does not
    # leak into the parent.
    assert (
        Sub.__dict__["__cached_property_names_cache__"]
        is not Base.__dict__["__cached_property_names_cache__"]
    )
