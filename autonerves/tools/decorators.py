import functools

import numpy as np


class CachedProperty(object):
    """
    A property that is only computed once per instance and then replaces
    itself with an ordinary attribute. Deleting the attribute resets the
    property.

    Source: https://github.com/bottlepy/bottle/commit/fa7733e075da0d790d809aa3d2f53071897e6f76
    """

    def __init__(self, func):
        self.func = func

    def __get__(self, obj, cls):
        if obj is None:
            return self
        if self.func.__name__ not in obj.__dict__:
            obj.__dict__[self.func.__name__] = self.func(obj)
        return obj.__dict__[self.func.__name__]


cached_property = CachedProperty


def cached_property_names(cls) -> frozenset:
    """
    Return the names of every ``cached_property``-style descriptor declared
    anywhere in ``cls``'s MRO.

    Recognises both stdlib :class:`functools.cached_property` and the
    autonerves :class:`CachedProperty` wrapper above. Walks the MRO so
    descriptors declared on base classes are picked up.

    The first call for a given class walks the MRO and caches the resulting
    frozenset on the class itself under ``__cached_property_names_cache__``;
    subsequent calls return the cached frozenset directly. The cache key is
    stored under a dunder name so the result itself never appears in any
    instance ``__dict__`` walk.

    Used by PyAutoFit and PyAutoArray to extend their existing
    ``__dict__``-iteration filters with a forward-compat guard: any future
    ``@cached_property`` declared on a model or Fit class will be
    automatically excluded from instance construction, ``ModelInstance.dict``,
    pickling, and JAX pytree flattening, preventing the class of bug that
    PR PyAutoFit#1300 fixed for ``parameterization``.

    Parameters
    ----------
    cls
        The class to inspect.

    Returns
    -------
    A frozenset of attribute names corresponding to ``cached_property`` or
    autonerves ``CachedProperty`` descriptors found in ``cls.__mro__``.
    """
    cache = cls.__dict__.get("__cached_property_names_cache__")
    if cache is not None:
        return cache

    names = set()
    for base in cls.__mro__:
        for attr_name, value in base.__dict__.items():
            if isinstance(value, (functools.cached_property, CachedProperty)):
                names.add(attr_name)

    result = frozenset(names)
    # Stash on the class itself (not a parent) so subclass overrides
    # of cached_property descriptors are re-discovered on the subclass.
    try:
        setattr(cls, "__cached_property_names_cache__", result)
    except (TypeError, AttributeError):
        # Some classes (slotted, built-in) reject setattr; that's fine,
        # the function still returns the correct value without memoisation.
        pass
    return result
