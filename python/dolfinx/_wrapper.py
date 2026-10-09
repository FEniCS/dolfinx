# Copyright (C) 2026 Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Support for wrapping C++ objects in the Python interface."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

__all__ = ["cached"]

_T = TypeVar("_T")


def cached(cache: dict[int, tuple[Any, _T]], wrapper: Callable[[Any], _T], cpp_object: Any) -> _T:
    """Return the Python wrapper for a C++ object, building it once.

    Repeated calls with the same C++ object return the same wrapper, so
    wrapper identity tracks the identity of the wrapped object. The
    cache is keyed on the C++ object rather than on the accessor
    arguments, so a new wrapper is built if the C++ layer replaces the
    object it hands back.

    Note:
        For accessors that take no arguments, use
        :func:`functools.cached_property` instead.

    Note:
        Entries are keyed on ``id``, since a bound type that defines
        ``__eq__``, such as ``AdjacencyList``, is not hashable. Each
        entry holds the C++ object it is keyed on, so a cached object
        cannot be collected and have its ``id`` reused by another live
        object. The C++ object is owned by the wrapped object in any
        case, so this adds no lifetime beyond that of the cache.

    Args:
        cache: Per-instance store of ``(C++ object, wrapper)`` pairs.
        wrapper: Called with ``cpp_object`` to build a missing wrapper.
        cpp_object: C++ object to wrap.

    Returns:
        Wrapper for ``cpp_object``.
    """
    key = id(cpp_object)
    try:
        return cache[key][1]
    except KeyError:
        w = wrapper(cpp_object)
        cache[key] = (cpp_object, w)
        return w
