# Copyright (C) 2018 Michal Habera
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""General tools for timing and configuration."""

import datetime
import functools
import typing

from dolfinx import cpp as _cpp
from dolfinx.cpp.common import (
    IndexMap,
    git_commit_hash,
    has_adios2,
    has_complex_ufcx_kernels,
    has_debug,
    has_kahip,
    has_parmetis,
    has_petsc,
    has_petsc4py,
    has_ptscotch,
    has_slepc,
    ufcx_signature,
)

__all__ = [
    "IndexMap",
    "Timer",
    "timed",
    "git_commit_hash",
    "has_adios2",
    "has_complex_ufcx_kernels",
    "has_debug",
    "has_kahip",
    "has_parmetis",
    "has_petsc",
    "has_petsc4py",
    "has_ptscotch",
    "has_slepc",
    "ufcx_signature",
]

Reduction = _cpp.common.Reduction


def timing(task: str) -> tuple[int, datetime.timedelta]:
    """Return the logged elapsed time.

    Timing data is for the calling process.

    Arguments:
        task: The task name using when logging the time.

    Returns:
        (number of times logged, total wall time)
    """
    return _cpp.common.timing(task)


def list_timings(comm, reduction=Reduction.max):
    """Print out a summary of all Timer measurements.

    When used in parallel, a reduction is applied across all processes.
    By default, the maximum time is shown.
    """
    _cpp.common.list_timings(comm, reduction)


class Timer:
    """A timer for timing section of code.

    The recommended usage is with a context manager.

    Example:
        With a context manager, the timer is started when entering
        and stopped at exit. With a named ``Timer``::

            with Timer(\"Some costly operation\"):
                costly_call_1()
                costly_call_2()

            delta = timing(\"Some costly operation\")
            print(delta)

        or with an un-named ``Timer``::

            with Timer() as t:
                costly_call_1()
                costly_call_2()
                print(f\"Elapsed time: {t.elapsed()}\")

    Example:
        It is possible to start and stop a timer explicitly::

            t = Timer(\"Some costly operation\")
            costly_call()
            delta = t.stop()

        and retrieve timing data using::

            delta = t.elapsed()

        To flush the timing data for a named ``Timer`` to the logger, the
        timer should be stopped and flushed::

            t.stop()
            t.flush()

    Timings are stored globally (if task name is given) and once flushed
    (if used without a context manager) may be printed using functions
    ``timing`` and ``list_timings``, e.g.::

        list_timings(comm)
    """

    _cpp_object: _cpp.common.Timer

    def __init__(self, name: typing.Optional[str] = None):
        """Create timer.

        Args:
            name: Identifier to use when storing elapsed time in logger.
        """
        self._cpp_object = _cpp.common.Timer(name)

    def __enter__(self):
        self._cpp_object.start()
        return self

    def __exit__(self, *args):
        self._cpp_object.stop()
        self._cpp_object.flush()

    def start(self) -> None:
        """Reset elapsed time and (re-)start timer."""
        self._cpp_object.start()

    def stop(self) -> datetime.timedelta:
        """Stop timer and return elapsed time.

        Returns:
            Elapsed time.
        """
        return self._cpp_object.stop()

    def resume(self) -> None:
        """Resume timer."""
        self._cpp_object.resume()

    def elapsed(self) -> datetime.timedelta:
        """Return elapsed time.

        Returns:
            Elapsed time.
        """
        return self._cpp_object.elapsed()

    def flush(self) -> None:
        """Flush timer duration to the logger.

        Note:
            Timer must have been stopped before flushing.

            Timer can be flushed only once. Subsequent calls will have
            no effect.
        """
        self._cpp_object.flush()


def timed(task: str):
    """Decorator for timing functions."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with Timer(task):
                return func(*args, **kwargs)

        return wrapper

    return decorator
