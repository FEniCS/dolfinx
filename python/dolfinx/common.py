# Copyright (C) 2018 Michal Habera
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""General tools for timing and configuration."""

from mpi4py import MPI as _MPI

import functools
import typing

from dolfinx import cpp as _cpp
from dolfinx.cpp.common import has_adios2  # noqa
from dolfinx.cpp.common import (IndexMap, git_commit_hash, has_debug,  # noqa
                                has_kahip, has_parmetis)

__all__ = ["IndexMap", "Timer", "timed", "list_timings", "timing", "TimingType", "Reduction"]

TimingType = _cpp.common.TimingType
Reduction = _cpp.common.Reduction


def timing(task: str):
    """Return timing (count, total wall time, total user time, total system time) for given task."""
    return _cpp.common.timing(task)


def list_timings(comm: _MPI.Comm, timing_types, reduction=Reduction.max):
    """Print out a summary of all Timer measurements.

    One can specify the timing types as a subset of wall time, system time or user time.

    Note:
        When used in parallel, a reduction is applied across all processes. By default, the maximum time is shown.

    Args:
        comm: Communicator to reduce timings over
        timing_types: List of timings to return
        reduction: Reduction type over MPI communicator
    """
    _cpp.common.list_timings(comm, timing_types, reduction)


class Timer:
    """A timer can be used for timing tasks. The basic usage is::

        with Timer('Some costly operation'):
            costly_call_1()
            costly_call_2()

    or::

        with Timer() as t:
            costly_call_1()
            costly_call_2()
            print(f'Elapsed time so far: {t.elapsed()[0]}')

    The timer is started when entering context manager and timing
    ends when exiting it. It is also possible to start and stop a
    timer explicitly by::

        t = Timer('Some costly operation')
        t.start()
        costly_call()
        t.stop()

    and retrieve timing data using::

        t.elapsed()

    Timings are stored globally (if task name is given) and
    may be printed using functions :func:`timing`, or :func:`list_timings`::

        list_timings(comm, [TimingType.wall, TimingType.user])
    """

    def __init__(self, name: typing.Optional[str] = None):
        if name is None:
            self._cpp_object = _cpp.common.Timer()
        else:
            self._cpp_object = _cpp.common.Timer(name)

    def __enter__(self):
        self._cpp_object.start()
        return self

    def __exit__(self, *args):
        self._cpp_object.stop()

    def start(self):
        self._cpp_object.start()

    def stop(self):
        return self._cpp_object.stop()

    def resume(self):
        self._cpp_object.resume()

    def elapsed(self):
        return self._cpp_object.elapsed()


def timed(task: str):
    """Decorator for timing functions. The basic usage is::

        @timed('Description of function')
        def f(x):
            return x[0]

    and obtained by::

        f([[0]],[1], [2]])
        print(timing('Description of function'))
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with Timer(task):
                return func(*args, **kwargs)

        return wrapper

    return decorator
