# Copyright (C) 2018 Michal Habera
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""General tools for timing and configuration"""

import functools

from dolfinx import cpp as _cpp
from dolfinx.cpp.common import (IndexMap, git_commit_hash, has_adios2,  # noqa
                                has_debug, has_kahip, has_parmetis)

__all__ = ["IndexMap", "Timer", "timed"]

TimingType = _cpp.common.TimingType


def timing(task: str):
    return _cpp.common.timing(task)


def list_timings(comm, timing_types: list):
    return _cpp.common.list_timings(comm, timing_types)


class Timer:
    """A timer can be used for timing tasks. The basic usage is::

        with Timer(\"Some costly operation\"):
            costly_call_1()
            costly_call_2()

    or::

        with Timer() as t:
            costly_call_1()
            costly_call_2()
            print(\"Ellapsed time so far: %s\" % t.elapsed()[0])

    The timer is started when entering context manager and timing
    ends when exiting it. It is also possible to start and stop a
    timer explicitly by::

        t = Timer(\"Some costly operation\")
        t.start()
        costly_call()
        t.stop()

    and retrieve timing data using::

        t.elapsed()

    Timings are stored globally (if task name is given) and
    may be printed using functions ``timing``, ``timings``,
    ``list_timings``, ``dump_timings_to_xml``, e.g.::

        list_timings(comm, [TimingType.wall, TimingType.user])
    """

    def __init__(self, name: str = None):
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
    """Decorator for timing functions."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with Timer(task):
                return func(*args, **kwargs)

        return wrapper

    return decorator
