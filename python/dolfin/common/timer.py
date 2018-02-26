# Copyright (C) 2017 Jan Blechta
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import functools
from dolfin import cpp

__all__ = ["Timer", "timed"]


class Timer(cpp.common.Timer):
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

        t.start()
        t.stop()

    and retrieve timing data using::

        t.elapsed()

    Timings are stored globally (if task name is given) and
    may be printed using functions ``timing``, ``timings``,
    ``list_timings``, ``dump_timings_to_xml``, e.g.::

        list_timings(TimingClear.keep, [TimingType.wall, TimingType.user])
    """

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


def timed(task):
    """Decorator for timing functions. Usage::

        @timed(\"Do Foo\")
        def do_foo(*args, **kwargs):
            # Do something costly
            pass

        do_foo()

        list_timings(TimingClear.keep, [TimingType.wall, TimingType.user])

        t = timing(\"Do Foo\", TimingClear.clear)
        print("Do foo wall time: %s" % t[1])
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with Timer(task):
                return func(*args, **kwargs)
        return wrapper
    return decorator
