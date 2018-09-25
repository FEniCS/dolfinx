# -*- coding: utf-8 -*-
# Copyright (C) 2018 Michal Habera
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import functools

from dolfin import cpp

# FIXME: replace possibly with numpy.finfo
DOLFIN_EPS = cpp.common.DOLFIN_EPS

has_debug = cpp.common.has_debug()
has_hdf5 = cpp.common.has_hdf5()
has_hdf5_parallel = cpp.common.has_hdf5_parallel()
has_mpi4py = cpp.common.has_mpi4py()
has_parmetis = cpp.common.has_parmetis()
has_scotch = cpp.common.has_scotch()
has_petsc_complex = cpp.common.has_petsc_complex()
has_slepc = cpp.common.has_slepc()
has_petsc4py = cpp.common.has_petsc4py()
has_slepc4py = cpp.common.has_slepc4py()

git_commit_hash = cpp.common.git_commit_hash()

TimingType = cpp.common.TimingType


def timing(task: str):
    return cpp.common.timing(task)


def timings(timing_types: list):
    return cpp.common.timings(timing_types)


def list_timings(timing_types: list):
    return cpp.common.list_timings(timing_types)


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

        t.start()
        t.stop()

    and retrieve timing data using::

        t.elapsed()

    Timings are stored globally (if task name is given) and
    may be printed using functions ``timing``, ``timings``,
    ``list_timings``, ``dump_timings_to_xml``, e.g.::

        list_timings([TimingType.wall, TimingType.user])
    """
    def __init__(self, name: str = None):
        if name is None:
            self._cpp_object = cpp.common.Timer()
        else:
            self._cpp_object = cpp.common.Timer(name)

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
    """
    Decorator for timing functions. Usage::

    @timed(\"Do Foo\")
    def do_foo(*args, **kwargs):
        # Do something costly
        pass

    do_foo()
    list_timings([TimingType.wall, TimingType.user])

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


if has_mpi4py:

    class MPICommWrapper:
        """DOLFIN is compiled without support for mpi4py. This object
        can be passed into DOLFIN as an MPI communicator, but is not
        an mpi4py comm.
        """

        def underlying_comm(self):
            return cpp.MPICommWrapper.underlying_comm()


class MPI:
    """MPI parallel communication wrapper
    """

    comm_world = cpp.MPI.comm_world
    comm_self = cpp.MPI.comm_self
    comm_null = cpp.MPI.comm_null

    @staticmethod
    def init():
        """Initilize MPI
        """
        cpp.MPI.init()

    @staticmethod
    def init_level(args: list, required_thread_level: int):
        """Initialise MPI with command-line args and required level
        of thread support.
        """
        cpp.MPI.init(args, required_thread_level)

    @staticmethod
    def responsible():
        return cpp.MPI.responsible()

    @staticmethod
    def initialized():
        return cpp.MPI.initialized()

    @staticmethod
    def finalized():
        return cpp.MPI.finalized()

    @staticmethod
    def barrier(comm: MPICommWrapper):
        return cpp.MPI.barrier(comm)

    @staticmethod
    def rank(comm: MPICommWrapper):
        return cpp.MPI.rank(comm)

    @staticmethod
    def size(comm: MPICommWrapper):
        return cpp.MPI.size(comm)

    @staticmethod
    def local_range(comm: MPICommWrapper, N: int):
        return cpp.MPI.local_range(comm, N)

    @staticmethod
    def max(comm: MPICommWrapper, value: float):
        return cpp.MPI.max(comm, value)

    @staticmethod
    def min(comm: MPICommWrapper, value: float):
        return cpp.MPI.min(comm, value)

    @staticmethod
    def sum(comm: MPICommWrapper, value: float):
        return cpp.MPI.sum(comm, value)

    @staticmethod
    def avg(comm: MPICommWrapper, value):
        return cpp.MPI.avs(comm, value)
