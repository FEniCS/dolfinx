# Copyright (C) 2018 Michal Habera
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from dolfin import cpp

comm_world = cpp.MPI.comm_world
comm_self = cpp.MPI.comm_self
comm_null = cpp.MPI.comm_null


def init():
    """Initilize MPI
    """
    cpp.MPI.init()


def init_level(args: list, required_thread_level: int):
    """Initialise MPI with command-line args and required level
    of thread support.
    """
    cpp.MPI.init(args, required_thread_level)


def responsible():
    return cpp.MPI.responsible()


def initialized():
    return cpp.MPI.initialized()


def finalized():
    return cpp.MPI.finalized()


def barrier(comm):
    return cpp.MPI.barrier(comm)


def rank(comm):
    return cpp.MPI.rank(comm)


def size(comm):
    return cpp.MPI.size(comm)


def local_range(comm, N: int):
    return cpp.MPI.local_range(comm, N)


def max(comm, value: float):
    return cpp.MPI.max(comm, value)


def min(comm, value: float):
    return cpp.MPI.min(comm, value)


def sum(comm, value: float):
    return cpp.MPI.sum(comm, value)


def avg(comm, value):
    return cpp.MPI.avs(comm, value)
