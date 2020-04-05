# Copyright (C) 2018 Michal Habera
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from dolfinx import cpp

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


def rank(comm):
    return cpp.MPI.rank(comm)


def size(comm):
    return cpp.MPI.size(comm)


def local_range(rank: int, N: int, size: int):
    return cpp.MPI.local_range(rank, N, size)
