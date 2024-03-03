# Copyright (C) 2017 Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for MPI facilities"""

import sys

from mpi4py import MPI

from dolfinx.mesh import create_unit_square


def test_mpi_comm_wrapper():
    """Test MPICommWrapper <-> mpi4py.MPI.Comm conversion"""
    comm0 = MPI.COMM_WORLD
    m = create_unit_square(comm0, 4, 4)
    comm1 = m.comm
    assert isinstance(comm0, MPI.Comm)
    assert isinstance(comm1, MPI.Comm)


def test_mpi_comm_refcount():
    """Test MPICommWrapper <-> mpi4py.MPI.Comm reference counting"""
    comm0 = MPI.COMM_WORLD
    m = create_unit_square(comm0, 4, 4)
    comm1 = m.comm
    assert comm1 != comm0
    comm2 = m.comm
    assert comm2 == comm1

    del m
    assert sys.getrefcount(comm1) == 2
    assert comm1.rank == comm0.rank
