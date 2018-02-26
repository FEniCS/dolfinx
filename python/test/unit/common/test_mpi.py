"""Unit tests for MPI facilities"""

# Copyright (C) 2017 Garth N. Wells
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import dolfin
from dolfin import MPI
from dolfin_utils.test import skip_if_not_petsc4py


def test_mpi_comm_wrapper():
    """
    Test MPICommWrapper <-> mpi4py.MPI.Comm conversion
    """
    if dolfin.has_mpi4py():
        from mpi4py import MPI
        w1 = MPI.COMM_WORLD
    else:
        w1 = dolfin.MPI.comm_world

    m = dolfin.UnitSquareMesh(w1, 4, 4)
    w2 = m.mpi_comm()

    if dolfin.has_mpi4py():
        assert isinstance(w1, MPI.Comm)
        assert isinstance(w2, MPI.Comm)
    else:
        assert isinstance(w1, dolfin.cpp.MPICommWrapper)
        assert isinstance(w2, dolfin.cpp.MPICommWrapper)
