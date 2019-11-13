"""Unit tests for MPI facilities"""

# Copyright (C) 2017 Garth N. Wells
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import dolfinx
from mpi4py import MPI


def test_mpi_comm_wrapper():
    """
    Test MPICommWrapper <-> mpi4py.MPI.Comm conversion
    """
    w1 = MPI.COMM_WORLD

    m = dolfinx.UnitSquareMesh(w1, 4, 4)
    w2 = m.mpi_comm()

    assert isinstance(w1, MPI.Comm)
    assert isinstance(w2, MPI.Comm)
