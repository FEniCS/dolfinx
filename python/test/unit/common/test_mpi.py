"""Unit tests for MPI facilities"""

# Copyright (C) 2017 Garth N. Wells
#
# This file is part of DOLFIN.
#
# DOLFIN is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DOLFIN is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.

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
