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

from dolfin import mpi_comm_world, mpi_comm_self
from dolfin_utils.test import skip_if_not_petsc4py


@skip_if_not_petsc4py
def test_mpi_comm_type_petsc4py():
    import petsc4py
    assert isinstance(mpi_comm_world(), petsc4py.PETSc.Comm)
    assert isinstance(mpi_comm_self(), petsc4py.PETSc.Comm)
