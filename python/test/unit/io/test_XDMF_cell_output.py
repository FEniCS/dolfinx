"""Unit test for XDMF output of DG0"""

# Copyright (C) 2014 Chris Richardson
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

import pytest
from dolfin import *
from dolfin_utils.test import *
from dolfin.la import PETScVector

ghost_mode = set_parameters_fixture("ghost_mode", ["shared_vertex", "none"])


@skip_if_not_HDF5
@xfail_with_serial_hdf5_in_parallel
def test_xdmf_cell_scalar_ghost(cd_tempdir, ghost_mode):
    n = 8
    mesh = UnitSquareMesh(MPI.comm_world, n, n)

    # print(mesh)

    #Q = FunctionSpace(mesh, "DG", 0)
    #F = Function(Q)
    #E = Expression("x[0]", degree=1)
    # F.interpolate(E)

    # with XDMFFile(mesh.mpi_comm(), "dg0.xdmf") as xdmf:
    #    xdmf.write(F)

    # with HDF5File(mesh.mpi_comm(), "dg0.h5", "r") as hdf:
    #     vec = PETScVector(mesh.mpi_comm())
    #     hdf.read(vec, "/VisualisationVector/0", False)

    # area = MPI.sum(mesh.mpi_comm(), sum(vec.get_local()))
    # assert abs(n*n - area) < 1e-9
