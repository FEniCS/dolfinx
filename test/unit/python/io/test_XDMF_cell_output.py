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

from __future__ import print_function
import pytest
from dolfin import *
from dolfin_utils.test import *

ghost_mode = set_parameters_fixture("ghost_mode", ["shared_vertex", "none"])

@skip_if_not_HDF5
def test_xdmf_cell_scalar_ghost(cd_tempdir, ghost_mode):
    n = 8
    mesh = UnitSquareMesh(n, n)
    Q = FunctionSpace(mesh, "DG", 0)
    F = Function(Q)
    E = Expression("x[0]")
    F.interpolate(E)

    xdmf = XDMFFile(mesh.mpi_comm(), "dg0.xdmf")
    xdmf << F
    del xdmf

    hdf = HDF5File(mesh.mpi_comm(), "dg0.h5", "r")
    vec = Vector()
    hdf.read(vec, "/VisualisationVector/0", False)
    del hdf

    area = MPI.sum(mesh.mpi_comm(), sum(vec.array()))
    assert abs(n*n - area) < 1e-9
