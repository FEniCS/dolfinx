#!/usr/bin/env py.test

"Unit tests for BoundaryMesh library"

# Copyright (C) 2012 Garth N. Wells
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
#
# Modified by Oeyvind Evju 2013
#
# First added:  2011-10-09
# Last changed: 2013-10-11

import pytest
import numpy
from dolfin import *


def test_1D_mesh():
    mesh = UnitIntervalMesh(32)

    # Create global boundary mesh
    bmesh1 = BoundaryMesh(mesh, "exterior")
    assert MPI.sum(mesh.mpi_comm(), bmesh1.num_cells()) == 2
    assert bmesh1.size_global(0) == 2
    assert bmesh1.topology().dim() == 0


def test_2D_mesh():
    mesh = UnitSquareMesh(8, 8)

    # Create global boundary mesh
    bmesh1 = BoundaryMesh(mesh, "exterior")
    assert MPI.sum(mesh.mpi_comm(), bmesh1.num_cells()) == 4*8
    assert bmesh1.size_global(1) == 4*8
    assert bmesh1.topology().dim() == 1


def test_3D_mesh():
    mesh = UnitCubeMesh(8, 8, 8)

    # Create global boundary mesh
    bmesh1 = BoundaryMesh(mesh, "exterior")
    assert MPI.sum(mesh.mpi_comm(), bmesh1.num_cells()) == 6*8*8*2
    assert bmesh1.size_global(2) == 6*8*8*2
    assert bmesh1.topology().dim() == 2
