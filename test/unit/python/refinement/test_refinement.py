#!/usr/bin/env py.test

""" Unit tests for refinement """

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
#
# First added:  2011-08-23
# Last changed:

import pytest
from dolfin import *
from dolfin_utils.test import skip_in_parallel

@skip_in_parallel
def test_uniform_refine1D():
    mesh = UnitIntervalMesh(20)
    mesh = refine(mesh)
    vol2 = 0.0
    for c in cells(mesh):
        vol2 += c.volume()
    assert round(1.0 - vol2, 7) == 0.0

def test_uniform_refine2D():
    mesh = UnitSquareMesh(4, 6)
    mesh = refine(mesh)
    vol2 = 0.0
    for c in cells(mesh):
        vol2 += c.volume()
    vol2 = MPI.sum(mesh.mpi_comm(), vol2)
    assert round(1.0 - vol2, 7) == 0.0

def test_uniform_refine3D():
    mesh = UnitCubeMesh(4, 4, 6)
    mesh = refine(mesh)
    vol2 = 0.0
    for c in cells(mesh):
        vol2 += c.volume()
    vol2 = MPI.sum(mesh.mpi_comm(), vol2)
    assert round(1.0 - vol2, 7) == 0.0

def test_marker_refine2D():
    mesh = UnitSquareMesh(4, 6)
    for j in range(3):
        marker = CellFunction("bool", mesh, False)
        for c in cells(mesh):
            if (c.midpoint().x() > 0.5):
                marker[c] = True
        mesh = refine(mesh, marker)
    vol2 = 0.0
    for c in cells(mesh):
        vol2 += c.volume()
    vol2 = MPI.sum(mesh.mpi_comm(), vol2)
    assert round(1.0 - vol2, 7) == 0.0

def test_marker_refine3D():
    mesh = UnitCubeMesh(4, 4, 6)
    for j in range(3):
        marker = CellFunction("bool", mesh, False)
        for c in cells(mesh):
            if (c.midpoint().x() > 0.5):
                marker[c] = True
        mesh = refine(mesh, marker)
    vol2 = 0.0
    for c in cells(mesh):
        vol2 += c.volume()
    vol2 = MPI.sum(mesh.mpi_comm(), vol2)
    assert round(1.0 - vol2, 7) == 0.0
