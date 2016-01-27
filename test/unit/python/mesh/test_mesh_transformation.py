#!/usr/bin/env py.test

"Unit tests for the mesh library"

# Copyright (C) 2012 Anders Logg
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

from dolfin import *


def test_translate_2d():
    mesh = UnitSquareMesh(8, 8)
    p = Point(1, 2)
    mesh.translate(p)


def test_translate_3d():
    mesh = UnitCubeMesh(8, 8, 8)
    p = Point(1, 2, 3)
    mesh.translate(p)


def test_rotate_2d():
    mesh = UnitSquareMesh(8, 8)
    p = Point(1, 2)
    mesh.rotate(10)
    mesh.rotate(10, 2)
    mesh.rotate(10, 2, p)


def test_rotate_3d():
    mesh = UnitCubeMesh(8, 8, 8)
    p = Point(1, 2, 3)
    mesh.rotate(30, 0)
    mesh.rotate(30, 1)
    mesh.rotate(30, 2)
    mesh.rotate(30, 0, p)


def test_rescale_2d():
    mesh = UnitSquareMesh(8, 8)
    p = Point(4, 4)
    s = 1.5
    MeshTransformation.rescale(mesh, s, p)
    comm = mesh.mpi_comm()
    assert MPI.sum(comm, sum(c.volume() for c in cells(mesh))) == s*s


def test_rescale_3d():
    mesh = UnitCubeMesh(8, 8, 8)
    p = Point(4, 4, 4)
    s = 1.5
    MeshTransformation.rescale(mesh, s, p)
    comm = mesh.mpi_comm()
    assert MPI.sum(comm, sum(c.volume() for c in cells(mesh))) == s*s*s
