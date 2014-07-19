"""Unit tests for the CollisionDetection class"""

# Copyright (C) 2014 Anders Logg and August Johansson
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
# First added:  2014-02-16
# Last changed: 2014-05-30

import unittest
from dolfin import *

def create_triangular_mesh_3D():
    editor = MeshEditor()
    mesh = Mesh()
    editor.open(mesh,2,3)
    editor.init_cells(2)
    editor.init_vertices(4)
    editor.add_cell(0, 0,1,2)
    editor.add_cell(1, 1,2,3)
    editor.add_vertex(0, 0,0,0.5)
    editor.add_vertex(1, 1,0,0.5)
    editor.add_vertex(2, 0,1,0.5)
    editor.add_vertex(3, 1,1,0.5)
    editor.close()
    return mesh;

@unittest.skipIf(MPI.size(mpi_comm_world()) > 1, "Skipping unit test(s) not working in parallel")
class IntervalTest(unittest.TestCase):
    "Test class for collision with interval"

    def test_collides_point(self):

        mesh = UnitIntervalMesh(1)
        cell = Cell(mesh, 0)

        self.assertEqual(cell.collides(Point(0.5)), True)
        self.assertEqual(cell.collides(Point(1.5)), False)

@unittest.skipIf(MPI.size(mpi_comm_world()) > 1, "Skipping unit test(s) not working in parallel")
class TriangleTest(unittest.TestCase):
    "Test class for collision with triangle"

    def test_collides_point(self):

        mesh = UnitSquareMesh(1, 1)
        cell = Cell(mesh, 0)

        self.assertEqual(cell.collides(Point(0.5)), True)
        self.assertEqual(cell.collides(Point(1.5)), False)

    def test_collides_triangle(self):

        m0 = UnitSquareMesh(8, 8)
        c0 = Cell(m0, 0)

        m1 = UnitSquareMesh(8, 8)
        m1.translate(Point(0.1, 0.1))
        c1 = Cell(m1, 0)
        c2 = Cell(m1, 1)

        self.assertEqual(c0.collides(c0), True)
        self.assertEqual(c0.collides(c1), True)
        # self.assertEqual(c0.collides(c2), False) # touching edges
        self.assertEqual(c1.collides(c0), True)
        self.assertEqual(c1.collides(c1), True)
        self.assertEqual(c1.collides(c2), False)
        # self.assertEqual(c2.collides(c0), False) # touching edges
        self.assertEqual(c2.collides(c1), False)
        self.assertEqual(c2.collides(c2), True)

@unittest.skipIf(MPI.size(mpi_comm_world()) > 1, "Skipping unit test(s) not working in parallel")
class TetrahedronTest(unittest.TestCase):
    "Test class for collision with tetrahedron"

    def test_collides_point(self):

        mesh = UnitCubeMesh(1, 1, 1)
        cell = Cell(mesh, 0)

        self.assertEqual(cell.collides(Point(0.5)), True)
        self.assertEqual(cell.collides(Point(1.5)), False)

    def test_collides_triangle(self):

        tetmesh = UnitCubeMesh(2, 2, 2)
        trimesh = create_triangular_mesh_3D()
        dx = Point(0.1, 0.1, -0.1)
        trimesh_shift = create_triangular_mesh_3D()
        trimesh_shift.translate(dx)

        tet0 = Cell(tetmesh, 18)
        tet1 = Cell(tetmesh, 19)
        tri0 = Cell(trimesh, 1)
        tri0shift = Cell(trimesh_shift, 1)

        # proper intersection
        self.assertEqual(tet0.collides(tri0shift), True)
        self.assertEqual(tri0shift.collides(tet0), True)
        self.assertEqual(tet1.collides(tri0shift), True)
        self.assertEqual(tri0shift.collides(tet1), True)

        # point touch
        self.assertEqual(tet0.collides(tri0), True)
        self.assertEqual(tri0.collides(tet0), True)

        # face alignment (true or false)
        self.assertEqual(tet1.collides(tri0), True)
        self.assertEqual(tri0.collides(tet1), True)

    def test_collides_tetrahedron(self):

        m0 = UnitCubeMesh(2, 2, 2)
        c19 = Cell(m0, 19)
        c26 = Cell(m0, 26)
        c37 = Cell(m0, 37)
        c43 = Cell(m0, 43)
        c45 = Cell(m0, 45)

        m1 = UnitCubeMesh(1,1,1)
        m1.translate(Point(0.5, 0.5, 0.5))
        c3 = Cell(m0, 3)
        c5 = Cell(m1, 5)

        # self collisions
        self.assertEqual(c3.collides(c3), True)
        self.assertEqual(c45.collides(c45), True)

        # standard collisions
        self.assertEqual(c3.collides(c37), True)
        self.assertEqual(c37.collides(c3), True)

        # edge face
        self.assertEqual(c5.collides(c45), True)
        self.assertEqual(c45.collides(c5), True)

        # touching edges
        self.assertEqual(c5.collides(c19), False)
        self.assertEqual(c19.collides(c5), False)

        # touching faces
        self.assertEqual(c3.collides(c43), True)
        self.assertEqual(c43.collides(c3), True)

if __name__ == "__main__":
        unittest.main()
