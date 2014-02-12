"""Unit tests for the Cell class"""

# Copyright (C) 2013 Anders Logg
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
# First added:  2013-04-18
# Last changed: 2014-01-31

import unittest
import numpy

from dolfin import *

class IntervalTest(unittest.TestCase):

    def test_collides_point(self):

        if MPI.size(mpi_comm_world()) > 1:
            return

        mesh = UnitIntervalMesh(1)
        cell = Cell(mesh, 0)

        self.assertEqual(cell.collides(Point(0.5)), True)
        self.assertEqual(cell.collides(Point(1.5)), False)

    def test_distance(self):

        if MPI.size(mpi_comm_world()) > 1: return

        mesh = UnitIntervalMesh(1)
        cell = Cell(mesh, 0)

        self.assertAlmostEqual(cell.distance(Point(-1.0)), 1.0)
        self.assertAlmostEqual(cell.distance(Point(0.5)), 0.0)

class TriangleTest(unittest.TestCase):

    def test_collides_point(self):

        if MPI.size(mpi_comm_world()) > 1: return

        mesh = UnitSquareMesh(1, 1)
        cell = Cell(mesh, 0)

        self.assertEqual(cell.collides(Point(0.5)), True)
        self.assertEqual(cell.collides(Point(1.5)), False)

    def test_collides_cell(self):

        if MPI.size(mpi_comm_world()) > 1: return

        m0 = UnitSquareMesh(8, 8)
        c0 = Cell(m0, 0)

        m1 = UnitSquareMesh(8, 8)
        m1.translate(Point(0.1, 0.1))
        c1 = Cell(m1, 0)
        c2 = Cell(m1, 1)

        # import pylab
        # plot_cell_2d(c0,pylab)
        # plot_cell_2d(c2,pylab)
        # pylab.show()

        self.assertEqual(c0.collides(c0), True)
        self.assertEqual(c0.collides(c1), True)
        self.assertEqual(c0.collides(c2), False)
        self.assertEqual(c1.collides(c0), True)
        self.assertEqual(c1.collides(c1), True)
        self.assertEqual(c1.collides(c2), False)
        self.assertEqual(c2.collides(c0), False)
        self.assertEqual(c2.collides(c1), False)
        self.assertEqual(c2.collides(c2), True)

    def test_distance(self):

        if MPI.size(mpi_comm_world()) > 1: return

        mesh = UnitSquareMesh(1, 1)
        cell = Cell(mesh, 1)

        self.assertAlmostEqual(cell.distance(Point(-1.0, -1.0)), numpy.sqrt(2))
        self.assertAlmostEqual(cell.distance(Point(-1.0, 0.5)), 1)
        self.assertAlmostEqual(cell.distance(Point(0.5, 0.5)), 0.0)

class TetrahedronTest(unittest.TestCase):

    def test_collides_point(self):

        if MPI.size(mpi_comm_world()) > 1: return

        mesh = UnitCubeMesh(1, 1, 1)
        cell = Cell(mesh, 0)

        self.assertEqual(cell.collides(Point(0.5)), True)
        self.assertEqual(cell.collides(Point(1.5)), False)

    def test_distance(self):

        if MPI.size(mpi_comm_world()) > 1: return

        mesh = UnitCubeMesh(1, 1, 1)
        cell = Cell(mesh, 5)

        self.assertAlmostEqual(cell.distance(Point(-1.0, -1.0, -1.0)), \
                               numpy.sqrt(3))
        self.assertAlmostEqual(cell.distance(Point(-1.0, 0.5, 0.5)), 1)
        self.assertAlmostEqual(cell.distance(Point(0.5, 0.5, 0.5)), 0.0)


def plot_triangulation_2d(triangulation, pylab):
    num_triangles = len(triangulation) / 6
    for i in range(len(triangulation) / 6):
        x0, y0, x1, y1, x2, y2 = triangulation[6*i:6*(i+1)]
        pylab.plot([x0, x1, x2, x0], [y0, y1, y2, y0], 'r')

def plot_cell_2d(cell, pylab):
    x = [v.point().x() for v in vertices(cell)]
    y = [v.point().y() for v in vertices(cell)]
    pylab.plot(x + [x[0]], y + [y[0]])

def plot_intersection_triangulation_2d(c0, c1):
    import pylab
    plot_cell_2d(c0, pylab)
    plot_cell_2d(c1, pylab)
    T = c0.triangulate_intersection(c1)
    plot_triangulation_2d(T, pylab)
    pylab.show()

def triangulation_to_mesh_top_2d(triangulation):
    editor = MeshEditor()
    mesh = Mesh()
    editor.open(mesh,2,3)
    num_cells = len(triangulation)/9
    num_vertices = len(triangulation)/3
    editor.init_cells(num_cells,1)
    editor.init_vertices(num_vertices,1)
    for i in xrange(num_cells):
        editor.add_cell(i, 3*i, 3*i+1, 3*i+2)
    for i in xrange(num_vertices):
        editor.add_vertex(i, triangulation[3*i], triangulation[3*i+1], triangulation[3*i+2])
    editor.close()
    return mesh


def triangulation_to_mesh_3d(triangulation):
    editor = MeshEditor()
    mesh = Mesh()
    editor.open(mesh,3,3)
    num_cells = len(triangulation)/12
    num_vertices = len(triangulation)/3
    editor.init_cells(num_cells,1)
    editor.init_vertices(num_vertices,1)
    for i in xrange(num_cells):
        editor.add_cell(i, 4*i, 4*i+1, 4*i+2, 4*i+3)
    for i in xrange(num_vertices):
        editor.add_vertex(i, triangulation[3*i], triangulation[3*i+1], triangulation[3*i+2])
    editor.close()
    return mesh


class TriangulationTest(unittest.TestCase):
    
    def test_triangulate_intersection_2d(self):

        if MPI.size(mpi_comm_world()) > 1: return

        # Create two meshes of the unit square
        mesh_0 = UnitSquareMesh(1, 1)
        mesh_1 = UnitSquareMesh(1, 1)

        # Translate second mesh
        dx = Point(-0.75, 0.75)
        mesh_1.translate(dx)

        # Extract cells
        c00 = Cell(mesh_0, 0)
        c01 = Cell(mesh_0, 1)
        c10 = Cell(mesh_1, 0)
        c11 = Cell(mesh_1, 1)

        # Compute triangulations
        cells = [c00, c01, c10, c11]
        for c0 in cells:
            for c1 in cells:
                c0.triangulate_intersection(c1)

        # For debugging and testing
        # plot_intersection_triangulation_2d(c01, c10)

    def test_triangulate_intersection_2d_3d(self):

        if MPI.size(mpi_comm_world()) > 1: return

        # Create a unit cube
        mesh_0 = UnitCubeMesh(1,1,1)

        # Create a 3D surface mesh
        editor = MeshEditor()
        mesh_1 = Mesh()
        editor.open(mesh_1,2,3)
        editor.init_cells(2,1)
        editor.init_vertices(4,1)
        # add cells
        editor.add_cell(0,0,1,2)
        editor.add_cell(1,1,2,3)
        # add vertices
        editor.add_vertex(0,0,0,0.5)
        editor.add_vertex(1,1,0,0.5)
        editor.add_vertex(2,0,1,0.5)
        editor.add_vertex(3,1,1,0.5)
        editor.close()
        
        # Rotate the triangle mesh around y axis a random angle in
        # (0,90) degrees
        mesh_1.rotate(numpy.random.rand()*90)
        
        # Exact area 
        exactarea = 1

        # Compute triangulation
        area = 0
        for c0 in cells(mesh_0):
            for c1 in cells(mesh_1):
                triangulation = c0.triangulate_intersection(c1)
                if (triangulation.size>0):
                    tmesh = triangulation_to_mesh_top_2d(triangulation)
                    for t in cells(tmesh):
                        area += t.volume()

    def test_triangulate_intersection_3d(self):

        if MPI.size(mpi_comm_world()) > 1: return
        
        # Create two meshes of the unit cube
        mesh_0 = UnitCubeMesh(1,1,1)
        mesh_1 = UnitCubeMesh(1,1,1)
    
        # Translate second mesh
        dx=Point(numpy.random.rand(),numpy.random.rand(),numpy.random.rand())
        mesh_1.translate(dx)
        exactvolume = (1-abs(dx[0]))*(1-abs(dx[1]))*(1-abs(dx[2]))

        # Compute triangulation
        volume = 0
        for c0 in cells(mesh_0):
            for c1 in cells(mesh_1):
                triangulation = c0.triangulate_intersection(c1)
                if (triangulation.size>0):
                    tmesh = triangulation_to_mesh_3d(triangulation)
                    for t in cells(tmesh):
                        volume += t.volume()
        
        self.assertAlmostEqual(volume,exactvolume)

if __name__ == "__main__":
        unittest.main()

