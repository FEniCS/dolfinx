"""Unit tests for the IntersectionTriangulation class"""

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
# Last changed: 2014-02-19

import unittest
import numpy
from dolfin import *


def triangulation_to_mesh_2d(triangulation):
    editor = MeshEditor()
    mesh = Mesh()
    editor.open(mesh, 2, 2)
    num_cells = len(triangulation) / 6
    num_vertices = len(triangulation) / 2
    editor.init_cells(num_cells)
    editor.init_vertices(num_vertices)
    for i in xrange(num_cells):
        editor.add_cell(i, 3*i, 3*i + 1, 3*i + 2)
    for i in xrange(num_vertices):
        editor.add_vertex(i, triangulation[2*i], triangulation[2*i + 1])
    editor.close()
    return mesh

def triangulation_to_mesh_2d_3d(triangulation):
    editor = MeshEditor()
    mesh = Mesh()
    editor.open(mesh,2,3)
    num_cells = len(triangulation)/9
    num_vertices = len(triangulation)/3
    editor.init_cells(num_cells)
    editor.init_vertices(num_vertices)
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
    editor.init_cells(num_cells)
    editor.init_vertices(num_vertices)
    for i in xrange(num_cells):
        editor.add_cell(i, 4*i, 4*i+1, 4*i+2, 4*i+3)
    for i in xrange(num_vertices):
        editor.add_vertex(i, triangulation[3*i], triangulation[3*i+1], triangulation[3*i+2])
    editor.close()
    return mesh


class TriangulateTest(unittest.TestCase):

    def test_triangulate_intersection_2d(self):

        if MPI.size(mpi_comm_world()) > 1: return

        # Create two meshes of the unit square
        mesh_0 = UnitSquareMesh(1, 1)
        mesh_1 = UnitSquareMesh(1, 1)

        # Translate second mesh randomly
        #dx = Point(numpy.random.rand(),numpy.random.rand())
        dx = Point(0.278498,0.546881,0.957506)
        mesh_1.translate(dx)

        exactvolume = (1-abs(dx[0]))*(1-abs(dx[1]))

        # Compute triangulation
        volume = 0
        for c0 in cells(mesh_0):
            for c1 in cells(mesh_1):
                triangulation = c0.triangulate_intersection(c1)
                if (triangulation.size>0):
                    tmesh = triangulation_to_mesh_2d(triangulation)
                    for t in cells(tmesh):
                        volume += t.volume()

        errorstring = "translation=" + str(dx[0]) + str(" ") + str(dx[1])
        self.assertAlmostEqual(volume, exactvolume,7,errorstring)

    def test_triangulate_intersection_2d_3d(self):

        # Note: this test will fail if the triangle mesh is aligned
        # with the tetrahedron mesh

        if MPI.size(mpi_comm_world()) > 1: return

        # Create a unit cube
        mesh_0 = UnitCubeMesh(1,1,1)

        # Create a 3D surface mesh
        editor = MeshEditor()
        mesh_1 = Mesh()
        editor.open(mesh_1,2,3)
        editor.init_cells(2)
        editor.init_vertices(4)
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
        #angle = numpy.random.rand()*90
        angle = 23.46354
        mesh_1.rotate(angle,1)

        # Exact area
        exactvolume = 1

        # Compute triangulation
        volume = 0
        for c0 in cells(mesh_0):
            for c1 in cells(mesh_1):
                triangulation = c0.triangulate_intersection(c1)
                if (triangulation.size>0):
                    tmesh = triangulation_to_mesh_2d_3d(triangulation)
                    for t in cells(tmesh):
                        volume += t.volume()

        errorstring = "rotation angle = " + str(angle)
        self.assertAlmostEqual(volume,exactvolume,7,errorstring)


    def test_triangulate_intersection_3d(self):

        if MPI.size(mpi_comm_world()) > 1: return

        # Create two meshes of the unit cube
        mesh_0 = UnitCubeMesh(1,1,1)
        mesh_1 = UnitCubeMesh(1,1,1)

        # Translate second mesh
        # dx = Point(numpy.random.rand(),numpy.random.rand(),numpy.random.rand())
        dx = Point(0.913375,0.632359,0.097540)

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

        errorstring = "translation="
        errorstring += str(dx[0])+" "+str(dx[1])+" "+str(dx[2])
        self.assertAlmostEqual(volume,exactvolume,7,errorstring)


if __name__ == "__main__":
        unittest.main()

