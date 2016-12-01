#!/usr/bin/env py.test

"""Unit tests for the IntersectionConstruction class"""

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

import pytest
import numpy
from dolfin import *
from six.moves import xrange as range
from dolfin_utils.test import skip_in_parallel

def triangulation_to_mesh_2d(triangulation):
    editor = MeshEditor()
    mesh = Mesh()
    editor.open(mesh, 2, 2)
    num_cells = len(triangulation) // 6
    num_vertices = len(triangulation) // 2
    editor.init_cells(num_cells)
    editor.init_vertices(num_vertices)
    for i in range(num_cells):
        editor.add_cell(i, 3*i, 3*i + 1, 3*i + 2)
    for i in range(num_vertices):
        editor.add_vertex(i, triangulation[2*i], triangulation[2*i + 1])
    editor.close()
    return mesh

def triangulation_to_mesh_2d_3d(triangulation):
    editor = MeshEditor()
    mesh = Mesh()
    editor.open(mesh,2,3)
    num_cells = len(triangulation) // 9
    num_vertices = len(triangulation) // 3
    editor.init_cells(num_cells)
    editor.init_vertices(num_vertices)
    for i in range(num_cells):
        editor.add_cell(i, 3*i, 3*i+1, 3*i+2)
    for i in range(num_vertices):
        editor.add_vertex(i, triangulation[3*i], triangulation[3*i+1], triangulation[3*i+2])
    editor.close()
    return mesh

def triangulation_to_mesh_3d(triangulation):
    editor = MeshEditor()
    mesh = Mesh()
    editor.open(mesh,3,3)
    num_cells = len(triangulation) // 12
    num_vertices = len(triangulation) // 3
    editor.init_cells(num_cells)
    editor.init_vertices(num_vertices)
    for i in range(num_cells):
        editor.add_cell(i, 4*i, 4*i+1, 4*i+2, 4*i+3)
    for i in range(num_vertices):
        editor.add_vertex(i, triangulation[3*i], triangulation[3*i+1], triangulation[3*i+2])
    editor.close()
    return mesh

@skip_in_parallel
@pytest.mark.skipif(True, reason="Missing swig typemap")
def test_triangulate_intersection_2d():

    # Create two meshes of the unit square
    mesh_0 = UnitSquareMesh(1, 1)
    mesh_1 = UnitSquareMesh(1, 1)

    # Translate second mesh randomly
    #dx = Point(numpy.random.rand(),numpy.random.rand())
    dx = Point(0.278498, 0.546881)
    mesh_1.translate(dx)

    # Exact volume of intersection
    exactvolume = (1 - abs(dx[0]))*(1 - abs(dx[1]))

    # Compute triangulation volume
    volume = 0
    for c0 in cells(mesh_0):
        for c1 in cells(mesh_1):
            intersection = c0.intersection(c1)
            if len(intersection) >= 3 :
                triangulation = ConvexTriangulation.triangulate(intersection, 2, 2)
                tmesh = triangulation_to_mesh_2d(triangulation)
                for t in cells(tmesh):
                    volume += t.volume()

    errorstring = "translation=" + str(dx[0]) + str(" ") + str(dx[1])
    assert round(volume - exactvolume, 7) == 0, errorstring

@skip_in_parallel
@pytest.mark.skipif(True, reason="Not implemented in 3D")
def test_triangulate_intersection_2d_3d():

    # Note: this test will fail if the triangle mesh is aligned
    # with the tetrahedron mesh

    # Create a unit cube
    mesh_0 = UnitCubeMesh(1,1,1)

    # Create a 3D surface mesh
    editor = MeshEditor()
    mesh_1 = Mesh()
    editor.open(mesh_1,2,3)
    editor.init_cells(2)
    editor.init_vertices(4)

    # Add cells
    editor.add_cell(0,0,1,2)
    editor.add_cell(1,1,2,3)

    # Add vertices
    editor.add_vertex(0,0,0,0.5)
    editor.add_vertex(1,1,0,0.5)
    editor.add_vertex(2,0,1,0.5)
    editor.add_vertex(3,1,1,0.5)
    editor.close()

    # Rotate the triangle mesh around y axis
    angle = 23.46354
    mesh_1.rotate(angle,1)

    # Exact area
    exact_volume = 1

    # Compute triangulation
    volume = 0
    for c0 in cells(mesh_0):
        for c1 in cells(mesh_1):
            intersection = c0.intersection(c1)
            triangulation = ConvexTriangulation.triangulate(intersection)
            if (triangulation.size>0):
                tmesh = triangulation_to_mesh_2d_3d(triangulation)
                for t in cells(tmesh):
                    volume += t.volume()

    errorstring = "rotation angle = " + str(angle)
    assert round(volume - exact_volume, 7) == 0, errorstring

@skip_in_parallel
@pytest.mark.skipif(True, reason="Not implemented in 3D")
def test_triangulate_intersection_3d():

    # Create two meshes of the unit cube
    mesh_0 = UnitCubeMesh(1, 1, 1)
    mesh_1 = UnitCubeMesh(1, 1, 1)

    # Translate second mesh
    # dx = Point(numpy.random.rand(),numpy.random.rand(),numpy.random.rand())
    dx = Point(0.913375, 0.632359, 0.097540)

    mesh_1.translate(dx)
    exactvolume = (1 - abs(dx[0]))*(1 - abs(dx[1]))*(1 - abs(dx[2]))

    # Compute triangulation
    volume = 0
    for c0 in cells(mesh_0):
        for c1 in cells(mesh_1):
            intersection = c0.intersection(c1)
            triangulation = ConvexTriangulation.triangulate(intersection)
            if (triangulation.size>0):
                tmesh = triangulation_to_mesh_3d(triangulation)
                for t in cells(tmesh):
                    volume += t.volume()

    errorstring = "translation="
    errorstring += str(dx[0])+" "+str(dx[1])+" "+str(dx[2])
    assert round(volume - exactvolume, 7) == 0, errorstring


def test_triangle_triangle_2d_trivial() :
    " These two triangles intersect in a common edge"
    res = IntersectionConstruction.intersection_triangle_triangle_2d(Point(0.0, 0.0),
	                                                             Point(1.0, 0.0),
							             Point(0.5, 1.0),
							             Point(0.5, 0.5),
							             Point(1.0, 1.5),
							             Point(0.0, 1.5))
    assert len(res) == 4


def test_triangle_triangle_2d() :
    " These two triangles intersect in a common edge"
    res = IntersectionConstruction.intersection_triangle_triangle_2d(Point(0.4960412972015322, 0.3953317542541379),
	                                                             Point(0.5, 0.3997044273055517),
							             Point(0.5, 0.4060889538943557),
							             Point(0.4960412972015322, 0.3953317542541379),
							             Point(0.5, 0.4060889538943557),
							             Point(.5, .5))
    assert len(res) == 2

@skip_in_parallel
def test_parallel_segments_2d():
    " These two segments should be parallel and the intersection computed accordingly"
    p0 = Point(0, 0)
    p1 = Point(1, 0)
    q0 = Point(0.4, 0)
    q1 = Point(1.4, 0)
    intersection = IntersectionConstruction.intersection_segment_segment_2d(p0, p1, q0, q1)
    assert len(intersection) == 2

def test_equal_segments_2d():
    " These two segments are equal and the intersection computed accordingly"
    p0 = Point(DOLFIN_PI / 7., 9. / DOLFIN_PI)
    p1 = Point(9. / DOLFIN_PI, DOLFIN_PI / 7.)
    q0 = Point(DOLFIN_PI / 7., 9. / DOLFIN_PI)
    q1 = Point(9. / DOLFIN_PI, DOLFIN_PI / 7.)
    intersection = IntersectionConstruction.intersection_segment_segment_2d(p0, p1, q0, q1)
    assert len(intersection) == 2

@skip_in_parallel
def test_triangle_segment_2D_1():
    "The intersection of a specific triangle and a specific segment"
    p0 = Point(1e-30, 0)
    p1 = Point(1, 2)
    p2 = Point(2, 1)
    q0 = Point(1, 0)
    q1 = Point(0, 0)
    intersection = IntersectionConstruction.intersection_triangle_segment_2d(p0, p1, p2, q0, q1)
    assert len(intersection) == 1
    intersection = IntersectionConstruction.intersection_triangle_segment_2d(p0, p1, p2, q1, q0)
    assert len(intersection) == 1



@skip_in_parallel
def test_segment_segment_1():
    "Case that previously failed in CGAL comparison."
    p0 = Point(0.70710678118654746172,0.70710678118654746172)
    p1 = Point(-0.70710678118654757274,-0.70710678118654735069)
    q0 = Point(-5,-5)
    q1 = Point(5,5)
    intersection = IntersectionConstruction.intersection_segment_segment_2d(p0, p1, q0, q1)

    # intersection should be p0 0.70710678118654746172 0.70710678118654746172
    for p in intersection:
        print p[0],p[1]

    assert len(intersection) == 1

@skip_in_parallel
def test_segment_segment_2():
    "Case that previously failed in CGAL comparison."
    p0 = Point(0.70710678118654746172, 0.70710678118654746172)
    p1 = Point(-2.1213203435596423851, 0.70710678118654768376)
    q0 = Point(-5, -5)
    q1 = Point(5, 5)
    intersection = IntersectionConstruction.intersection_segment_segment_2d(p0, p1, q0, q1)

    # intersection should be p0 0.70710678118654746172 0.70710678118654746172
    for p in intersection:
        print p[0],p[1]

    assert len(intersection) == 1

@skip_in_parallel
def test_segment_segment_3():
    "Case that previously failed in CGAL comparison."
    p0 = Point(-5, -5)
    p1 = Point(5, 5)
    q0 = Point(-0.70710678118654757274,-0.70710678118654735069)
    q1 = Point(0.70710678118654768376,0.70710678118654723967)
    intersection = IntersectionConstruction.intersection_segment_segment_2d(p0, p1, q0, q1)

    # intersection should be  -0.23570226039551583908 -0.23570226039551583908
    for p in intersection:
        print p[0],p[1]

    assert len(intersection) == 1

@skip_in_parallel
def test_segment_segment_4():
    "Case that previously failed in CGAL comparison."
    p0 = Point(-5, -5)
    p1 = Point(5, 5)
    q0 = Point(-0.70710678118654757274,-0.70710678118654735069)
    q1 = Point(2.1213203435596423851,-0.70710678118654801683)
    intersection = IntersectionConstruction.intersection_segment_segment_2d(p0, p1, q0, q1)

    # intersection should be  -0.23570226039551583908 -0.23570226039551583908
    for p in intersection:
        print p[0],p[1]

    assert len(intersection) == 1

@skip_in_parallel
def test_triangle_triangle():
    "Tri tri case corresponding to test_segment_segment_1 - 4 above"
    p0 = Point(-5,-5)
    p1 = Point(5,-5)
    p2 = Point(5,5)
    q0 = Point(-0.70710678118654757274,-0.70710678118654735069)
    q1 = Point(0.70710678118654768376,0.70710678118654723967)
    q2 = Point(2.1213203435596423851,-0.70710678118654801683)
    intersection
    = IntersectionConstruction.intersection_triangle_triangle_2d(p0, p1, p2,
                                                                 q0, q1, q2)

    '''
    intersection should be
    2.1213203435596423851 -0.70710678118654801683
    -0.70710678118654746172 -0.70710678118654746172
    -0.23570226039551583908 -0.23570226039551583908
    0.70710678118654768376 0.70710678118654723967
    '''

    for p in intersection:
        print p[0],p[1]

    assert len(intersection) == 4
