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
#@pytest.mark.skipif(True, reason="Missing swig typemap")
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
    for p in res:
        print p[0],p[1]

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
    "Case that fails CGAL comparison. We get a different intersection point but still correct area."
    p0 = Point(-0.50000000000000710543,-0.50000000000000710543)
    p1 = Point(0.99999999999999955591,-2)
    q0 = Point(0.9142135623730932581,-1.9142135623730944793)
    q1 = Point(-0.29289321881346941367,-0.70710678118654635149)
    intersection = IntersectionConstruction.intersection_segment_segment_2d(p0, p1, q0, q1)

    # The intersection should according to CGAL be
    cgal = Point(0.91066799144849319703, -1.9106679914484945293)

    # We get
    computed = Point(0.9049990067915167, -1.904999006791518)

    assert len(intersection) == 1
    assert (abs(intersection[0][0] - cgal[0]) < DOLFIN_EPS and abs(intersection[0][1] - cgal[1]) < DOLFIN_EPS) or \
        (abs(intersection[0][0] - computed[0]) < DOLFIN_EPS and abs(intersection[0][1] - computed[1]) < DOLFIN_EPS)


@skip_in_parallel
def test_segment_segment_2():
    "Case that fails CGAL comparison. We get a different intersection point but still correct area."
    p0 = Point(0.70710678118654746172,-0.70710678118654746172)
    p1 = Point(0.70710678118654612945,0.70710678118654612945)
    q0 = Point(0.70710678118654612945,0.70710678118654113344)
    q1 = Point(0.70710678118654657354,0.2928932188134645842)
    intersection = IntersectionConstruction.intersection_segment_segment_2d(p0, p1, q0, q1)

    # The intersection should according to CGAL be
    cgal = Point(0.70710678118654612945, 0.7071067811865050512)

    # We get
    computed = Point(0.70710678118654612945, 0.5000000000000027)

    assert len(intersection) == 1
    assert (abs(intersection[0][0] - cgal[0]) < DOLFIN_EPS and abs(intersection[0][1] - cgal[1]) < DOLFIN_EPS) or \
        (abs(intersection[0][0] - computed[0]) < DOLFIN_EPS and abs(intersection[0][1] - computed[1]) < DOLFIN_EPS)


@skip_in_parallel
def test_segment_segment_3():
    "Case that fails CGAL comparison. We get a different intersection point but still correct area."
    p0 = Point(0.70710678118654746172,-0.70710678118654746172)
    p1 = Point(0.70710678118654612945,0.70710678118654612945)
    q0 = Point(0.70710678118654757274,-0.097631072937819973756)
    q1 = Point(0.70710678118654257673,-0.1601886205085209236)
    intersection = IntersectionConstruction.intersection_segment_segment_2d(p0, p1, q0, q1)

    # The intersection should according to CGAL be
    cgal = Point(0.70710678118654679558, -0.10611057050352221132)

    # We get
    computed = Point(0.70710678118654679558, -0.10597207928058017)

    assert len(intersection) == 1
    assert (abs(intersection[0][0] - cgal[0]) < DOLFIN_EPS and abs(intersection[0][1] - cgal[1]) < DOLFIN_EPS) or \
        (abs(intersection[0][0] - computed[0]) < DOLFIN_EPS and abs(intersection[0][1] - computed[1]) < DOLFIN_EPS)


@skip_in_parallel
def test_segment_segment_4():
    "Case that fails CGAL comparison. We get a different intersection point but still correct area."
    p0 = Point(0.70710678118654746172,-0.70710678118654746172)
    p1 = Point(3.5527136788005009294e-14,3.5527136788005009294e-14)
    q0 = Point(0.35355339059326984508,-0.35355339059327078877)
    q1 = Point(0.70710678118655057034,-0.70710678118654701763)
    intersection = IntersectionConstruction.intersection_segment_segment_2d(p0, p1, q0, q1)

    # The intersection should according to CGAL be
    cgal = Point(0.67572340116162599166, -0.67572340116162288304)

    # We get
    computed = Point(0.6754566614934188, -0.6754566614934155)

    assert len(intersection) == 1
    assert (abs(intersection[0][0] - cgal[0]) < DOLFIN_EPS and abs(intersection[0][1] - cgal[1]) < DOLFIN_EPS) or \
        (abs(intersection[0][0] - computed[0]) < DOLFIN_EPS and abs(intersection[0][1] - computed[1]) < DOLFIN_EPS)

@skip_in_parallel
def test_parallel_segments_topological_1d():
    "Parallel segments with small intersection"
    p0 = Point(0.70710678118654757274, 1)
    p1 = Point(0, 1)
    q0 = Point(2.2204460492503130808e-16, 1)
    q1 = Point(-2, 1)
    intersection = IntersectionConstruction.intersection_segment_segment_2d(p0, p1, q0, q1)

    assert len(intersection) == 2
    assert abs(intersection[0][0]) < DOLFIN_EPS and abs(intersection[0][1]-1) < DOLFIN_EPS and abs(intersection[1][0]-2.2204460492503130808e-16)<DOLFIN_EPS and abs(intersection[1][1]-1)<DOLFIN_EPS

@skip_in_parallel
def test_parallel_segments_almost_topological_1d():
    "Almost parallel segments"
    p0 = Point(0.70710678118654757274, 1.000000000000000222)
    p1 = Point(0, 1)
    q0 = Point(2.2204460492503130808e-16, 1)
    q1 = Point(-2, 1.0000000000000004441)
    intersection = IntersectionConstruction.intersection_segment_segment_2d(p0, p1, q0, q1)

    # exact intersection points found using CGAL
    assert len(intersection) == 1
    assert abs(intersection[0][0]-9.19739e-17) < DOLFIN_EPS and abs(intersection[0][1]-1)<DOLFIN_EPS

'''
    //-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
'''
@skip_in_parallel
def test_segment_segment_10():
    "Case that fails CGAL comparison. We get a different intersection point but still correct area."
    p0 = Point(0.5257311121191330372,0.5257311121191330372)
    p1 = Point(-0.3330225275452586287,2.211130026965254736)
    q0 = Point(0.4584638764865089366,0.6577504954480113408)
    q1 = Point(0.3968022466674201443,0.7787682579175255748)
    intersection = IntersectionConstruction.intersection_segment_segment_2d(p0, p1, q0, q1)

    # The intersection should according to CGAL be
    cgal = Point(0.43271476769028760501, 0.70828596687887079852 )

    # We get
    computed = Point(0.43175174708592745088, 0.71017600123400592693 )

    assert len(intersection) == 1
    assert (abs(intersection[0][0] - cgal[0]) < DOLFIN_EPS and abs(intersection[0][1] - cgal[1]) < DOLFIN_EPS) or \
        (abs(intersection[0][0] - computed[0]) < DOLFIN_EPS and abs(intersection[0][1] - computed[1]) < DOLFIN_EPS)

@skip_in_parallel
def test_segment_segment_11():
    "Case that fails CGAL comparison. We get a different intersection point but still correct area."
    p0 = Point(0.5877852522924732481,0.8090169943749473402)
    p1 = Point(-2.205819241042367818,0.366553510209999156)
    q0 = Point(-0.6180339887498947915,0.6180339887498949025)
    q1 = Point(-0.6623669916539393077,0.6110123308970318856)
    intersection = IntersectionConstruction.intersection_segment_segment_2d(p0, p1, q0, q1)

    # The intersection should according to CGAL be
    cgal = Point(-0.62784103695643067589, 0.61648070490846706981)

    # We get
    computed = Point(-0.64532926296564563362, 0.61371084201972847527)

    assert len(intersection) == 1
    assert (abs(intersection[0][0] - cgal[0]) < DOLFIN_EPS and abs(intersection[0][1] - cgal[1]) < DOLFIN_EPS) or \
        (abs(intersection[0][0] - computed[0]) < DOLFIN_EPS and abs(intersection[0][1] - computed[1]) < DOLFIN_EPS)

def test_segment_segment_12():
    "Case that fails CGAL comparison. We get a different intersection point but still correct area."
    p0 = Point(0.2840790438404123819,1)
    p1 = Point(0.8632712640026805051,-0.1367287359973194949)
    q0 = Point(0.3404406452539165961,0.8893841289587627319)
    q1 = Point(0.3968022466674206994,0.7787682579175255748)
    intersection = IntersectionConstruction.intersection_segment_segment_2d(p0, p1, q0, q1)

    # The intersection should according to CGAL be
    cgal = Point(0.46507661264164801285, 0.64477226999980374877)

    # We get
    computed = Point(0.3829134612675635374, 0.80602653405199187198)

    assert len(intersection) == 1
    assert (abs(intersection[0][0] - cgal[0]) < DOLFIN_EPS and abs(intersection[0][1] - cgal[1]) < DOLFIN_EPS) or \
        (abs(intersection[0][0] - computed[0]) < DOLFIN_EPS and abs(intersection[0][1] - computed[1]) < DOLFIN_EPS)


@skip_in_parallel
def test_segment_segment_13():
    "Case that fails CGAL comparison. We get a different intersection point but still correct area."
    p0 = Point(0.3404406452539165961,0.8893841289587627319)
    p1 = Point(0.3968022466674206994,0.7787682579175255748)
    q0 = Point(0.2840790438404123819,1)
    q1 = Point(0.8632712640026805051,-0.1367287359973194949)
    intersection = IntersectionConstruction.intersection_segment_segment_2d(p0, p1, q0, q1)

    # The intersection should according to CGAL be
    cgal = Point(0.3829134612675635374, 0.80602653405199187198)

    # We get
    computed = Point(0.46507661264164801285, 0.64477226999980374877)

    assert len(intersection) == 1
    assert (abs(intersection[0][0] - cgal[0]) < DOLFIN_EPS and abs(intersection[0][1] - cgal[1]) < DOLFIN_EPS) or \
        (abs(intersection[0][0] - computed[0]) < DOLFIN_EPS and abs(intersection[0][1] - computed[1]) < DOLFIN_EPS)


@skip_in_parallel
def test_segment_segment_14():
    "Case that fails CGAL comparison. We get a different intersection point but still correct area."
    p0 = Point(0.2840790438404123819,1)
    p1 = Point(0.8632712640026805051,-0.1367287359973194949)
    q0 = Point(0.3404406452539165961,0.8893841289587627319)
    q1 = Point(0.3968022466674206994,0.7787682579175255748)
    intersection = IntersectionConstruction.intersection_segment_segment_2d(p0, p1, q0, q1)

    # The intersection should according to CGAL be
    cgal = Point(0.3829134612675635374, 0.80602653405199187198)

    # We get
    computed = Point(0.37649557393344845746, 0.81862234715707460353)

    assert len(intersection) == 1
    assert (abs(intersection[0][0] - cgal[0]) < DOLFIN_EPS and abs(intersection[0][1] - cgal[1]) < DOLFIN_EPS) or \
        (abs(intersection[0][0] - computed[0]) < DOLFIN_EPS and abs(intersection[0][1] - computed[1]) < DOLFIN_EPS)

@skip_in_parallel
def test_segment_segment_15():
    "Case that fails CGAL comparison. We get an intersection point outside the interval."
    p0 = Point(0.5877852522924732481,0.8090169943749473402)
    p1 = Point(-2.205819241042367818,0.366553510209999156)
    q0 = Point(-0.6623669916539391966,0.6110123308970317746)
    q1 = Point(-0.9154969939148456248,0.5709204771595896011)
    intersection = IntersectionConstruction.intersection_segment_segment_2d(p0, p1, q0, q1)

    # The intersection should according to CGAL be
    cgal = Point(-0.83988301296301948184, 0.58289655521335470567)

    assert len(intersection) == 1
    assert abs(intersection[0][0] - cgal[0]) < DOLFIN_EPS and abs(intersection[0][1] - cgal[1]) < DOLFIN_EPS


@skip_in_parallel
def test_segment_segment_16():
    "Case that fails CGAL comparison"
    p0 = Point(0.19615242270663213464,2)
    p1 = Point(0.63397459621556140341,0.36602540378443870761)
    q0 = Point(0.50000000000000055511,0.86602540378443704228)
    q1 = Point(0.86602540378443859659,-0.50000000000000022204)
    intersection = IntersectionConstruction.intersection_segment_segment_2d(p0, p1, q0, q1)

    cgal = Point(0.50614970528545843997, 0.84307439120753291384)

    assert len(intersection) == 1
    assert abs(intersection[0][0] - cgal[0]) < DOLFIN_EPS and abs(intersection[0][1] - cgal[1]) < DOLFIN_EPS
