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
import numpy as np
from dolfin import *
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
        editor.add_cell(i, np.array( (3*i, 3*i + 1, 3*i + 2), dtype='uint') )
    for i in range(num_vertices):
        editor.add_vertex(i, np.array( (triangulation[2*i], triangulation[2*i + 1]), dtype='float'))
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
        editor.add_cell(i, np.array( (3*i, 3*i+1, 3*i+2), dtype='uint'))
    for i in range(num_vertices):
        editor.add_vertex(i, np.array( (triangulation[3*i], triangulation[3*i+1], triangulation[3*i+2]), dtype='float') )
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
        editor.add_cell(i, np.array( (4*i, 4*i+1, 4*i+2, 4*i+3), dtype='uint'))
    for i in range(num_vertices):
        editor.add_vertex(i, np.array( (triangulation[3*i], triangulation[3*i+1], triangulation[3*i+2]), dtype='float'))
    editor.close()
    return mesh

@skip_in_parallel
@pytest.mark.skipif(True, reason="Missing swig typemap")
def test_triangulate_intersection_2d():

    # Create two meshes of the unit square
    mesh_0 = UnitSquareMesh(1, 1)
    mesh_1 = UnitSquareMesh(1, 1)

    # Translate second mesh randomly
    #dx = Point(np.random.rand(),np.random.rand())
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
                triangulation = cpp.geometry.ConvexTriangulation.triangulate(intersection, 2, 2)
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
    editor.add_cell(0, np.array( (0,1,2), dtype='uint'))
    editor.add_cell(1, np.array( (1,2,3), dtype='uint'))

    # Add vertices
    editor.add_vertex(0, np.array( (0, 0, 0.5), dtype='float'))
    editor.add_vertex(1, np.array( (1, 0, 0.5), dtype='float'))
    editor.add_vertex(2, np.array( (0, 1, 0.5), dtype='float'))
    editor.add_vertex(3, np.array( (1, 1, 0.5), dtype='float'))
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
            triangulation = cpp.geometry.ConvexTriangulation.triangulate(intersection, 3, 2)
            if (triangulation.size>0):
                tmesh = triangulation_to_mesh_2d_3d(triangulation)
                for t in cells(tmesh):
                    volume += t.volume()

    errorstring = "rotation angle = " + str(angle)
    assert round(volume - exact_volume, 7) == 0, errorstring

@skip_in_parallel
@pytest.mark.skipif(True, reason="Missing swig typemap for call to ConvexTriangulation")
def test_triangulate_intersection_3d():

    # Create two meshes of the unit cube
    mesh_0 = UnitCubeMesh(1, 1, 1)
    mesh_1 = UnitCubeMesh(1, 1, 1)

    # Translate second mesh
    # dx = Point(np.random.rand(),np.random.rand(),np.random.rand())
    dx = Point(0.913375, 0.632359, 0.097540)

    mesh_1.translate(dx)
    exactvolume = (1 - abs(dx[0]))*(1 - abs(dx[1]))*(1 - abs(dx[2]))

    # Compute triangulation
    volume = 0
    for c0 in cells(mesh_0):
        for c1 in cells(mesh_1):
            intersection = c0.intersection(c1)
            triangulation = cpp.geometry.ConvexTriangulation.triangulate(intersection, 3, 3)
            if (triangulation.size>0):
                tmesh = triangulation_to_mesh_3d(triangulation)
                for t in cells(tmesh):
                    volume += t.volume()

    errorstring = "translation="
    errorstring += str(dx[0])+" "+str(dx[1])+" "+str(dx[2])
    assert round(volume - exactvolume, 7) == 0, errorstring

def test_triangle_triangle_2d_trivial() :
    " These two triangles intersect in a common edge"
    res = cpp.geometry.IntersectionConstruction.intersection_triangle_triangle_2d(Point(0.0, 0.0),
	                                                                          Point(1.0, 0.0),
							                          Point(0.5, 1.0),
							                          Point(0.5, 0.5),
							                          Point(1.0, 1.5),
							                          Point(0.0, 1.5))
    assert len(res) == 4

def test_triangle_triangle_2d() :
    " These two triangles intersect in a common edge"
    res = cpp.geometry.IntersectionConstruction.intersection_triangle_triangle_2d(Point(0.4960412972015322, 0.3953317542541379),
	                                                                          Point(0.5, 0.3997044273055517),
							                          Point(0.5, 0.4060889538943557),
							                          Point(0.4960412972015322, 0.3953317542541379),
							                          Point(0.5, 0.4060889538943557),
                                                                                  Point(.5, .5))
    for p in res:
        print(p[0], p[1])

    assert len(res) == 2

@skip_in_parallel
def test_parallel_segments_2d():
    " These two segments should be parallel and the intersection computed accordingly"
    p0 = Point(0, 0)
    p1 = Point(1, 0)
    q0 = Point(0.4, 0)
    q1 = Point(1.4, 0)
    intersection = cpp.geometry.IntersectionConstruction.intersection_segment_segment_2d(p0, p1, q0, q1)
    assert len(intersection) == 2

def test_equal_segments_2d():
    " These two segments are equal and the intersection computed accordingly"
    p0 = Point(DOLFIN_PI / 7., 9. / DOLFIN_PI)
    p1 = Point(9. / DOLFIN_PI, DOLFIN_PI / 7.)
    q0 = Point(DOLFIN_PI / 7., 9. / DOLFIN_PI)
    q1 = Point(9. / DOLFIN_PI, DOLFIN_PI / 7.)
    intersection = cpp.geometry.IntersectionConstruction.intersection_segment_segment_2d(p0, p1, q0, q1)
    assert len(intersection) == 2

@skip_in_parallel
def test_triangle_segment_2D_1():
    "The intersection of a specific triangle and a specific segment"
    p0 = Point(1e-30, 0)
    p1 = Point(1, 2)
    p2 = Point(2, 1)
    q0 = Point(1, 0)
    q1 = Point(0, 0)
    intersection = cpp.geometry.IntersectionConstruction.intersection_triangle_segment_2d(p0, p1, p2, q0, q1)
    assert len(intersection) == 1
    intersection = cpp.geometry.IntersectionConstruction.intersection_triangle_segment_2d(p0, p1, p2, q1, q0)
    assert len(intersection) == 1

def compare_with_cgal(p0, p1, q0, q1, cgal):
    intersection = cpp.geometry.IntersectionConstruction.intersection_segment_segment_2d(p0, p1, q0, q1)

    #for p in intersection:
    #    print(*p)

    return abs(intersection[0][0] - cgal[0]) < DOLFIN_EPS and \
           abs(intersection[0][1] - cgal[1]) < DOLFIN_EPS

@skip_in_parallel
@pytest.mark.skipif(True, reason="This is a case where the intersection currently fails")
def test_segment_segment_1():
    "Case that fails CGAL comparison. We get a different intersection point but still correct area."
    p0 = Point(-0.50000000000000710543,-0.50000000000000710543)
    p1 = Point(0.99999999999999955591,-2)
    q0 = Point(0.9142135623730932581,-1.9142135623730944793)
    q1 = Point(-0.29289321881346941367,-0.70710678118654635149)

    # The intersection should according to CGAL be
    cgal = Point(0.91066799144849319703, -1.9106679914484945293)

    assert compare_with_cgal(p0, p1, q0, q1, cgal)

@skip_in_parallel
@pytest.mark.skipif(True, reason="This is a case where the intersection currently fails")
def test_segment_segment_2():
    "Case that fails CGAL comparison. We get a different intersection point but still correct area."
    p0 = Point(0.70710678118654746172,-0.70710678118654746172)
    p1 = Point(0.70710678118654612945,0.70710678118654612945)
    q0 = Point(0.70710678118654612945,0.70710678118654113344)
    q1 = Point(0.70710678118654657354,0.2928932188134645842)
    cgal = Point(0.70710678118654612945, 0.7071067811865050512)
    assert compare_with_cgal(p0, p1, q0, q1, cgal)


@skip_in_parallel
#@pytest.mark.skipif(True, reason="This test needs to be updated")
def test_segment_segment_3():
    "Case that fails CGAL comparison. We get a different intersection point but still correct area."
    p0 = Point(0.70710678118654746172,-0.70710678118654746172)
    p1 = Point(0.70710678118654612945,0.70710678118654612945)
    q0 = Point(0.70710678118654757274,-0.097631072937819973756)
    q1 = Point(0.70710678118654257673,-0.1601886205085209236)
    cgal = Point(0.70710678118654679558, -0.10611057050352221132)
    assert compare_with_cgal(p0, p1, q0, q1, cgal)


@skip_in_parallel
@pytest.mark.skipif(True, reason="This is a case where the intersection currently fails")
def test_segment_segment_4():
    "Case that fails CGAL comparison. We get a different intersection point but still correct area."
    p0 = Point(0.70710678118654746172,-0.70710678118654746172)
    p1 = Point(3.5527136788005009294e-14,3.5527136788005009294e-14)
    q0 = Point(0.35355339059326984508,-0.35355339059327078877)
    q1 = Point(0.70710678118655057034,-0.70710678118654701763)
    cgal = Point(0.67572340116162599166, -0.67572340116162288304)
    assert compare_with_cgal(p0, p1, q0, q1, cgal)


@skip_in_parallel
def test_segment_segment_5():
    "Case that failed CGAL comparison but passed when scaling the numerator in x = p0 + o / d * v"
    p0 = Point(1.1429047494274684563e-12,0.5)
    p1 = Point(0.42146018366139809119,0.9214601836602551721)
    q0 = Point(0.34292036732279607136,0.8429203673205103442)
    q1 = Point(0.3429203673205103442,0.8429203673205103442)
    cgal = Point(0.3429203673216533188,0.8429203673205103442)
    assert compare_with_cgal(p0, p1, q0, q1, cgal)


@skip_in_parallel
def test_segment_segment_6():
    "Test that demonstrates, among other things, that we must check the orientation for p0, p1 in intersection_segment_segment_2d"
    p0 = Point(0.045342566799435518599,0.41358248517265505662);
    p1 = Point(0.045342566799434436131,0.41358248517265394639);
    q0 = Point(1.8601965322712701917e-16,0.5);
    q1 = Point(1.873501354054951662e-16,0.3499999999999999778);

    intersection = cpp.geometry.IntersectionConstruction.intersection_segment_segment_2d(p0, p1, q0, q1)

    assert len(intersection) == 0
