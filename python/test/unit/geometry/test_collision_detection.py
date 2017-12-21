"""Unit tests for the CollisionPredicates class"""

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
# Last changed: 2017-09-21

import pytest
from dolfin import *
from dolfin_utils.test import skip_in_parallel
import numpy as np

@skip_in_parallel
def create_triangular_mesh_3D(vertices, cells):
    editor = MeshEditor()
    mesh = Mesh()
    editor.open(mesh,"triangle", 2,3)
    editor.init_cells(2)
    editor.init_vertices(4)
    editor.add_cell(0, cells[0])
    editor.add_cell(1, cells[1])

    editor.add_vertex(0, vertices[0])
    editor.add_vertex(1, vertices[1])
    editor.add_vertex(2, vertices[2])
    editor.add_vertex(3, vertices[3])
    editor.close()
    return mesh;

@skip_in_parallel
def test_interval_collides_point():
    """Test if point collide with interval"""

    mesh = UnitIntervalMesh(1)
    cell = Cell(mesh, 0)

    assert cell.collides(Point(0.5)) == True
    assert cell.collides(Point(1.5)) == False

@skip_in_parallel
def test_segment_collides_point_2D():
    """Test if segment collide with point in 2D"""
    mesh = Mesh()
    editor = MeshEditor()
    editor.open(mesh, "interval", 1, 2)
    editor.init_vertices(2)
    editor.init_cells(1)
    a = np.array( (1./8., 1./4.), dtype='float')
    b = np.array( (2./8., 3./4.), dtype='float')
    editor.add_vertex(0, a)
    editor.add_vertex(1, b)
    editor.add_cell(0, np.array( (0,1), dtype='uint'))
    editor.close()
    cell = Cell(mesh, 0)
    mid = Point(1.5/8., 2./4.)
    mid_average = (a + b) / 2
    assert cell.contains(mid)
    assert cell.contains(Point(a[0], a[1]))
    assert cell.contains(cell.midpoint())

@skip_in_parallel
def test_point_on_segment():
    a = Point(1e-30, 0)
    b = Point(1e-3, 0)
    c = Point(0, 0)
    d = Point(-1e-30, 0)
    q0 = Point(1, 0)
    q1 = Point(0, 0)
    assert cpp.geometry.CollisionPredicates.collides_segment_point_2d(q0, q1, a)
    assert cpp.geometry.CollisionPredicates.collides_segment_point_2d(q0, q1, b)
    assert cpp.geometry.CollisionPredicates.collides_segment_point_2d(q0, q1, c)
    assert not cpp.geometry.CollisionPredicates.collides_segment_point_2d(q0, q1, d)

@skip_in_parallel
def test_point_on_small_segment():
    a = Point(1e-30, 0)
    b = Point(0, 0)
    c = Point(1e-31, 0)
    d = Point(-1e-30, 0)
    q0 = Point(0, 0)
    q1 = Point(1e-30, 0)
    assert cpp.geometry.CollisionPredicates.collides_segment_point_2d(q0, q1, a)
    assert cpp.geometry.CollisionPredicates.collides_segment_point_2d(q0, q1, b)
    assert cpp.geometry.CollisionPredicates.collides_segment_point_2d(q0, q1, c)
    assert not cpp.geometry.CollisionPredicates.collides_segment_point_2d(q0, q1, d)

@skip_in_parallel
def test_segment_collides_point_3D():
    """Test if segment collide with point in 3D"""
    mesh = Mesh()
    editor = MeshEditor()
    editor.open(mesh, "interval", 1, 3)
    editor.init_vertices(2)
    editor.init_cells(1)
    editor.add_vertex(0, np.array( (1./16., 1./8., 1./4.), dtype='float'))
    editor.add_vertex(1, np.array( ( 2./16., 3./8., 2./4.), dtype='float'))
    editor.add_cell(0, np.array( (0,1), dtype='uint') )
    editor.close()
    cell = Cell(mesh, 0)
    mid = Point(1.5/16., 2./8., 1.5/4.)
    assert cell.contains(mid)
    assert cell.contains(cell.midpoint())

@skip_in_parallel
def test_triangle_collides_point():
    """Tests if point collide with triangle"""

    mesh = UnitSquareMesh(1, 1)
    cell = Cell(mesh, 0)

    assert cell.collides(Point(0.5)) == True
    assert cell.collides(Point(1.5)) == False

@skip_in_parallel
def test_degenerate_triangle_collides_point():
    """Test a degenerate triangle that does not collide"""

    p0 = Point(-0.10950608157830554745,0.14049391842169450806)
    p1 = Point(-0.10950608157830354905,0.14049391842169650646)
    p2 = Point(0.32853262580480108168,0.57853262580480113719)
    q = Point(3.5952674716233090635e-06,0.25000359526747162331)

    assert cpp.geometry.CollisionPredicates.collides_triangle_point_2d(p0, p1, p2, q) == False

@skip_in_parallel
@pytest.mark.xfail(strict=True, raises=RuntimeError)
def test_quadrilateral_collides_point():
    mesh = UnitSquareMesh.create(1, 1, CellType.Type.quadrilateral)
    cell = Cell(mesh, 0)
    assert cell.collides(Point(0.5)) == True
    assert cell.collides(Point(1.5)) == False

@skip_in_parallel
@pytest.mark.xfail(strict=True, raises=RuntimeError)
def test_hexahedron_collides_point():
    """Test if point collide with hexahedron"""
    mesh = UnitCubeMesh.create(1, 1, 1, CellType.Type.hexahedron)
    cell = Cell(mesh, 0)

    assert cell.collides(Point(0.5)) == True
    # FIXME: cell.collides(Point) returns True for any 1D, 2D Point
    # cell.collides(Point) returns False if Point[2] != 0
    assert cell.collides(Point(1.5)) == False


@skip_in_parallel
def test_triangle_collides_triangle():
    """Test if triangle collide with triangle"""

    m0 = UnitSquareMesh(8, 8)
    c0 = Cell(m0, 0)

    m1 = UnitSquareMesh(8, 8)
    m1.translate(Point(0.1, 0.1))
    c1 = Cell(m1, 0)
    c2 = Cell(m1, 1)

    assert c0.collides(c0) == True
    assert c0.collides(c1) == True
    assert c0.collides(c2) == True # touching edges
    assert c1.collides(c0) == True
    assert c1.collides(c1) == True
    assert c1.collides(c2) == True
    assert c2.collides(c0) == True # touching edges
    assert c2.collides(c1) == True
    assert c2.collides(c2) == True


@skip_in_parallel
def test_triangle_triangle_collision() :
    "Test that has been failing"
    assert cpp.geometry.CollisionPredicates.collides_triangle_triangle_2d(Point(0.177432070718943, 0.5),
                                                                          Point(0.176638957524249, 0.509972290857582),
                                                                          Point(0.217189283468892, 0.550522616802225),
                                                                          Point(0.333333333333333, 0.52399308981973),
                                                                          Point(0.333333333333333, 0.666666666666667),
                                                                          Point(0.211774439087554, 0.545107772420888))



@skip_in_parallel
def test_triangle_collides_point_3D():
    """Test if point collide with triangle (inspired by test_manifold_dg0_functions)"""
    vertices = [ np.array( (0.0, 0.0, 1.0), dtype='float'),
                 np.array( (1.0, 1.0, 1.0), dtype='float'),
                 np.array( (1.0, 0.0, 0.0), dtype='float'),
                 np.array( (0.0, 1.0, 0.0), dtype='float') ]
    cells = [ np.array( (0, 1, 2), dtype='uint'),
              np.array( (0, 1, 3), dtype='uint') ]
    mesh = create_triangular_mesh_3D(vertices, cells)
    points = [ Point(0.0, 0.0, 1.0),
               Point(1.0, 1.0, 1.0),
               Point(1.0, 0.0, 0.0),
               Point(0.0, 1.0, 0.0),
               Point(0.25, 0.5, 0.75),
               Point(0.5, 0.25, 0.75)
              ]
    A = Cell(mesh, 0)
    B = Cell(mesh, 1)
    assert A.collides(points[0]) == True
    assert B.collides(points[0]) == True
    assert A.collides(points[1]) == True
    assert B.collides(points[1]) == True
    assert A.collides(points[2]) == True
    assert B.collides(points[2]) == False
    assert A.collides(points[3]) == False
    assert B.collides(points[3]) == True
    assert A.collides(points[4]) == False
    assert B.collides(points[4]) == True
    assert A.collides(points[5]) == True
    assert B.collides(points[5]) == False

#@pytest.mark.skipif(True, reason="Not implemented in 3D")
@skip_in_parallel
def test_tetrahedron_collides_point():
    """Test if point collide with tetrahedron"""

    mesh = UnitCubeMesh(1, 1, 1)
    cell = Cell(mesh, 0)

    assert cell.collides(Point(0.5)) == True
    assert cell.collides(Point(1.5)) == False

@skip_in_parallel
#@pytest.mark.skipif(True, reason="Not implemented in 3D")
def test_tetrahedron_collides_triangle():
    """Test if point collide with tetrahedron"""

    tetmesh = UnitCubeMesh(2, 2, 2)
    vertices = [ np.array( (0, 0, 0.5), dtype='float'),
                 np.array( (1, 0, 0.5), dtype='float'),
                 np.array( (0, 1, 0.5), dtype='float'),
                 np.array( (1, 1, 0.5), dtype='float')]
    cells = [ np.array( (0, 1, 2), dtype='uint'),
              np.array( (1, 2, 3), dtype='uint')]

    trimesh = create_triangular_mesh_3D(vertices, cells)
    dx = Point(0.1, 0.1, -0.1)
    trimesh_shift = create_triangular_mesh_3D(vertices, cells)
    trimesh_shift.translate(dx)

    tet0 = Cell(tetmesh, 18)
    tet1 = Cell(tetmesh, 19)
    tri0 = Cell(trimesh, 1)
    tri0shift = Cell(trimesh_shift, 1)

    # proper intersection
    assert tet0.collides(tri0shift) == True
    assert tri0shift.collides(tet0) == True
    assert tet1.collides(tri0shift) == True
    assert tri0shift.collides(tet1) == True

    # point touch
    assert tet0.collides(tri0) == True
    assert tri0.collides(tet0) == True

    # face alignment (true or false)
    assert tet1.collides(tri0) == True
    assert tri0.collides(tet1) == True

@skip_in_parallel
#@pytest.mark.skipif(True, reason="Not implemented in 3D")
def test_tetrahedron_collides_tetrahedron():
    """Test if point collide with tetrahedron"""

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
    assert c3.collides(c3) == True
    assert c45.collides(c45) == True

    # standard collisions
    assert c3.collides(c37) == True
    assert c37.collides(c3) == True

    # edge face
    assert c5.collides(c45) == True
    assert c45.collides(c5) == True

    # touching edges
    assert c5.collides(c19) == False
    assert c19.collides(c5) == False

    # touching faces
    assert c3.collides(c43) == True
    assert c43.collides(c3) == True
