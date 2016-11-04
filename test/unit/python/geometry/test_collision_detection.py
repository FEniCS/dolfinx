#!/usr/bin/env py.test

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
# Last changed: 2016-11-04

import pytest
from dolfin import *
from dolfin_utils.test import skip_in_parallel
import numpy as np

@skip_in_parallel
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
    editor.open(mesh, 1, 2)
    editor.init_vertices(2)
    editor.init_cells(1)
    editor.add_vertex(0, 1./7., 1./3.)
    editor.add_vertex(1, 2./7., 3./3.)
    editor.add_cell(0,0,1)
    editor.close()
    cell = Cell(mesh, 0)
    mid = Point(1.5/7., 2./3.)
    assert cell.contains(mid)
    assert cell.contains(cell.midpoint())

@pytest.mark.skipif(True, reason="Not implemented in 3D")
def test_segment_collides_point_3D_1():
    """Test if segment collide with point in 3D"""
    mesh = Mesh()
    editor = MeshEditor()
    editor.open(mesh, 1, 3)
    editor.init_vertices(2)
    editor.init_cells(1)
    editor.add_vertex(0, 1./7., 1./3., 1./DOLFIN_PI)
    editor.add_vertex(1, 2./7., 3./3., 2./DOLFIN_PI)
    editor.add_cell(0,0,1)
    editor.close()
    cell = Cell(mesh, 0)
    mid = Point(1.5/7., 2./3., 1.5/DOLFIN_PI)
    assert cell.contains(mid)
    assert cell.contains(cell.midpoint())


@pytest.mark.skipif(True, reason="Not implemented in 3D")
def test_segment_collides_point_3D_2():
    """Test case by Oyvind from https://bitbucket.org/fenics-project/dolfin/issue/296 for segment point collision in 3D"""
    mesh = Mesh()
    editor = MeshEditor()
    editor.open(mesh, 1, 3)
    editor.init_vertices(2)
    editor.init_cells(1)
    editor.add_vertex(0, 41.06309891, 63.74219894, 68.10320282)
    editor.add_vertex(1, 41.45830154, 62.61560059, 66.43019867)
    editor.add_cell(0,0,1)
    editor.close()
    cell = Cell(mesh, 0)
    assert cell.contains(cell.midpoint())


@skip_in_parallel
def test_triangle_collides_point():
    """Tests if point collide with triangle"""

    mesh = UnitSquareMesh(1, 1)
    cell = Cell(mesh, 0)

    assert cell.collides(Point(0.5)) == True
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

@pytest.mark.skipif(True, reason="Not implemented in 3D")
@skip_in_parallel
def test_tetrahedron_collides_point():
    """Test if point collide with tetrahedron"""

    mesh = UnitCubeMesh(1, 1, 1)
    cell = Cell(mesh, 0)

    assert cell.collides(Point(0.5)) == True
    assert cell.collides(Point(1.5)) == False

@skip_in_parallel
@pytest.mark.skipif(True, reason="Not implemented in 3D")
def test_tetrahedron_collides_triangle():
    """Test if point collide with tetrahedron"""

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
@pytest.mark.skipif(True, reason="Not implemented in 3D")
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

def _test_collision_robustness_2d(aspect, y, step):
    nx = 10
    ny = int(aspect*nx)
    mesh = UnitSquareMesh(nx, ny, 'crossed')
    bb = mesh.bounding_box_tree()

    x = 0.0
    p = Point(x, y)
    while x <= 1.0:
        c = bb.compute_first_entity_collision(Point(x, y))
        assert c < np.uintc(-1)
        x += step

@pytest.mark.skipif(True, reason="Not implemented in 3D")
def _test_collision_robustness_3d(aspect, y, z, step):
    nx = nz = 10
    ny = int(aspect*nx)
    mesh = UnitCubeMesh(nx, ny, nz)
    bb = mesh.bounding_box_tree()

    x = 0.0
    while x <= 1.0:
        c = bb.compute_first_entity_collision(Point(x, y, z))
        assert c < np.uintc(-1)
        x += step

@skip_in_parallel
# @pytest.mark.slow
def test_collision_robustness_slow():
    """Test cases from https://bitbucket.org/fenics-project/dolfin/issue/296"""
    _test_collision_robustness_2d( 100, 1e-14,       1e-5)
    _test_collision_robustness_2d(  40, 1e-03,       1e-5)
    _test_collision_robustness_2d( 100, 0.5 + 1e-14, 1e-5)
    _test_collision_robustness_2d(4.43, 0.5,      4.03e-6)
    #_test_collision_robustness_3d( 100, 1e-14, 1e-14, 1e-5)

@skip_in_parallel
@pytest.mark.slow
#@pytest.mark.skipif(True, reason='Very slow test cases')
def test_collision_robustness_very_slow():
    """Test cases from https://bitbucket.org/fenics-project/dolfin/issue/296"""
    _test_collision_robustness_2d(  10, 1e-16,       1e-7)
    _test_collision_robustness_2d(4.43, 1e-17,    4.03e-6)
    _test_collision_robustness_2d(  40, 0.5,         1e-6)
    _test_collision_robustness_2d(  10, 0.5 + 1e-16, 1e-7)
