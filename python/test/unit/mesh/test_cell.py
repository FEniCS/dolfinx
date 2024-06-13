# Copyright (C) 2013 Anders Logg
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from mpi4py import MPI

import numpy as np
import pytest

import basix
import ufl
from basix.ufl import element
from dolfinx.geometry import squared_distance
from dolfinx.mesh import create_mesh, create_unit_interval


@pytest.mark.skip_in_parallel
def test_distance_interval():
    mesh = create_unit_interval(MPI.COMM_SELF, 1)
    d = np.array([-1.0, 0.0, 0.0])
    assert squared_distance(mesh, mesh.topology.dim, np.array([0]), d) == pytest.approx(1.0)
    d = np.array([0.5, 0.0, 0.0])
    assert squared_distance(mesh, mesh.topology.dim, np.array([0]), d) == pytest.approx(0.0)


@pytest.mark.skip_in_parallel
def test_distance_triangle():
    shape, degree = "triangle", 1
    domain = basix.create_element(basix.ElementFamily.P, basix.CellType[shape], degree)
    x = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]], dtype=np.float64)
    cells = [[0, 1, 2]]
    mesh = create_mesh(MPI.COMM_WORLD, cells, x, domain)
    d = np.array([-1.0, -1.0, 0.0])
    assert squared_distance(mesh, mesh.topology.dim, np.array([0]), d) == pytest.approx(2.0)
    d = np.array([-1.0, 0.5, 0.0])
    assert squared_distance(mesh, mesh.topology.dim, np.array([0]), d) == pytest.approx(1.0)
    d = np.array([0.5, 0.5, 0.0])
    assert squared_distance(mesh, mesh.topology.dim, np.array([0]), d) == pytest.approx(0.0)


@pytest.mark.skip_in_parallel
def test_distance_tetrahedron():
    shape = "tetrahedron"
    degree = 1
    domain = ufl.Mesh(element("Lagrange", shape, degree, shape=(3,), dtype=np.float64))
    x = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1, 1.0, 1]], dtype=np.float64)
    cells = [[0, 1, 2, 3]]
    mesh = create_mesh(MPI.COMM_WORLD, cells, x, domain)
    d = np.array([-1.0, -1.0, -1.0])
    assert squared_distance(mesh, mesh.topology.dim, np.array([0]), d) == pytest.approx(3.0)
    d = np.array([-1.0, 0.5, 0.5])
    assert squared_distance(mesh, mesh.topology.dim, np.array([0]), d) == pytest.approx(1.0)
    d = np.array([0.5, 0.5, 0.5])
    assert squared_distance(mesh, mesh.topology.dim, np.array([0]), d) == pytest.approx(0.0)


# @pytest.mark.skip("volume_entities needs fixing")
# @pytest.mark.parametrize(
#     'mesh', [
#         create_unit_interval(MPI.COMM_WORLD, 18),
#         create_unit_square(MPI.COMM_WORLD, 8, 9, CellType.triangle),
#         create_unit_square(MPI.COMM_WORLD, 8, 9, CellType.quadrilateral),
#         create_unit_cube(MPI.COMM_WORLD, 8, 9, 5, CellType.tetrahedron)
#     ])
# def test_volume_cells(mesh):
#     tdim = mesh.topology.dim
#     map = mesh.topology.index_map(tdim)
#     num_cells = map.size_local
#     v = _cpp.mesh.volume_entities(mesh, range(num_cells), mesh.topology.dim)
#     assert mesh.comm.allreduce(v.sum(), MPI.SUM) == pytest.approx(1.0, rel=1e-9)


# @pytest.mark.skip("volume_entities needs fixing")
# def test_volume_quadrilateralR2():
#     mesh = create_unit_square(MPI.COMM_SELF, 1, 1, CellType.quadrilateral)
#     assert _cpp.mesh.volume_entities(mesh, [0], mesh.topology.dim) == 1.0


# @pytest.mark.skip("volume_entities needs fixing")
# @pytest.mark.parametrize(
#     'x',
#     [[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
#      [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 1.0, 1.0]]])
# def test_volume_quadrilateralR3(x):
#     cells = [[0, 1, 2, 3]]
#     domain = ufl.Mesh(element("Lagrange", "quadrilateral", 1, shape=(2,)))
#     mesh = create_mesh(MPI.COMM_SELF, cells, x, domain)
#     assert _cpp.mesh.volume_entities(mesh, [0], mesh.topology.dim) == 1.0


# @pytest.mark.skip("volume_entities needs fixing")
# @pytest.mark.parametrize(
#     'scaling',
#     [1e0, 1e-5, 1e-10, 1e-15, 1e-20, 1e-30, 1e5, 1e10, 1e15, 1e20, 1e30])
# def test_volume_quadrilateral_coplanarity_check_1(scaling):
#     with pytest.raises(RuntimeError) as error:
#         # Unit square cell scaled down by 'scaling' and the first vertex
#         # is distorted so that the vertices are clearly non coplanar
#         x = [[scaling, 0.5 * scaling, 0.6 * scaling],
#              [0.0, scaling, 0.0],
#              [0.0, 0.0, scaling],
#              [0.0, scaling, scaling]]
#         cells = [[0, 1, 2, 3]]
#         domain = ufl.Mesh(element("Lagrange", "quadrilateral", 1, shape=(2,)))
#         mesh = create_mesh(MPI.COMM_SELF, cells, x, domain)
#         _cpp.mesh.volume_entities(mesh, [0], mesh.topology.dim)

#     assert "Not coplanar" in str(error.value)


# Test when |p0-p3| is ~ 1 but |p1-p2| is small
# The cell is degenerate when scale is below 1e-17, it is expected to
# fail the test.
# @pytest.mark.skip("volume_entities needs fixing")
# @pytest.mark.parametrize('scaling', [1e0, 1e-5, 1e-10, 1e-15])
# def test_volume_quadrilateral_coplanarity_check_2(scaling):
#     with pytest.raises(RuntimeError) as error:
#         # Unit square cell scaled down by 'scaling' and the first vertex
#         # is distorted so that the vertices are clearly non coplanar
#         x = [[1.0, 0.5, 0.6], [0.0, scaling, 0.0],
#              [0.0, 0.0, scaling], [0.0, 1.0, 1.0]]
#         cells = [[0, 1, 2, 3]]
#         domain = ufl.Mesh(element("Lagrange", "quadrilateral", 1, shape=(2,)))
#         mesh = create_mesh(MPI.COMM_SELF, cells, x, domain)
#         _cpp.mesh.volume_entities(mesh, [0], mesh.topology.dim)

#     assert "Not coplanar" in str(error.value)
