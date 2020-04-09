# Copyright (C) 2013-2014 Anders Logg
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for BoundingBoxTree"""

import numpy
import pytest
from mpi4py import MPI

from dolfinx import (UnitCubeMesh, UnitIntervalMesh, UnitSquareMesh, cpp,
                     geometry)
from dolfinx.geometry import BoundingBoxTree
from dolfinx_utils.test.skips import skip_in_parallel

# --- compute_collisions with point ---


@skip_in_parallel
def test_compute_collisions_point_1d():

    reference = {1: set([4])}

    p = numpy.array([0.3, 0, 0])
    mesh = UnitIntervalMesh(MPI.COMM_WORLD, 16)
    for dim in range(1, 2):
        tree = BoundingBoxTree(mesh, mesh.topology.dim)
        entities, _ = geometry.compute_collisions_point(tree, p)
        assert set(entities) == reference[dim]


# @skip_in_parallel
# def test_compute_collisions_point_2d():
#     reference = {1: set([226]),
#                  2: set([136, 137])}
#     p = numpy.array([0.3, 0.3, 0.0])
#     mesh = UnitSquareMesh(MPI.COMM_WORLD, 16, 16)
#     for dim in range(1, 3):
#         tree = BoundingBoxTree(mesh, mesh.topology.dim)
#         entities = tree.compute_collisions_point(p)
#         mp = cpp.mesh.midpoints(mesh, dim, entities)
#         assert set(entities) == reference[dim]


@skip_in_parallel
def test_compute_collisions_point_3d():
    reference = {
        1: set([1364]),
        2: set([1967, 1968, 1970, 1972, 1974, 1976]),
        3: set([876, 877, 878, 879, 880, 881])
    }
    p = numpy.array([0.3, 0.3, 0.3])
    mesh = UnitCubeMesh(MPI.COMM_WORLD, 8, 8, 8)
    tree = BoundingBoxTree(mesh, mesh.topology.dim)
    for dim in range(1, 4):
        entities, _ = geometry.compute_collisions_point(tree, p)

        # FIXME: Face and edges tests are excluded because test
        # mistakenly relies on the face and edge indices
        tdim = mesh.topology.dim
        if dim != tdim - 1 and dim != tdim - 2:
            assert set(entities) == reference[dim]


# --- compute_collisions with tree ---


@skip_in_parallel
@pytest.mark.parametrize("point,cells", [(numpy.array([0.52, 0, 0]), [
    set([8, 9, 10, 11, 12, 13, 14, 15]),
    set([0, 1, 2, 3, 4, 5, 6, 7])]),
    (numpy.array([0.9, 0, 0]), [set([14, 15]), set([0, 1])])])
def test_compute_collisions_tree_1d(point, cells):
    mesh_A = UnitIntervalMesh(MPI.COMM_WORLD, 16)
    mesh_B = UnitIntervalMesh(MPI.COMM_WORLD, 16)

    bgeom = mesh_B.geometry.x
    bgeom += point

    tree_A = BoundingBoxTree(mesh_A, mesh_A.topology.dim)
    tree_B = BoundingBoxTree(mesh_B, mesh_B.topology.dim)
    entities_A, entities_B = geometry.compute_collisions_bb(tree_A, tree_B)

    assert set(entities_A) == cells[0]
    assert set(entities_B) == cells[1]


@skip_in_parallel
@pytest.mark.parametrize("point,cells", [(numpy.array([0.52, 0.51, 0.0]), [
    [20, 21, 22, 23, 28, 29, 30, 31],
    [0, 1, 2, 3, 8, 9, 10, 11]]),
    (numpy.array([0.9, -0.9, 0.0]), [[6, 7], [24, 25]])])
def test_compute_collisions_tree_2d(point, cells):
    mesh_A = UnitSquareMesh(MPI.COMM_WORLD, 4, 4)
    mesh_B = UnitSquareMesh(MPI.COMM_WORLD, 4, 4)
    bgeom = mesh_B.geometry.x
    bgeom += point
    tree_A = BoundingBoxTree(mesh_A, mesh_A.topology.dim)
    tree_B = BoundingBoxTree(mesh_B, mesh_B.topology.dim)
    entities_A, entities_B = geometry.compute_collisions_bb(tree_A, tree_B)
    assert set(entities_A) == set(cells[0])
    assert set(entities_B) == set(cells[1])


@skip_in_parallel
def test_compute_collisions_tree_3d():

    references = [[
        set([18, 19, 20, 21, 22, 23, 42, 43, 44, 45, 46, 47]),
        set([0, 1, 2, 3, 4, 5, 24, 25, 26, 27, 28, 29])
    ], [
        set([6, 7, 8, 9, 10, 11, 30, 31, 32, 33, 34, 35]),
        set([12, 13, 14, 15, 16, 17, 36, 37, 38, 39, 40, 41])
    ]]

    points = [numpy.array([0.52, 0.51, 0.3]),
              numpy.array([0.9, -0.9, 0.3])]

    for i, point in enumerate(points):

        mesh_A = UnitCubeMesh(MPI.COMM_WORLD, 2, 2, 2)
        mesh_B = UnitCubeMesh(MPI.COMM_WORLD, 2, 2, 2)

        bgeom = mesh_B.geometry.x
        bgeom += point

        tree_A = BoundingBoxTree(mesh_A, mesh_A.topology.dim)
        tree_B = BoundingBoxTree(mesh_B, mesh_B.topology.dim)
        entities_A, entities_B = geometry.compute_collisions_bb(tree_A, tree_B)

        assert set(entities_A) == references[i][0]
        assert set(entities_B) == references[i][1]


# --- compute_entity_collisions with point ---


@skip_in_parallel
def test_compute_entity_collisions_1d():
    reference = set([4])
    p = numpy.array([0.3, 0.0, 0.0])
    mesh = UnitIntervalMesh(MPI.COMM_WORLD, 16)
    tree = BoundingBoxTree(mesh, mesh.topology.dim)
    entities, _ = geometry.compute_entity_collisions_mesh(tree, mesh, p)
    assert set(entities) == reference


@skip_in_parallel
def test_compute_entity_collisions_2d():
    reference = set([136, 137])
    p = numpy.array([0.3, 0.3, 0.0])
    mesh = UnitSquareMesh(MPI.COMM_WORLD, 16, 16)
    tree = BoundingBoxTree(mesh, mesh.topology.dim)
    entities, _ = geometry.compute_entity_collisions_mesh(tree, mesh, p)
    assert set(entities) == reference


@skip_in_parallel
def test_compute_entity_collisions_3d():
    reference = set([876, 877, 878, 879, 880, 881])
    p = numpy.array([0.3, 0.3, 0.3])
    mesh = UnitCubeMesh(MPI.COMM_WORLD, 8, 8, 8)
    tree = BoundingBoxTree(mesh, mesh.topology.dim)
    entities, _ = geometry.compute_entity_collisions_mesh(tree, mesh, p)
    assert set(entities) == reference


# --- compute_entity_collisions with tree ---


@skip_in_parallel
@pytest.mark.parametrize("point,cells", [(numpy.array([0.52, 0, 0]), [
    set([8, 9, 10, 11, 12, 13, 14, 15]),
    set([0, 1, 2, 3, 4, 5, 6, 7])]),
    (numpy.array([0.9, 0, 0]), [set([14, 15]), set([0, 1])])])
def test_compute_entity_collisions_tree_1d(point, cells):
    mesh_A = UnitIntervalMesh(MPI.COMM_WORLD, 16)
    mesh_B = UnitIntervalMesh(MPI.COMM_WORLD, 16)

    bgeom = mesh_B.geometry.x
    bgeom += point

    tree_A = BoundingBoxTree(mesh_A, mesh_A.topology.dim)
    tree_B = BoundingBoxTree(mesh_B, mesh_B.topology.dim)
    entities_A, entities_B = geometry.compute_entity_collisions_bb(
        tree_A, mesh_A, tree_B, mesh_B)
    assert set(entities_A) == cells[0]
    assert set(entities_B) == cells[1]


@skip_in_parallel
def test_compute_entity_collisions_tree_2d():
    references = [[
        set([20, 21, 22, 23, 28, 29, 30, 31]),
        set([0, 1, 2, 3, 8, 9, 10, 11])
    ], [set([6]), set([25])]]

    points = [numpy.array([0.52, 0.51, 0.0]), numpy.array([0.9, -0.9, 0.0])]
    for i, point in enumerate(points):
        mesh_A = UnitSquareMesh(MPI.COMM_WORLD, 4, 4)
        mesh_B = UnitSquareMesh(MPI.COMM_WORLD, 4, 4)

        bgeom = mesh_B.geometry.x
        bgeom += point

        tree_A = BoundingBoxTree(mesh_A, mesh_A.topology.dim)
        tree_B = BoundingBoxTree(mesh_B, mesh_B.topology.dim)
        entities_A, entities_B = geometry.compute_entity_collisions_bb(
            tree_A, mesh_A, tree_B, mesh_B)
        assert set(entities_A) == references[i][0]
        assert set(entities_B) == references[i][1]


@skip_in_parallel
def test_compute_entity_collisions_tree_3d():
    references = [[
        set([18, 19, 20, 21, 22, 23, 42, 43, 44, 45, 46, 47]),
        set([0, 1, 2, 3, 4, 5, 24, 25, 26, 27, 28, 29])
    ], [set([7, 8, 30, 31, 32]),
        set([15, 16, 17, 39, 41])]]

    points = [numpy.array([0.52, 0.51, 0.3]), numpy.array([0.9, -0.9, 0.3])]
    for i, point in enumerate(points):

        mesh_A = UnitCubeMesh(MPI.COMM_WORLD, 2, 2, 2)
        mesh_B = UnitCubeMesh(MPI.COMM_WORLD, 2, 2, 2)

        bgeom = mesh_B.geometry.x
        bgeom += point

        tree_A = BoundingBoxTree(mesh_A, mesh_A.topology.dim)
        tree_B = BoundingBoxTree(mesh_B, mesh_B.topology.dim)
        entities_A, entities_B = geometry.compute_entity_collisions_bb(
            tree_A, mesh_A, tree_B, mesh_B)
        assert set(entities_A) == references[i][0]
        assert set(entities_B) == references[i][1]


# --- compute_first_collision with point ---


@skip_in_parallel
def test_compute_first_collision_1d():
    reference = {1: [4]}
    p = numpy.array([0.3, 0, 0])
    mesh = UnitIntervalMesh(MPI.COMM_WORLD, 16)
    for dim in range(1, 2):
        tree = BoundingBoxTree(mesh, dim)
        first = geometry.compute_first_collision(tree, p)
        assert first in reference[dim]


@skip_in_parallel
def test_compute_first_collision_2d():
    # FIXME: This test should not use facet indices as there are no
    # guarantees on how DOLFINX numbers facets
    reference = {1: [226], 2: [136, 137]}

    p = numpy.array([0.3, 0.3, 0.0])
    mesh = UnitSquareMesh(MPI.COMM_WORLD, 16, 16)
    for dim in range(1, 3):
        tree = BoundingBoxTree(mesh, dim)
        first = geometry.compute_first_collision(tree, p)

        # FIXME: Facet test is excluded because it mistakenly relies in
        # the facet indices
        if dim != mesh.topology.dim - 1:
            assert first in reference[dim]


@skip_in_parallel
def test_compute_first_collision_3d():
    # FIXME: This test should not use facet indices as there are no
    # guarantees on how DOLFINX numbers facets
    reference = {
        1: [1364],
        2: [1967, 1968, 1970, 1972, 1974, 1976],
        3: [876, 877, 878, 879, 880, 881]
    }

    p = numpy.array([0.3, 0.3, 0.3])
    mesh = UnitCubeMesh(MPI.COMM_WORLD, 8, 8, 8)
    for dim in range(1, 4):
        tree = BoundingBoxTree(mesh, dim)
        first = cpp.geometry.compute_first_collision(tree._cpp_object, p)

        # FIXME: Face and test is excluded because it mistakenly relies
        # in the facet indices
        tdim = mesh.topology.dim
        if dim != tdim - 1 and dim != tdim - 2:
            assert first in reference[dim]


# --- compute_first_entity_collision with point ---


@skip_in_parallel
def test_compute_first_entity_collision_1d():
    reference = [4]
    p = numpy.array([0.3, 0, 0])
    mesh = UnitIntervalMesh(MPI.COMM_WORLD, 16)
    tree = BoundingBoxTree(mesh, mesh.topology.dim)
    first = geometry.compute_first_entity_collision(tree, mesh, p)
    assert first in reference


@skip_in_parallel
def test_compute_first_entity_collision_2d():
    reference = [136, 137]
    p = numpy.array([0.3, 0.3, 0.0])
    mesh = UnitSquareMesh(MPI.COMM_WORLD, 16, 16)
    tree = BoundingBoxTree(mesh, mesh.topology.dim)
    first = geometry.compute_first_entity_collision(tree, mesh, p)
    assert first in reference


@skip_in_parallel
def test_compute_first_entity_collision_3d():
    reference = [876, 877, 878, 879, 880, 881]
    p = numpy.array([0.3, 0.3, 0.3])
    mesh = UnitCubeMesh(MPI.COMM_WORLD, 8, 8, 8)
    tree = BoundingBoxTree(mesh, mesh.topology.dim)
    first = geometry.compute_first_entity_collision(tree, mesh, p)
    assert first in reference


# --- compute_closest_entity with point ---


@skip_in_parallel
def test_compute_closest_entity_1d():
    reference = (0, 1.0)
    p = numpy.array([-1.0, 0, 0])
    mesh = UnitIntervalMesh(MPI.COMM_WORLD, 16)
    tree_mid = geometry.BoundingBoxTree.create_midpoint_tree(mesh)
    tree = BoundingBoxTree(mesh, mesh.topology.dim)
    entity, distance = geometry.compute_closest_entity(tree, tree_mid, mesh, p)
    assert entity == reference[0]
    assert distance[0] == pytest.approx(reference[1], 1.0e-12)


@skip_in_parallel
def test_compute_closest_entity_2d():
    reference = (1, 1.0)
    p = numpy.array([-1.0, 0.01, 0.0])
    mesh = UnitSquareMesh(MPI.COMM_WORLD, 16, 16)
    tree_mid = geometry.BoundingBoxTree.create_midpoint_tree(mesh)
    tree = BoundingBoxTree(mesh, mesh.topology.dim)
    entity, distance = geometry.compute_closest_entity(tree, tree_mid, mesh, p)
    assert entity == reference[0]
    assert distance[0] == pytest.approx(reference[1], 1.0e-12)


@skip_in_parallel
def test_compute_closest_entity_3d():
    reference = (0, 0.1)
    p = numpy.array([0.1, 0.05, -0.1])
    mesh = UnitCubeMesh(MPI.COMM_WORLD, 8, 8, 8)
    tree = BoundingBoxTree(mesh, mesh.topology.dim)
    tree_mid = geometry.BoundingBoxTree.create_midpoint_tree(mesh)
    entity, distance = geometry.compute_closest_entity(tree, tree_mid, mesh, p)
    assert entity == reference[0]
    assert distance[0] == pytest.approx(reference[1], 1.0e-12)
