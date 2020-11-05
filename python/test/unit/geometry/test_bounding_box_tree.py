# Copyright (C) 2013-2020 Anders Logg, Jørgen S. Dokken
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for BoundingBoxTree"""

import numpy
import pytest
from dolfinx import BoxMesh, UnitCubeMesh, UnitIntervalMesh, UnitSquareMesh, geometry, cpp
from dolfinx.geometry import BoundingBoxTree
from dolfinx_utils.test.skips import skip_in_parallel
from mpi4py import MPI


def extract_geometricial_data(mesh, dim, entities):
    """
    For a set of entities in a mesh, return the coordinates of the vertices
    """
    mesh_nodes = []
    geom = mesh.geometry
    g_indices = cpp.mesh.entities_to_geometry(mesh, dim,
                                              numpy.array(entities, dtype=numpy.int32),
                                              False)
    for cell in g_indices:
        nodes = numpy.zeros((len(cell), 3), dtype=numpy.float64)
        for j, entity in enumerate(cell):
            nodes[j] = geom.x[entity]
        mesh_nodes.append(nodes)
    return mesh_nodes


@pytest.mark.parametrize("padding", [True, False])
@skip_in_parallel
def test_padded_bbox(padding):
    """
    Test collision between two meshes separated by a distance of epsilon,
    and check if padding the mesh creates a possible collision
    """
    eps = 1e-12
    x0 = numpy.array([0, 0, 0])
    x1 = numpy.array([1, 1, 1 - eps])
    mesh_0 = BoxMesh(MPI.COMM_WORLD, [x0, x1], [1, 1, 2], cpp.mesh.CellType.hexahedron)
    x2 = numpy.array([0, 0, 1 + eps])
    x3 = numpy.array([1, 1, 2])
    mesh_1 = BoxMesh(MPI.COMM_WORLD, [x2, x3], [1, 1, 2], cpp.mesh.CellType.hexahedron)
    if padding:
        pad = eps
    else:
        pad = 0
    bbox_0 = BoundingBoxTree(mesh_0, mesh_0.topology.dim, padding=pad)
    bbox_1 = BoundingBoxTree(mesh_1, mesh_1.topology.dim, padding=pad)
    collisions = cpp.geometry.compute_collisions(bbox_0._cpp_object, bbox_1._cpp_object)
    if padding:
        assert(len(collisions) == 1)
        # Check that the colliding elements are separated by a distance 2*epsilon
        element_0 = extract_geometricial_data(mesh_0, mesh_0.topology.dim, [collisions[0][0]])[0]
        element_1 = extract_geometricial_data(mesh_1, mesh_1.topology.dim, [collisions[0][1]])[0]
        distance = numpy.linalg.norm(cpp.geometry.compute_distance_gjk(element_0, element_1))
        assert(numpy.isclose(distance, 2 * eps))
    else:
        assert(len(collisions) == 0)


@skip_in_parallel
def test_compute_collisions_point_1d():

    reference = {1: set([4])}

    p = numpy.array([0.3, 0, 0])
    mesh = UnitIntervalMesh(MPI.COMM_WORLD, 16)
    for dim in range(1, 2):
        tree = BoundingBoxTree(mesh, mesh.topology.dim)
        entities = geometry.compute_collisions_point(tree, p)
        assert set(entities) == reference[dim]


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
    entities = geometry.compute_collisions(tree_A, tree_B)

    entities_A = set([q[0] for q in entities])
    entities_B = set([q[1] for q in entities])

    assert entities_A == cells[0]
    assert entities_B == cells[1]


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
    entities = geometry.compute_collisions(tree_A, tree_B)

    entities_A = set([q[0] for q in entities])
    entities_B = set([q[1] for q in entities])
    assert entities_A == set(cells[0])
    assert entities_B == set(cells[1])


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
        entities = geometry.compute_collisions(tree_A, tree_B)

        entities_A = set([q[0] for q in entities])
        entities_B = set([q[1] for q in entities])
        assert entities_A == references[i][0]
        assert entities_B == references[i][1]


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
