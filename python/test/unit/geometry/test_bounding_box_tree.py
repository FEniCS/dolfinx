"""Unit tests for BoundingBoxTree"""

# Copyright (C) 2013-2014 Anders Logg
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import pytest
import numpy

from dolfin import BoundingBoxTree
from dolfin import UnitIntervalMesh, UnitSquareMesh, UnitCubeMesh
from dolfin import Point
from dolfin import MeshEntity
from dolfin import MPI
from dolfin_utils.test import skip_in_parallel


#--- compute_collisions with point ---

@skip_in_parallel
def test_compute_collisions_point_1d():

    reference = {1: set([4])}

    p = Point(0.3)
    mesh = UnitIntervalMesh(MPI.comm_world, 16)
    for dim in range(1, 2):
        tree = BoundingBoxTree(mesh.geometry().dim())
        tree.build(mesh, dim)
        entities = tree.compute_collisions(p)
        assert set(entities) == reference[dim]

@skip_in_parallel
def test_compute_collisions_point_2d():

    reference = {1: set([226]),
                  2: set([136, 137])}

    p = Point(0.3, 0.3)
    mesh = UnitSquareMesh(MPI.comm_world, 16, 16)
    for dim in range(1, 3):
        tree = BoundingBoxTree(mesh.geometry().dim())
        tree.build(mesh, dim)
        entities = tree.compute_collisions(p)
        for e in entities:
            ent = MeshEntity(mesh, dim, e)
            mp = ent.midpoint()
            x = (mp[0], mp[1])
            print("test: {}".format(x))
        #assert set(entities) == reference[dim]

@skip_in_parallel
def test_compute_collisions_point_3d():

    reference = {1: set([1364]),
                  2: set([1967, 1968, 1970, 1972, 1974, 1976]),
                  3: set([876, 877, 878, 879, 880, 881])}

    p = Point(0.3, 0.3, 0.3)
    mesh = UnitCubeMesh(MPI.comm_world, 8, 8, 8)
    for dim in range(1, 4):
        tree = BoundingBoxTree(mesh.geometry().dim())
        tree.build(mesh, dim)
        entities = tree.compute_collisions(p)

        # FIXME: Face and edges tests are excluded because test
        # mistakingly relies on the face and edge indices
        tdim = mesh.topology().dim()
        if dim != tdim - 1 and dim != tdim - 2:
            assert set(entities) == reference[dim]

#--- compute_collisions with tree ---

@skip_in_parallel
def test_compute_collisions_tree_1d():

    references = [[set([8, 9, 10, 11, 12, 13, 14, 15]),
                    set([0, 1, 2, 3, 4, 5, 6, 7])],
                  [set([14, 15]),
                    set([0, 1])]]

    points = [Point(0.52), Point(0.9)]

    for i, point in enumerate(points):

        mesh_A = UnitIntervalMesh(MPI.comm_world, 16)
        mesh_B = UnitIntervalMesh(MPI.comm_world, 16)

        bgeom = mesh_B.geometry().x()
        bgeom += point[0]

        tree_A = BoundingBoxTree(1)
        tree_A.build(mesh_A, 1)

        tree_B = BoundingBoxTree(1)
        tree_B.build(mesh_B, 1)

        entities_A, entities_B = tree_A.compute_collisions(tree_B)

        assert set(entities_A) == references[i][0]
        assert set(entities_B) == references[i][1]

@skip_in_parallel
def test_compute_collisions_tree_2d():

    references = [[set([20, 21, 22, 23, 28, 29, 30, 31]),
                    set([0, 1, 2, 3, 8, 9, 10, 11])],
                  [set([6, 7]),
                    set([24, 25])]]

    points = [Point(0.52, 0.51), Point(0.9, -0.9)]

    for i, point in enumerate(points):

        mesh_A = UnitSquareMesh(MPI.comm_world, 4, 4)
        mesh_B = UnitSquareMesh(MPI.comm_world, 4, 4)

        bgeom = mesh_B.geometry().x()
        bgeom += point.array()[:2]

        tree_A = BoundingBoxTree(2)
        tree_A.build(mesh_A, 2)

        tree_B = BoundingBoxTree(2)
        tree_B.build(mesh_B, 2)

        entities_A, entities_B = tree_A.compute_collisions(tree_B)

        assert set(entities_A) == references[i][0]
        assert set(entities_B) == references[i][1]

@skip_in_parallel
def test_compute_collisions_tree_3d():

    references = [[set([18, 19, 20, 21, 22, 23, 42, 43, 44, 45, 46, 47]),
                    set([0, 1, 2, 3, 4, 5, 24, 25, 26, 27, 28, 29])],
                  [set([6, 7, 8, 9, 10, 11, 30, 31, 32, 33, 34, 35]),
                    set([12, 13, 14, 15, 16, 17, 36, 37, 38, 39, 40, 41])]]

    points = [Point(0.52, 0.51, 0.3), Point(0.9, -0.9, 0.3)]

    for i, point in enumerate(points):

        mesh_A = UnitCubeMesh(MPI.comm_world, 2, 2, 2)
        mesh_B = UnitCubeMesh(MPI.comm_world, 2, 2, 2)

        bgeom = mesh_B.geometry().x()
        bgeom += point.array()

        tree_A = BoundingBoxTree(3)
        tree_A.build(mesh_A, 3)

        tree_B = BoundingBoxTree(3)
        tree_B.build(mesh_B, 3)

        entities_A, entities_B = tree_A.compute_collisions(tree_B)

        assert set(entities_A) == references[i][0]
        assert set(entities_B) == references[i][1]

#--- compute_entity_collisions with point ---

@skip_in_parallel
def test_compute_entity_collisions_1d():

    reference = set([4])

    p = Point(0.3)
    mesh = UnitIntervalMesh(MPI.comm_world, 16)

    tree = BoundingBoxTree(mesh.geometry().dim())
    tree.build(mesh, mesh.topology().dim())
    entities = tree.compute_entity_collisions(p, mesh)
    assert set(entities) == reference

    tree = mesh.bounding_box_tree()
    entities = tree.compute_entity_collisions(p, mesh)
    assert set(entities) == reference

@skip_in_parallel
def test_compute_entity_collisions_2d():

    reference = set([136, 137])

    p = Point(0.3, 0.3)
    mesh = UnitSquareMesh(MPI.comm_world, 16, 16)

    tree = BoundingBoxTree(mesh.geometry().dim())
    tree.build(mesh, mesh.topology().dim())
    entities = tree.compute_entity_collisions(p, mesh)
    assert set(entities) == reference

    tree = mesh.bounding_box_tree()
    entities = tree.compute_entity_collisions(p, mesh)
    assert set(entities) == reference

@skip_in_parallel
def test_compute_entity_collisions_3d():

    reference = set([876, 877, 878, 879, 880, 881])

    p = Point(0.3, 0.3, 0.3)
    mesh = UnitCubeMesh(MPI.comm_world, 8, 8, 8)

    tree = BoundingBoxTree(mesh.geometry().dim())
    tree.build(mesh, mesh.topology().dim())
    entities = tree.compute_entity_collisions(p, mesh)
    assert set(entities) == reference

#--- compute_entity_collisions with tree ---

@skip_in_parallel
def test_compute_entity_collisions_tree_1d():

    references = [[set([8, 9, 10, 11, 12, 13, 14, 15]),
                    set([0, 1, 2, 3, 4, 5, 6, 7])],
                  [set([14, 15]),
                    set([0, 1])]]

    points = [Point(0.52), Point(0.9)]

    for i, point in enumerate(points):

        mesh_A = UnitIntervalMesh(MPI.comm_world, 16)
        mesh_B = UnitIntervalMesh(MPI.comm_world, 16)

        bgeom = mesh_B.geometry().x()
        bgeom += point[0]

        tree_A = BoundingBoxTree(1)
        tree_A.build(mesh_A, 1)

        tree_B = BoundingBoxTree(1)
        tree_B.build(mesh_B, 1)

        entities_A, entities_B = tree_A.compute_entity_collisions(tree_B, mesh_A, mesh_B)

        assert set(entities_A) == references[i][0]
        assert set(entities_B) == references[i][1]

@skip_in_parallel
def test_compute_entity_collisions_tree_2d():

    references = [[set([20, 21, 22, 23, 28, 29, 30, 31]),
                    set([0, 1, 2, 3, 8, 9, 10, 11])],
                  [set([6]),
                    set([25])]]

    points = [Point(0.52, 0.51), Point(0.9, -0.9)]

    for i, point in enumerate(points):

        mesh_A = UnitSquareMesh(MPI.comm_world, 4, 4)
        mesh_B = UnitSquareMesh(MPI.comm_world, 4, 4)

        bgeom = mesh_B.geometry().x()
        bgeom += point.array()[:2]

        tree_A = BoundingBoxTree(2)
        tree_A.build(mesh_A, 2)

        tree_B = BoundingBoxTree(2)
        tree_B.build(mesh_B, 2)

        entities_A, entities_B = tree_A.compute_entity_collisions(tree_B, mesh_A, mesh_B)

        assert set(entities_A) == references[i][0]
        assert set(entities_B) == references[i][1]

@skip_in_parallel
def test_compute_entity_collisions_tree_3d():

    references = [[set([18, 19, 20, 21, 22, 23, 42, 43, 44, 45, 46, 47]),
                    set([0, 1, 2, 3, 4, 5, 24, 25, 26, 27, 28, 29])],
                  [set([7, 8, 30, 31, 32]),
                    set([15, 16, 17, 39, 41])]]

    points = [Point(0.52, 0.51, 0.3), Point(0.9, -0.9, 0.3)]

    for i, point in enumerate(points):

        mesh_A = UnitCubeMesh(MPI.comm_world, 2, 2, 2)
        mesh_B = UnitCubeMesh(MPI.comm_world, 2, 2, 2)

        bgeom = mesh_B.geometry().x()
        bgeom += point.array()

        tree_A = BoundingBoxTree(3)
        tree_A.build(mesh_A, 3)

        tree_B = BoundingBoxTree(3)
        tree_B.build(mesh_B, 3)

        entities_A, entities_B = tree_A.compute_entity_collisions(tree_B, mesh_A, mesh_B)

        assert set(entities_A) == references[i][0]
        assert set(entities_B) == references[i][1]

#--- compute_first_collision with point ---

@skip_in_parallel
def test_compute_first_collision_1d():

    reference = {1: [4]}

    p = Point(0.3)
    mesh = UnitIntervalMesh(MPI.comm_world, 16)
    for dim in range(1, 2):
        tree = BoundingBoxTree(mesh.geometry().dim())
        tree.build(mesh, dim)
        first = tree.compute_first_collision(p)
        assert first in reference[dim]

    tree = mesh.bounding_box_tree()
    first = tree.compute_first_collision(p)
    assert first in reference[mesh.topology().dim()]

@skip_in_parallel
def test_compute_first_collision_2d():

    # FIXME: This test should not use facet indices as there are no guarantees
    # on how DOLFIN numbers facets
    reference = {1: [226],
                  2: [136, 137]}

    p = Point(0.3, 0.3)
    mesh = UnitSquareMesh(MPI.comm_world, 16, 16)
    for dim in range(1, 3):
        tree = BoundingBoxTree(mesh.geometry().dim())
        tree.build(mesh, dim)
        first = tree.compute_first_collision(p)

        # FIXME: Facet test is excluded because it mistakingly relies in the
        # facet indices
        if dim != mesh.topology().dim() - 1:
            assert first in reference[dim]

    tree = mesh.bounding_box_tree()
    first = tree.compute_first_collision(p)
    assert first in reference[mesh.topology().dim()]

@skip_in_parallel
def test_compute_first_collision_3d():

    # FIXME: This test should not use facet indices as there are no guarantees
    # on how DOLFIN numbers facets

    reference = {1: [1364],
                  2: [1967, 1968, 1970, 1972, 1974, 1976],
                  3: [876, 877, 878, 879, 880, 881]}

    p = Point(0.3, 0.3, 0.3)
    mesh = UnitCubeMesh(MPI.comm_world, 8, 8, 8)
    for dim in range(1, 4):
        tree = BoundingBoxTree(mesh.geometry().dim())
        tree.build(mesh, dim)
        first = tree.compute_first_collision(p)

        # FIXME: Face and test is excluded because it mistakingly
        # relies in the facet indices
        tdim = mesh.topology().dim()
        if dim != tdim - 1 and dim != tdim - 2:
            assert first in reference[dim]

    tree = mesh.bounding_box_tree()
    first = tree.compute_first_collision(p)
    assert first in reference[mesh.topology().dim()]

#--- compute_first_entity_collision with point ---

@skip_in_parallel
def test_compute_first_entity_collision_1d():

    reference = [4]

    p = Point(0.3)
    mesh = UnitIntervalMesh(MPI.comm_world, 16)
    tree = BoundingBoxTree(mesh.geometry().dim())
    tree.build(mesh, mesh.topology().dim())
    first = tree.compute_first_entity_collision(p, mesh)
    assert first in reference

    tree = mesh.bounding_box_tree()
    first = tree.compute_first_entity_collision(p, mesh)
    assert first in reference

@skip_in_parallel
def test_compute_first_entity_collision_2d():

    reference = [136, 137]

    p = Point(0.3, 0.3)
    mesh = UnitSquareMesh(MPI.comm_world, 16, 16)
    tree = BoundingBoxTree(mesh.geometry().dim())
    tree.build(mesh, mesh.topology().dim())
    first = tree.compute_first_entity_collision(p, mesh)
    assert first in reference

    tree = mesh.bounding_box_tree()
    first = tree.compute_first_entity_collision(p, mesh)
    assert first in reference

@skip_in_parallel
def test_compute_first_entity_collision_3d():

    reference = [876, 877, 878, 879, 880, 881]

    p = Point(0.3, 0.3, 0.3)
    mesh = UnitCubeMesh(MPI.comm_world, 8, 8, 8)
    tree = BoundingBoxTree(mesh.geometry().dim())
    tree.build(mesh, mesh.topology().dim())
    first = tree.compute_first_entity_collision(p, mesh)
    assert first in reference

    tree = mesh.bounding_box_tree()
    first = tree.compute_first_entity_collision(p, mesh)
    assert first in reference

#--- compute_closest_entity with point ---

@skip_in_parallel
def test_compute_closest_entity_1d():

    reference = (0, 1.0)

    p = Point(-1.0)
    mesh = UnitIntervalMesh(MPI.comm_world, 16)
    tree = BoundingBoxTree(mesh.geometry().dim())
    tree.build(mesh, mesh.topology().dim())
    entity, distance = tree.compute_closest_entity(p, mesh)

    assert entity == reference[0]
    assert round(distance - reference[1], 7) == 0

    tree = mesh.bounding_box_tree()
    entity, distance = tree.compute_closest_entity(p, mesh)
    assert entity == reference[0]
    assert round(distance - reference[1], 7) == 0

@skip_in_parallel
def test_compute_closest_entity_2d():

    reference = (1, 1.0)

    p = Point(-1.0, 0.01)
    mesh = UnitSquareMesh(MPI.comm_world, 16, 16)
    tree = BoundingBoxTree(mesh.geometry().dim())
    tree.build(mesh, mesh.topology().dim())
    entity, distance = tree.compute_closest_entity(p, mesh)

    assert entity == reference[0]
    assert round(distance - reference[1], 7) == 0

    tree = mesh.bounding_box_tree()
    entity, distance = tree.compute_closest_entity(p, mesh)
    assert entity == reference[0]
    assert round(distance - reference[1], 7) == 0

@skip_in_parallel
def test_compute_closest_entity_3d():

    reference = (0, 0.1)

    p = Point(0.1, 0.05, -0.1)
    mesh = UnitCubeMesh(MPI.comm_world, 8, 8, 8)
    tree = BoundingBoxTree(mesh.geometry().dim())
    tree.build(mesh, mesh.topology().dim())
    entity, distance = tree.compute_closest_entity(p, mesh)

    assert entity == reference[0]
    assert round(distance - reference[1], 7) == 0

    tree = mesh.bounding_box_tree()
    entity, distance = tree.compute_closest_entity(p, mesh)
    assert entity == reference[0]
    assert round(distance - reference[1], 7) == 0
