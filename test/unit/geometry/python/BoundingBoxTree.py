"""Unit tests for BoundingBoxTree"""

# Copyright (C) 2013 Anders Logg
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
# First added:  2013-04-15
# Last changed: 2014-05-30

import unittest
import numpy

from dolfin import BoundingBoxTree
from dolfin import UnitIntervalMesh, UnitSquareMesh, UnitCubeMesh
from dolfin import Point
from dolfin import MPI, mpi_comm_world

@unittest.skipIf(MPI.size(mpi_comm_world()) > 1, "Skipping unit test(s) not working in parallel")
class BoundingBoxTreeTest(unittest.TestCase):

    #--- compute_collisions with point ---

    def test_compute_collisions_point_1d(self):

        reference = {1: set([4])}

        p = Point(0.3)
        mesh = UnitIntervalMesh(16)
        for dim in range(1, 2):
            tree = BoundingBoxTree()
            tree.build(mesh, dim)
            entities = tree.compute_collisions(p)
            self.assertEqual(set(entities), reference[dim])

    def test_compute_collisions_point_2d(self):

        reference = {1: set([226]),
                     2: set([136, 137])}

        p = Point(0.3, 0.3)
        mesh = UnitSquareMesh(16, 16)
        for dim in range(1, 3):
            tree = BoundingBoxTree()
            tree.build(mesh, dim)
            entities = tree.compute_collisions(p)
            self.assertEqual(set(entities), reference[dim])

    def test_compute_collisions_point_3d(self):

        reference = {1: set([1364]),
                     2: set([1967, 1968, 1970, 1972, 1974, 1976]),
                     3: set([876, 877, 878, 879, 880, 881])}

        p = Point(0.3, 0.3, 0.3)
        mesh = UnitCubeMesh(8, 8, 8)
        for dim in range(1, 4):
            tree = BoundingBoxTree()
            tree.build(mesh, dim)
            entities = tree.compute_collisions(p)
            self.assertEqual(set(entities), reference[dim])

    #--- compute_collisions with tree ---

    def test_compute_collisions_tree_1d(self):

        references = [[set([8, 9, 10, 11, 12, 13, 14, 15]),
                       set([0, 1, 2, 3, 4, 5, 6, 7])],
                      [set([14, 15]),
                       set([0, 1])]]

        points = [Point(0.52), Point(0.9)]

        for i, point in enumerate(points):

            mesh_A = UnitIntervalMesh(16)
            mesh_B = UnitIntervalMesh(16)

            mesh_B.translate(point)

            tree_A = BoundingBoxTree()
            tree_A.build(mesh_A)

            tree_B = BoundingBoxTree()
            tree_B.build(mesh_B)

            entities_A, entities_B = tree_A.compute_collisions(tree_B)

            self.assertEqual(set(entities_A), references[i][0])
            self.assertEqual(set(entities_B), references[i][1])

    def test_compute_collisions_tree_2d(self):

        references = [[set([20, 21, 22, 23, 28, 29, 30, 31]),
                       set([0, 1, 2, 3, 8, 9, 10, 11])],
                      [set([6, 7]),
                       set([24, 25])]]

        points = [Point(0.52, 0.51), Point(0.9, -0.9)]

        for i, point in enumerate(points):

            mesh_A = UnitSquareMesh(4, 4)
            mesh_B = UnitSquareMesh(4, 4)

            mesh_B.translate(point)

            tree_A = BoundingBoxTree()
            tree_A.build(mesh_A)

            tree_B = BoundingBoxTree()
            tree_B.build(mesh_B)

            entities_A, entities_B = tree_A.compute_collisions(tree_B)

            self.assertEqual(set(entities_A), references[i][0])
            self.assertEqual(set(entities_B), references[i][1])

    def test_compute_collisions_tree_3d(self):

        references = [[set([18, 19, 20, 21, 22, 23, 42, 43, 44, 45, 46, 47]),
                       set([0, 1, 2, 3, 4, 5, 24, 25, 26, 27, 28, 29])],
                      [set([6, 7, 8, 9, 10, 11, 30, 31, 32, 33, 34, 35]),
                       set([12, 13, 14, 15, 16, 17, 36, 37, 38, 39, 40, 41])]]

        points = [Point(0.52, 0.51, 0.3), Point(0.9, -0.9, 0.3)]

        for i, point in enumerate(points):

            mesh_A = UnitCubeMesh(2, 2, 2)
            mesh_B = UnitCubeMesh(2, 2, 2)

            mesh_B.translate(point)

            tree_A = BoundingBoxTree()
            tree_A.build(mesh_A)

            tree_B = BoundingBoxTree()
            tree_B.build(mesh_B)

            entities_A, entities_B = tree_A.compute_collisions(tree_B)

            self.assertEqual(set(entities_A), references[i][0])
            self.assertEqual(set(entities_B), references[i][1])

    #--- compute_entity_collisions ---

    def test_compute_entity_collisions_1d(self):

        reference = set([4])

        p = Point(0.3)
        mesh = UnitIntervalMesh(16)

        tree = BoundingBoxTree()
        tree.build(mesh)
        entities = tree.compute_entity_collisions(p)
        self.assertEqual(set(entities), reference)

        tree = mesh.bounding_box_tree()
        entities = tree.compute_entity_collisions(p)
        self.assertEqual(set(entities), reference)

    def test_compute_entity_collisions_2d(self):

        reference = set([136, 137])

        p = Point(0.3, 0.3)
        mesh = UnitSquareMesh(16, 16)

        tree = BoundingBoxTree()
        tree.build(mesh)
        entities = tree.compute_entity_collisions(p)
        self.assertEqual(set(entities), reference)

        tree = mesh.bounding_box_tree()
        entities = tree.compute_entity_collisions(p)
        self.assertEqual(set(entities), reference)

    def test_compute_entity_collisions_3d(self):

        reference = set([876, 877, 878, 879, 880, 881])

        p = Point(0.3, 0.3, 0.3)
        mesh = UnitCubeMesh(8, 8, 8)

        tree = BoundingBoxTree()
        tree.build(mesh)
        entities = tree.compute_entity_collisions(p)
        self.assertEqual(set(entities), reference)

    #--- compute_entity_collisions with tree ---

    def test_compute_entity_collisions_tree_1d(self):

        references = [[set([8, 9, 10, 11, 12, 13, 14, 15]),
                       set([0, 1, 2, 3, 4, 5, 6, 7])],
                      [set([14, 15]),
                       set([0, 1])]]

        points = [Point(0.52), Point(0.9)]

        for i, point in enumerate(points):

            mesh_A = UnitIntervalMesh(16)
            mesh_B = UnitIntervalMesh(16)

            mesh_B.translate(point)

            tree_A = BoundingBoxTree()
            tree_A.build(mesh_A)

            tree_B = BoundingBoxTree()
            tree_B.build(mesh_B)

            entities_A, entities_B = tree_A.compute_entity_collisions(tree_B)

            self.assertEqual(set(entities_A), references[i][0])
            self.assertEqual(set(entities_B), references[i][1])

    def test_compute_entity_collisions_tree_2d(self):

        references = [[set([20, 21, 22, 23, 28, 29, 30, 31]),
                       set([0, 1, 2, 3, 8, 9, 10, 11])],
                      [set([6]),
                       set([25])]]

        points = [Point(0.52, 0.51), Point(0.9, -0.9)]

        for i, point in enumerate(points):

            mesh_A = UnitSquareMesh(4, 4)
            mesh_B = UnitSquareMesh(4, 4)

            mesh_B.translate(point)

            tree_A = BoundingBoxTree()
            tree_A.build(mesh_A)

            tree_B = BoundingBoxTree()
            tree_B.build(mesh_B)

            entities_A, entities_B = tree_A.compute_entity_collisions(tree_B)

            self.assertEqual(set(entities_A), references[i][0])
            self.assertEqual(set(entities_B), references[i][1])

    def test_compute_entity_collisions_tree_3d(self):

        references = [[set([18, 19, 20, 21, 22, 23, 42, 43, 44, 45, 46, 47]),
                       set([0, 1, 2, 3, 4, 5, 24, 25, 26, 27, 28, 29])],
                      [set([7, 8, 30, 31, 32]),
                       set([15, 16, 17, 39, 41])]]

        points = [Point(0.52, 0.51, 0.3), Point(0.9, -0.9, 0.3)]

        for i, point in enumerate(points):

            mesh_A = UnitCubeMesh(2, 2, 2)
            mesh_B = UnitCubeMesh(2, 2, 2)

            mesh_B.translate(point)

            tree_A = BoundingBoxTree()
            tree_A.build(mesh_A)

            tree_B = BoundingBoxTree()
            tree_B.build(mesh_B)

            entities_A, entities_B = tree_A.compute_entity_collisions(tree_B)

            self.assertEqual(set(entities_A), references[i][0])
            self.assertEqual(set(entities_B), references[i][1])

    #--- compute_first_collision ---

    def test_compute_first_collision_1d(self):

        reference = {1: [4]}

        p = Point(0.3)
        mesh = UnitIntervalMesh(16)
        for dim in range(1, 2):
            tree = BoundingBoxTree()
            tree.build(mesh, dim)
            first = tree.compute_first_collision(p)
            self.assertIn(first, reference[dim])

        tree = mesh.bounding_box_tree()
        first = tree.compute_first_collision(p)
        self.assertIn(first, reference[mesh.topology().dim()])

    def test_compute_first_collision_2d(self):

        reference = {1: [226],
                     2: [136, 137]}

        p = Point(0.3, 0.3)
        mesh = UnitSquareMesh(16, 16)
        for dim in range(1, 3):
            tree = BoundingBoxTree()
            tree.build(mesh, dim)
            first = tree.compute_first_collision(p)
            self.assertIn(first, reference[dim])

        tree = mesh.bounding_box_tree()
        first = tree.compute_first_collision(p)
        self.assertIn(first, reference[mesh.topology().dim()])

    def test_compute_first_collision_3d(self):

        reference = {1: [1364],
                     2: [1967, 1968, 1970, 1972, 1974, 1976],
                     3: [876, 877, 878, 879, 880, 881]}

        p = Point(0.3, 0.3, 0.3)
        mesh = UnitCubeMesh(8, 8, 8)
        for dim in range(1, 4):
            tree = BoundingBoxTree()
            tree.build(mesh, dim)
            first = tree.compute_first_collision(p)
            self.assertIn(first, reference[dim])

        tree = mesh.bounding_box_tree()
        first = tree.compute_first_collision(p)
        self.assertIn(first, reference[mesh.topology().dim()])

    #--- compute_first_entity_collision ---

    def test_compute_first_entity_collision_1d(self):

        reference = [4]

        p = Point(0.3)
        mesh = UnitIntervalMesh(16)
        tree = BoundingBoxTree()
        tree.build(mesh)
        first = tree.compute_first_entity_collision(p)
        self.assertIn(first, reference)

        tree = mesh.bounding_box_tree()
        first = tree.compute_first_entity_collision(p)
        self.assertIn(first, reference)

    def test_compute_first_entity_collision_2d(self):

        reference = [136, 137]

        p = Point(0.3, 0.3)
        mesh = UnitSquareMesh(16, 16)
        tree = BoundingBoxTree()
        tree.build(mesh)
        first = tree.compute_first_entity_collision(p)
        self.assertIn(first, reference)

        tree = mesh.bounding_box_tree()
        first = tree.compute_first_entity_collision(p)
        self.assertIn(first, reference)

    def test_compute_first_entity_collision_3d(self):

        reference = [876, 877, 878, 879, 880, 881]

        p = Point(0.3, 0.3, 0.3)
        mesh = UnitCubeMesh(8, 8, 8)
        tree = BoundingBoxTree()
        tree.build(mesh)
        first = tree.compute_first_entity_collision(p)
        self.assertIn(first, reference)

        tree = mesh.bounding_box_tree()
        first = tree.compute_first_entity_collision(p)
        self.assertIn(first, reference)

    #--- compute_closest_entity ---

    def test_compute_closest_entity_1d(self):

        reference = (0, 1.0)

        p = Point(-1.0)
        mesh = UnitIntervalMesh(16)
        tree = BoundingBoxTree()
        tree.build(mesh)
        entity, distance = tree.compute_closest_entity(p)

        self.assertEqual(entity, reference[0])
        self.assertAlmostEqual(distance, reference[1])

        tree = mesh.bounding_box_tree()
        entity, distance = tree.compute_closest_entity(p)
        self.assertEqual(entity, reference[0])
        self.assertAlmostEqual(distance, reference[1])

    def test_compute_closest_entity_2d(self):

        reference = (1, 1.0)

        p = Point(-1.0, 0.01)
        mesh = UnitSquareMesh(16, 16)
        tree = BoundingBoxTree()
        tree.build(mesh)
        entity, distance = tree.compute_closest_entity(p)

        self.assertEqual(entity, reference[0])
        self.assertAlmostEqual(distance, reference[1])

        tree = mesh.bounding_box_tree()
        entity, distance = tree.compute_closest_entity(p)
        self.assertEqual(entity, reference[0])
        self.assertAlmostEqual(distance, reference[1])

    def test_compute_closest_entity_3d(self):

        reference = (0, 0.1)

        p = Point(0.1, 0.05, -0.1)
        mesh = UnitCubeMesh(8, 8, 8)
        tree = BoundingBoxTree()
        tree.build(mesh)
        entity, distance = tree.compute_closest_entity(p)

        self.assertEqual(entity, reference[0])
        self.assertAlmostEqual(distance, reference[1])

        tree = mesh.bounding_box_tree()
        entity, distance = tree.compute_closest_entity(p)
        self.assertEqual(entity, reference[0])
        self.assertAlmostEqual(distance, reference[1])

if __name__ == "__main__":
    print ""
    print "Testing BoundingBoxTree"
    print "------------------------------------------------"
    unittest.main()
