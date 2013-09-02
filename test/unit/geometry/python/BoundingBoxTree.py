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
# Last changed: 2013-09-02

import unittest
import numpy

from dolfin import BoundingBoxTree
from dolfin import UnitIntervalMesh, UnitSquareMesh, UnitCubeMesh
from dolfin import Point
from dolfin import MPI

class BoundingBoxTreeTest(unittest.TestCase):

    #--- compute_collisions with point ---

    def test_compute_collisions_point_1d(self):

        reference = {1: [4]}

        p = Point(0.3)
        mesh = UnitIntervalMesh(16)
        for dim in range(1, 2):
            tree = BoundingBoxTree()
            tree.build(mesh, dim)
            entities = tree.compute_collisions(p)
            if MPI.num_processes() == 1:
                self.assertEqual(sorted(entities), reference[dim])

    def test_compute_collisions_point_2d(self):

        reference = {1: [226],
                     2: [136, 137]}

        p = Point(0.3, 0.3)
        mesh = UnitSquareMesh(16, 16)
        for dim in range(1, 3):
            tree = BoundingBoxTree()
            tree.build(mesh, dim)
            entities = tree.compute_collisions(p)
            if MPI.num_processes() == 1:
                self.assertEqual(sorted(entities), reference[dim])

    def test_compute_collisions_point_3d(self):

        reference = {1: [1364],
                     2: [1967, 1968, 1970, 1972, 1974, 1976],
                     3: [876, 877, 878, 879, 880, 881]}

        p = Point(0.3, 0.3, 0.3)
        mesh = UnitCubeMesh(8, 8, 8)
        for dim in range(1, 4):
            tree = BoundingBoxTree()
            tree.build(mesh, dim)
            entities = tree.compute_collisions(p)
            if MPI.num_processes() == 1:
                self.assertEqual(sorted(entities), reference[dim])

    #--- compute_collisions with tree ---

    def test_compute_collisions_tree_1d(self):
        # Not yet implemented in library
        print "FIXME"

    def test_compute_collisions_tree_2d(self):

        references = [[[20, 21, 22, 23, 28, 29, 30, 31],
                      [0, 1, 2, 3, 8, 9, 10, 11]],
                      [[6, 7],
                       [24, 25]]]

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

            self.assertEqual(sorted(entities_A), references[i][0])
            self.assertEqual(sorted(entities_B), references[i][1])

    def test_compute_collisions_tree_3d(self):
        # Not yet implemented in library
        print "FIXME"

    #--- compute_entity_collisions ---

    def test_compute_entity_collisions_1d(self):

        reference = [4]

        p = Point(0.3)
        mesh = UnitIntervalMesh(16)
        tree = BoundingBoxTree()
        tree.build(mesh)
        entities = tree.compute_entity_collisions(p)
        if MPI.num_processes() == 1:
            self.assertEqual(sorted(entities), reference)

        tree = mesh.bounding_box_tree()
        entities = tree.compute_entity_collisions(p, mesh)

        if MPI.num_processes() == 1:
            self.assertEqual(sorted(entities), reference)

    def test_compute_entity_collisions_2d(self):

        reference = [136, 137]

        p = Point(0.3, 0.3)
        mesh = UnitSquareMesh(16, 16)
        tree = BoundingBoxTree()
        tree.build(mesh)
        entities = tree.compute_entity_collisions(p)
        if MPI.num_processes() == 1:
            self.assertEqual(sorted(entities), reference)

        tree = mesh.bounding_box_tree()
        entities = tree.compute_entity_collisions(p, mesh)

        if MPI.num_processes() == 1:
            self.assertEqual(sorted(entities), reference)

    def test_compute_entity_collisions_3d(self):

        reference = [876, 877, 878, 879, 880, 881]

        p = Point(0.3, 0.3, 0.3)
        mesh = UnitCubeMesh(8, 8, 8)
        tree = BoundingBoxTree()
        tree.build(mesh)
        entities = tree.compute_entity_collisions(p)
        if MPI.num_processes() == 1:
            self.assertEqual(sorted(entities), reference)

    #--- compute_entity_collisions with tree ---

    def test_compute_entity_collisions_tree_1d(self):
        # Not yet implemented in library
        print "FIXME"

    def test_compute_entity_collisions_tree_2d(self):

        references = [[[20, 21, 22, 23, 28, 29, 30, 31],
                      [0, 1, 2, 3, 8, 9, 10, 11]],
                      [[6],
                       [25]]]

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

            self.assertEqual(sorted(entities_A), references[i][0])
            self.assertEqual(sorted(entities_B), references[i][1])

    def test_compute_entity_collisions_tree_3d(self):
        # Not yet implemented in library
        print "FIXME"

    #--- compute_first_collision ---

    def test_compute_first_collision_1d(self):

        reference = {1: [4]}

        p = Point(0.3)
        mesh = UnitIntervalMesh(16)
        for dim in range(1, 2):
            tree = BoundingBoxTree()
            tree.build(mesh, dim)
            first = tree.compute_first_collision(p)
            if MPI.num_processes() == 1:
                self.assertIn(first, reference[dim])

        tree = mesh.bounding_box_tree()
        first = tree.compute_first_collision(p)
        if MPI.num_processes() == 1:
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
            if MPI.num_processes() == 1:
                self.assertIn(first, reference[dim])

        tree = mesh.bounding_box_tree()
        first = tree.compute_first_collision(p)
        if MPI.num_processes() == 1:
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
            if MPI.num_processes() == 1:
                self.assertIn(first, reference[dim])

        tree = mesh.bounding_box_tree()
        first = tree.compute_first_collision(p)
        if MPI.num_processes() == 1:
            self.assertIn(first, reference[mesh.topology().dim()])

    #--- compute_first_entity_collision ---

    def test_compute_first_entity_collision_1d(self):

        reference = [4]

        p = Point(0.3)
        mesh = UnitIntervalMesh(16)
        tree = BoundingBoxTree()
        tree.build(mesh)
        first = tree.compute_first_entity_collision(p)
        if MPI.num_processes() == 1:
            self.assertIn(first, reference)

        tree = mesh.bounding_box_tree()
        first = tree.compute_first_entity_collision(p, mesh)
        if MPI.num_processes() == 1:
            self.assertIn(first, reference)

    def test_compute_first_entity_collision_2d(self):

        reference = [136, 137]

        p = Point(0.3, 0.3)
        mesh = UnitSquareMesh(16, 16)
        tree = BoundingBoxTree()
        tree.build(mesh)
        first = tree.compute_first_entity_collision(p)
        if MPI.num_processes() == 1:
            self.assertIn(first, reference)


        tree = mesh.bounding_box_tree()
        first = tree.compute_first_entity_collision(p, mesh)
        if MPI.num_processes() == 1:
            self.assertIn(first, reference)

    def test_compute_first_entity_collision_3d(self):

        reference = [876, 877, 878, 879, 880, 881]

        p = Point(0.3, 0.3, 0.3)
        mesh = UnitCubeMesh(8, 8, 8)
        tree = BoundingBoxTree()
        tree.build(mesh)
        first = tree.compute_first_entity_collision(p)
        if MPI.num_processes() == 1:
            self.assertIn(first, reference)

        tree = mesh.bounding_box_tree()
        first = tree.compute_first_entity_collision(p, mesh)
        if MPI.num_processes() == 1:
            self.assertIn(first, reference)

    #--- compute_closest_entity ---

    def test_compute_closest_entity_1d(self):

        reference = (0, 1.0)

        p = Point(-1.0)
        mesh = UnitIntervalMesh(16)
        tree = BoundingBoxTree()
        tree.build(mesh)
        entity, distance = tree.compute_closest_entity(p)

        if MPI.num_processes() == 1:
            self.assertEqual(entity, reference[0])
            self.assertAlmostEqual(distance, reference[1])

        tree = mesh.bounding_box_tree()
        entity, distance = tree.compute_closest_entity(p, mesh)
        if MPI.num_processes() == 1:
            self.assertEqual(entity, reference[0])
            self.assertAlmostEqual(distance, reference[1])

    def test_compute_closest_entity_2d(self):

        reference = (0, numpy.sqrt(2.0))

        p = Point(-1.0, -1.0)
        mesh = UnitSquareMesh(16, 16)
        tree = BoundingBoxTree()
        tree.build(mesh)
        entity, distance = tree.compute_closest_entity(p)

        if MPI.num_processes() == 1:
            self.assertEqual(entity, reference[0])
            self.assertAlmostEqual(distance, reference[1])

        tree = mesh.bounding_box_tree()
        entity, distance = tree.compute_closest_entity(p, mesh)
        if MPI.num_processes() == 1:
            self.assertEqual(entity, reference[0])
            self.assertAlmostEqual(distance, reference[1])

    def test_compute_closest_entity_3d(self):

        reference = (2, numpy.sqrt(3.0))

        p = Point(-1.0, -1.0, -1.0)
        mesh = UnitCubeMesh(8, 8, 8)
        tree = BoundingBoxTree()
        tree.build(mesh)
        entity, distance = tree.compute_closest_entity(p)

        if MPI.num_processes() == 1:
            self.assertEqual(entity, reference[0])
            self.assertAlmostEqual(distance, reference[1])

        tree = mesh.bounding_box_tree()
        entity, distance = tree.compute_closest_entity(p, mesh)
        if MPI.num_processes() == 1:
            self.assertEqual(entity, reference[0])
            self.assertAlmostEqual(distance, reference[1])

if __name__ == "__main__":
    print ""
    print "Testing BoundingBoxTree"
    print "------------------------------------------------"

    print "FIXME: Temporary testing"
    unittest.main()
