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
# Last changed: 2013-05-22

import unittest

from dolfin import BoundingBoxTree
from dolfin import UnitIntervalMesh, UnitSquareMesh, UnitCubeMesh
from dolfin import Point

class BoundingBoxTreeTest(unittest.TestCase):

    #compute_collisions(const Point& point) const;
    #compute_entity_collisions(const Point& point) const;
    #compute_first_collision(const Point& point) const;
    #compute_first_entity_collision(const Point& point) const;
    #compute_closest_entity(const Point& point) const;

    #--- compute_collisions ---

    def test_compute_collisions_1d(self):

        reference = {1: [4]}

        p = Point(0.3)
        mesh = UnitIntervalMesh(16)
        for dim in range(1, 2):
            tree = BoundingBoxTree(mesh, dim)
            tree.build()
            entities = tree.compute_collisions(p)
            self.assertEqual(sorted(entities), reference[dim])

    def test_compute_collisions_2d(self):

        reference = {1: [226],
                     2: [136, 137]}

        p = Point(0.3, 0.3)
        mesh = UnitSquareMesh(16, 16)
        for dim in range(1, 3):
            tree = BoundingBoxTree(mesh, dim)
            tree.build()
            entities = tree.compute_collisions(p)
            self.assertEqual(sorted(entities), reference[dim])

    def test_compute_collisions_3d(self):
        "Test basic creation and point location for unit cube"

        reference = {1: [1364],
                     2: [1967, 1968, 1970, 1972, 1974, 1976],
                     3: [876, 877, 878, 879, 880, 881]}

        p = Point(0.3, 0.3, 0.3)
        mesh = UnitCubeMesh(8, 8, 8)
        for dim in range(1, 4):
            tree = BoundingBoxTree(mesh, dim)
            tree.build()
            entities = tree.compute_collisions(p)
            self.assertEqual(sorted(entities), reference[dim])

    #--- compute_entity_collisions ---

    def test_compute_entity_collisions_1d(self):

        reference = [4]

        p = Point(0.3)
        mesh = UnitIntervalMesh(16)
        tree = BoundingBoxTree(mesh, 1)
        tree.build()
        entities = tree.compute_entity_collisions(p)
        self.assertEqual(sorted(entities), reference)

    def test_compute_entity_collisions_2d(self):

        reference = [136, 137]

        p = Point(0.3, 0.3)
        mesh = UnitSquareMesh(16, 16)
        tree = BoundingBoxTree(mesh, 2)
        tree.build()
        entities = tree.compute_entity_collisions(p)
        self.assertEqual(sorted(entities), reference)

    def test_compute_entity_collisions_3d(self):
        "Test basic creation and point location for unit cube"

        reference = [876, 877, 878, 879, 880, 881]

        p = Point(0.3, 0.3, 0.3)
        mesh = UnitCubeMesh(8, 8, 8)
        tree = BoundingBoxTree(mesh, 3)
        tree.build()
        entities = tree.compute_collisions(p)
        self.assertEqual(sorted(entities), reference)

    #--- compute_first_collision ---

    def test_compute_first_collision_1d(self):

        reference = {1: 4}

        p = Point(0.3)
        mesh = UnitIntervalMesh(16)
        for dim in range(1, 2):
            tree = BoundingBoxTree(mesh, dim)
            tree.build()
            first = tree.compute_first_collision(p)
            self.assertEqual(first, reference[dim])

    def test_compute_collisions_2d(self):

        reference = {1: 226,
                     2: 137}

        p = Point(0.3, 0.3)
        mesh = UnitSquareMesh(16, 16)
        for dim in range(1, 3):
            tree = BoundingBoxTree(mesh, dim)
            tree.build()
            first = tree.compute_first_collision(p)
            self.assertEqual(first, reference[dim])

    def test_compute_collisions_3d(self):
        "Test basic creation and point location for unit cube"

        reference = {1: 1364,
                     2: 1974,
                     3: 879}

        p = Point(0.3, 0.3, 0.3)
        mesh = UnitCubeMesh(8, 8, 8)
        for dim in range(1, 4):
            tree = BoundingBoxTree(mesh, dim)
            tree.build()
            first = tree.compute_first_collision(p)
            self.assertEqual(first, reference[dim])

if __name__ == "__main__":
    print ""
    print "Testing BoundingBoxTree"
    print "------------------------------------------------"
    unittest.main()
