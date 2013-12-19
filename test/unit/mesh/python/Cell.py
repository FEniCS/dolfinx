"""Unit tests for the Cell class"""

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
# First added:  2013-04-18
# Last changed: 2013-08-27

import unittest
import numpy

from dolfin import *

# MPI_COMM_WORLD wrapper
comm = MPICommWrapper()

class IntervalTest(unittest.TestCase):

    def test_collides_point(self):

        if MPI.num_processes(comm.comm()) > 1:
            return

        mesh = UnitIntervalMesh(1)
        cell = Cell(mesh, 0)

        self.assertEqual(cell.collides(Point(0.5)), True)
        self.assertEqual(cell.collides(Point(1.5)), False)

    def test_distance(self):

        if MPI.num_processes(comm.comm()) > 1: return

        mesh = UnitIntervalMesh(1)
        cell = Cell(mesh, 0)

        self.assertAlmostEqual(cell.distance(Point(-1.0)), 1.0)
        self.assertAlmostEqual(cell.distance(Point(0.5)), 0.0)

class TriangleTest(unittest.TestCase):

    def test_collides_point(self):

        if MPI.num_processes(comm.comm()) > 1: return

        mesh = UnitSquareMesh(1, 1)
        cell = Cell(mesh, 0)

        self.assertEqual(cell.collides(Point(0.5)), True)
        self.assertEqual(cell.collides(Point(1.5)), False)

    def test_collides_cell(self):

        if MPI.num_processes(comm.comm()) > 1: return

        m0 = UnitSquareMesh(8, 8)
        c0 = Cell(m0, 0)

        m1 = UnitSquareMesh(8, 8)
        m1.translate(Point(0.1, 0.1))
        c1 = Cell(m1, 0)
        c2 = Cell(m1, 1)

        self.assertEqual(c0.collides(c0), True)
        self.assertEqual(c0.collides(c1), True)
        self.assertEqual(c0.collides(c2), False)
        self.assertEqual(c1.collides(c0), True)
        self.assertEqual(c1.collides(c1), True)
        self.assertEqual(c1.collides(c2), False)
        self.assertEqual(c2.collides(c0), False)
        self.assertEqual(c2.collides(c1), False)
        self.assertEqual(c2.collides(c2), True)

    def test_distance(self):

        if MPI.num_processes(comm.comm()) > 1: return

        mesh = UnitSquareMesh(1, 1)
        cell = Cell(mesh, 1)

        self.assertAlmostEqual(cell.distance(Point(-1.0, -1.0)), numpy.sqrt(2))
        self.assertAlmostEqual(cell.distance(Point(-1.0, 0.5)), 1)
        self.assertAlmostEqual(cell.distance(Point(0.5, 0.5)), 0.0)

class TetrahedronTest(unittest.TestCase):

    def test_collides_point(self):

        if MPI.num_processes(comm.comm()) > 1: return

        mesh = UnitCubeMesh(1, 1, 1)
        cell = Cell(mesh, 0)

        self.assertEqual(cell.collides(Point(0.5)), True)
        self.assertEqual(cell.collides(Point(1.5)), False)

    def test_distance(self):

        if MPI.num_processes(comm.comm()) > 1: return

        mesh = UnitCubeMesh(1, 1, 1)
        cell = Cell(mesh, 5)

        self.assertAlmostEqual(cell.distance(Point(-1.0, -1.0, -1.0)), \
                               numpy.sqrt(3))
        self.assertAlmostEqual(cell.distance(Point(-1.0, 0.5, 0.5)), 1)
        self.assertAlmostEqual(cell.distance(Point(0.5, 0.5, 0.5)), 0.0)

if __name__ == "__main__":
        unittest.main()
