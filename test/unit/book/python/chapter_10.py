"""
Unit tests for Chapter 10 (DOLFIN: A C++/Python finite element library).
Page numbering starts at 1 and is relative to the chapter (not the book).
"""

# Copyright (C) 2011 Anders Logg
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
# First added:  2011-10-20
# Last changed: 2011-10-20

import unittest
from dolfin import *

class TestPage4(unittest.TestCase):

    def test_box_1(self):
        x = Vector()

    def test_box_2(self):
        A = Matrix()

class TestPage5(unittest.TestCase):

    def create_data(self):
        mesh = UnitSquare(2, 2)
        V = FunctionSpace(mesh, "Lagrange", 1)
        u = TrialFunction(V)
        v = TestFunction(V)
        A = assemble(u*v*dx)
        b = assemble(v*dx)
        x = Vector()
        return A, x, b

    def test_box_1(self):
        x = Vector(100)

    def test_box_2(self):
        A, x, b = self.create_data()
        solve(A, x, b)

    #def test_box_3(self):
    #    A, x, b = self.create_data()
    #    solver = LUSolver(A)
    #    solver.solve(x, b)

if __name__ == "__main__":
    print ""
    print "Testing the FEniCS Book, Chapter 10"
    print "-----------------------------------"
    unittest.main()
