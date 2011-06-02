"""Unit tests for the Function class"""

# Copyright (C) 2011 Garth N. Wells
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
# First added:  2011-03-23
# Last changed: 2011-03-23

import unittest
from dolfin import *

mesh = UnitCube(8, 8, 8)
V = FunctionSpace(mesh, 'CG', 1)
W = VectorFunctionSpace(mesh, 'CG', 1)

class Interpolate(unittest.TestCase):

    def test_interpolation_mismatch_rank0(self):
        f = Expression("1.0")
        self.assertRaises(RuntimeError, interpolate, f, W)

    def test_interpolation_mismatch_rank1(self):
        f = Expression(("1.0", "1.0"))
        self.assertRaises(RuntimeError, interpolate, f, W)

    def test_interpolation_jit_rank0(self):
        f = Expression("1.0")
        w = interpolate(f, V)
        x = w.vector()
        self.assertEqual(x.max(), 1)
        self.assertEqual(x.min(), 1)

    def test_interpolation_jit_rank1(self):
        f = Expression(("1.0", "1.0", "1.0"))
        w = interpolate(f, W)
        x = w.vector()
        self.assertEqual(x.max(), 1)
        self.assertEqual(x.min(), 1)

    def testInterpolation(self):
        class F0(Expression):
            def eval(self, values, x):
                values[0] = 1.0
        class F1(Expression):
            def eval(self, values, x):
                values[0] = 1.0
                values[1] = 1.0
            def value_shape(self):
                return (2,)

        # Interpolation not working in parallel yet (need number of global vertices in tests)
        if MPI.num_processes() == 1:
            # Scalar interpolation
            f0 = F0()
            f = Function(V)
            f.interpolate(f0)
            self.assertAlmostEqual(f.vector().norm("l1"), mesh.num_vertices())

            # Vector interpolation
            f1 = F1()
            W = V * V
            f = Function(W)
            f.interpolate(f1)
            self.assertAlmostEqual(f.vector().norm("l1"), 2*mesh.num_vertices())

if __name__ == "__main__":
    unittest.main()
