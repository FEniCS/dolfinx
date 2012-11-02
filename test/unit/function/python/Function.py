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
import ufl

mesh = UnitCube(8, 8, 8)
R = FunctionSpace(mesh, 'R', 0)
V = FunctionSpace(mesh, 'CG', 1)
W = VectorFunctionSpace(mesh, 'CG', 1)

class Interface(unittest.TestCase):
    def test_in_function_space(self):
        u = Function(W)
        v = Function(W)
        self.assertTrue(u in W)
        self.assertTrue(u in u.function_space())
        self.assertTrue(u in v.function_space())
        for i, usub in enumerate(u.split()):
            self.assertTrue(usub in W.sub(i))

    def test_compute_vertex_values(self):
        from numpy import zeros, all, array
        u = Function(V)
        v = Function(W)

        u.vector()[:] = 1.
        v.vector()[:] = 1.

        u_values = u.compute_vertex_values(mesh)
        v_values = v.compute_vertex_values(mesh)

        self.assertTrue(all(u_values==1))

    def xtest_float_conversion(self):
        c = Function(R)
        self.assertTrue(float(c) == 0.0)

        c.vector()[:] = 1.23
        self.assertTrue(float(c) == 1.23)

        c.assign(Constant(2.34))
        self.assertTrue(float(c) == 2.34)

        c = Constant(3.45)
        self.assertTrue(float(c) == 3.45)

    def test_scalar_conditions(self):
        c = Function(R)
        c.vector()[:] = 1.5

        # Float conversion does not interfere with boolean ufl expressions
        self.assertTrue(isinstance(lt(c, 3), ufl.classes.LT))
        self.assertFalse(isinstance(lt(c, 3), bool))

        # Float conversion is not implicit in boolean python expressions
        self.assertFalse(isinstance(c < 3, ufl.classes.LT))
        self.assertTrue(isinstance(c < 3, bool))

        # This looks bad, but there is a very good reason for this behaviour :)
        # Put shortly, the reason is that boolean operators are used for comparison
        # and ordering of ufl expressions in python data structures.
        self.assertTrue((c < 2) == (c < 1))
        self.assertTrue((c > 2) == (c > 1))
        self.assertTrue((c < 2) == (not c > 1))


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

    def test_extrapolation(self):
        f0 = Function(V)
        self.assertRaises(RuntimeError, f0.__call__, (0., 0, -1))

        if MPI.num_processes() == 1:
            mesh1 = UnitSquare(3,3)
            V1 = FunctionSpace(mesh1, "CG", 1)

            parameters["allow_extrapolation"] = True
            f1 = Function(V1)
            f1.vector()[:] = 1.0
            self.assertAlmostEqual(f1(0.,-1), 1.0)

            mesh2 = UnitTriangle()
            V2 = FunctionSpace(mesh2, "CG", 1)

            parameters["allow_extrapolation"] = False
            f2 = Function(V2)
            self.assertRaises(RuntimeError, f2.__call__, (0.,-1.))

            parameters["allow_extrapolation"] = True
            f3 = Function(V2)
            f3.vector()[:] = 1.0

            self.assertAlmostEqual(f3(0.,-1), 1.0)

    def test_interpolation_jit_rank1(self):
        f = Expression(("1.0", "1.0", "1.0"))
        w = interpolate(f, W)
        x = w.vector()
        self.assertEqual(x.max(), 1)
        self.assertEqual(x.min(), 1)

    def test_interpolation_old(self):
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
