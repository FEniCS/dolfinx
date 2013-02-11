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

mesh = UnitCubeMesh(8, 8, 8)
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

    def test_assign(self):
        from ufl.algorithms import replace
        
        for V0, V1, vector_space in [(V, W, False), (W, V, True)]:
            u = Function(V0)
            u0 = Function(V0)
            u1 = Function(V0)
            u2 = Function(V0)
            u3 = Function(V1)
            
            u.vector()[:] =  1.0
            u0.vector()[:] = 2.0
            u1.vector()[:] = 3.0
            u2.vector()[:] = 4.0
            u3.vector()[:] = 5.0
            
            scalars = {u:1.0, u0:2.0, u1:3.0, u2:4.0, u3:5.0}
            
            uu = Function(V0)
            uu.assign(2*u)
            self.assertEqual(uu.vector().sum(), u0.vector().sum())
            
            uu = Function(V1)
            uu.assign(3*u)
            self.assertEqual(uu.vector().sum(), u1.vector().sum())
            
            # Test complex assignment
            expr = 3*u-4*u1-0.1*4*u*4+u2+3*u0/3./0.5
            expr_scalar = 3-4*3-0.1*4*4+4.+3*2./3./0.5
            uu.assign(expr)
            self.assertAlmostEqual(uu.vector().sum(), \
                                   float(expr_scalar*uu.vector().size()))
            
            # Test expression scaling
            expr = 3*expr
            expr_scalar *= 3
            uu.assign(expr)
            self.assertAlmostEqual(uu.vector().sum(), \
                                   float(expr_scalar*uu.vector().size()))
            
            # Test expression scaling
            expr = expr/4.5
            expr_scalar /= 4.5
            uu.assign(expr)
            self.assertAlmostEqual(uu.vector().sum(), \
                                   float(expr_scalar*uu.vector().size()))
            
            # Test self assignment
            expr = 3*u - Constant(5)*u2 + u1 - 5*u
            expr_scalar = 3 - 5*4. + 3. - 5
            u.assign(expr)
            self.assertAlmostEqual(u.vector().sum(), \
                                   float(expr_scalar*u.vector().size()))

            # Test zero assignment
            u.assign(-u2/2+2*u1-u1/0.5+u2*0.5)
            self.assertAlmostEqual(u.vector().sum(), 0.0)
            
            # Test errounious assignments
            uu = Function(V1)
            f = Expression("1.0")
            self.assertRaises(RuntimeError, lambda: uu.assign(1.0))
            self.assertRaises(RuntimeError, lambda: uu.assign(4*f))
            
            if not vector_space:
                self.assertRaises(RuntimeError, lambda: uu.assign(u*u0))
                self.assertRaises(RuntimeError, lambda: uu.assign(4/u0))
                self.assertRaises(RuntimeError, lambda: uu.assign(4*u*u1))

class ScalarFunctions(unittest.TestCase):
    def test_constant_float_conversion(self):
        c = Constant(3.45)
        self.assertTrue(float(c) == 3.45)

    def test_real_function_float_conversion1(self):
        c = Function(R)
        self.assertTrue(float(c) == 0.0)

    def test_real_function_float_conversion2(self):
        c = Function(R)
        c.assign(Constant(2.34))
        self.assertTrue(float(c) == 2.34)

    def test_real_function_float_conversion3(self):
        c = Function(R)
        c.vector()[:] = 1.23
        self.assertTrue(float(c) == 1.23)

    def test_scalar_conditions(self):
        c = Function(R)
        c.vector()[:] = 1.5

        # Float conversion does not interfere with boolean ufl expressions
        self.assertTrue(isinstance(lt(c, 3), ufl.classes.LT))
        self.assertFalse(isinstance(lt(c, 3), bool))

        # Float conversion is not implicit in boolean Python expressions
        self.assertTrue(isinstance(c < 3, ufl.classes.LT))
        self.assertFalse(isinstance(c < 3, bool))

        # == is used in ufl to compare equivalent representations,
        # <,> result in LT/GT expressions, bool conversion is illegal
        # Note that 1.5 < 0 == False == 1.5 < 1, but that is not what we compare here:
        self.assertFalse((c < 0) == (c < 1))
        # This protects from "if c < 0: ..." misuse:
        self.assertRaises(ufl.UFLException, lambda: bool(c < 0))
        self.assertRaises(ufl.UFLException, lambda: not c < 0)


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
            mesh1 = UnitSquareMesh(3,3)
            V1 = FunctionSpace(mesh1, "CG", 1)

            parameters["allow_extrapolation"] = True
            f1 = Function(V1)
            f1.vector()[:] = 1.0
            self.assertAlmostEqual(f1(0.,-1), 1.0)

            mesh2 = UnitTriangleMesh()
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
