"""Unit tests for the function library"""

# Copyright (C) 2007 Anders Logg
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
# First added:  2007-05-24
# Last changed: 2011-01-28

import unittest
from dolfin import *
from math   import sin
from numpy  import array, zeros

mesh = UnitCube(8, 8, 8)
V = FunctionSpace(mesh, 'CG', 1)
W = VectorFunctionSpace(mesh, 'CG', 1)

class Eval(unittest.TestCase):

     def test_arbitraryEval(self):
          class F0(Expression):
               def eval(self, values, x):
                    values[0] = sin(3.0*x[0])*sin(3.0*x[1])*sin(3.0*x[2])

          f0 = F0()
          f1 = Expression("a*sin(3.0*x[0])*sin(3.0*x[1])*sin(3.0*x[2])", \
                          degree=2, a=1.)
          x = array([0.31, 0.32, 0.33])
          u00 = zeros(1); u01 = zeros(1)
          u10 = zeros(1); u20 = zeros(1)

          # Test original and vs short evaluation
          f0.eval(u00, x)
          f0(x, values = u01)
          self.assertAlmostEqual(u00[0], u01[0])

          # Evaluation with and without return value
          f1(x, values = u10);
          u11 = f1(x);
          self.assertAlmostEqual(u10[0], u11)

          # Test *args for coordinate argument
          f1(0.31, 0.32, 0.33, values = u20)
          u21 = f0(0.31, 0.32, 0.33)
          self.assertAlmostEqual(u20[0], u21)

          # Test Point evaluation
          p0 = Point(0.31, 0.32, 0.33)
          u21 = f1(p0)
          self.assertAlmostEqual(u20[0], u21)

          same_result = sin(3.0*x[0])*sin(3.0*x[1])*sin(3.0*x[2])
          self.assertAlmostEqual(u00[0], same_result)
          self.assertAlmostEqual(u11, same_result)
          self.assertAlmostEqual(u21, same_result)

          # Projection requires CGAL
          if not has_cgal():
               return

          # FIXME: eval does not work in parallel yet
          if MPI.num_processes() == 1:
               f2 = Expression("1.0 + 3.0*x[0] + 4.0*x[1] + 0.5*x[2]", degree=2)
               V2 = FunctionSpace(mesh, 'CG', 2)
               g = project(f2, V=V2)
               u3 = f2(x)
               u4 = g(x)
               self.assertAlmostEqual(u3, u4, places=5)
               self.assertRaises(TypeError, g, [0,0,0,0])
               self.assertRaises(TypeError, g, Point(0,0))

     def test_ufl_eval(self):
          class F0(Expression):
               def eval(self, values, x):
                    values[0] = sin(3.0*x[0])*sin(3.0*x[1])*sin(3.0*x[2])

          class V0(Expression):
               def eval(self, values, x):
                    values[0] = x[0]**2
                    values[1] = x[1]**2
                    values[2] = x[2]**2
               def value_shape(self):
                    return (3,)

          f0 = F0()
          v0 = V0()

          x = (2.0, 1.0, 3.0)

          # Test ufl evaluation through mapping (overriding the Expression with N here):
          def N(x):
               return x[0]**2 + x[1] + 3*x[2]

          self.assertEqual(f0(x, { f0: N }), 14)

          a = f0**2
          b = a(x, { f0: N })
          self.assertEqual(b, 196)

          # Test ufl evaluation together with Expression evaluation by dolfin
          # scalar
          self.assertEqual(f0(x), f0(*x))
          self.assertEqual((f0**2)(x), f0(*x)**2)
          # vector
          self.assertTrue(all(a == b for a,b in zip(v0(x), v0(*x))))
          self.assertEqual(dot(v0,v0)(x), sum(v**2 for v in v0(*x)))
          self.assertEqual(dot(v0,v0)(x), 98)

     def test_overload_and_call_back(self):
          class F0(Expression):
               def eval(self, values, x):
                    values[0] = sin(3.0*x[0])*sin(3.0*x[1])*sin(3.0*x[2])

          class F1(Expression):
               def __init__(self, mesh, *arg, **kwargs):
                    self.mesh = mesh
               def eval_cell(self, values, x, cell):
                    c = Cell(self.mesh, cell.index)
                    values[0] = sin(3.0*x[0])*sin(3.0*x[1])*sin(3.0*x[2])

          e0 = F0(degree=2)
          e1 = F1(mesh, degree=2)
          e2 = Expression("sin(3.0*x[0])*sin(3.0*x[1])*sin(3.0*x[2])", degree=2)

          s0 = norm(interpolate(e0, V))
          s1 = norm(interpolate(e1, V))
          s2 = norm(interpolate(e2, V))

          ref = 0.36557637568519191
          self.assertAlmostEqual(s0, ref)
          self.assertAlmostEqual(s1, ref)
          self.assertAlmostEqual(s2, ref)

     def test_wrong_eval(self):
          # Test wrong evaluation
          class F0(Expression):
               def eval(self, values, x):
                    values[0] = sin(3.0*x[0])*sin(3.0*x[1])*sin(3.0*x[2])

          f0 = F0()
          f1 = Expression("sin(3.0*x[0])*sin(3.0*x[1])*sin(3.0*x[2])", degree=2)

          for f in [f0, f1]:
               self.assertRaises(TypeError, f, "s")
               self.assertRaises(TypeError, f, [])
               self.assertRaises(TypeError, f, 0.5, 0.5, 0.5, values = zeros(3,'i'))
               self.assertRaises(TypeError, f, [0.3, 0.2, []])
               self.assertRaises(TypeError, f, 0.3, 0.2, {})
               self.assertRaises(TypeError, f, zeros(3), values = zeros(4))
               self.assertRaises(TypeError, f, zeros(4), values = zeros(3))

     def test_no_write_to_const_array(self):
          class F1(Expression):
               def eval(self, values, x):
                    x[0] = 1.0
                    values[0] = sin(3.0*x[0])*sin(3.0*x[1])*sin(3.0*x[2])

          mesh = UnitCube(3,3,3)
          f1 = F1()
          self.assertRaises(RuntimeError, lambda : assemble(f1*dx, mesh=mesh))

class MeshEvaluation(unittest.TestCase):

     def test_compute_vertex_values(self):
          from numpy import zeros, all, array

          e0 = Expression("1")
          e1 = Expression(("1", "2", "3"))

          e0_values = e0.compute_vertex_values(mesh)
          e1_values = e1.compute_vertex_values(mesh)

          self.assertTrue(all(e0_values==1))
          self.assertTrue(all(e1_values[:mesh.num_vertices()]==1))
          self.assertTrue(all(e1_values[mesh.num_vertices():mesh.num_vertices()*2]==2))
          self.assertTrue(all(e1_values[mesh.num_vertices()*2:mesh.num_vertices()*3]==3))

class Instantiation(unittest.TestCase):

     def test_wrong_sub_classing(self):

          def noAttributes():
               class NoAttributes(Expression):pass

          def wrongEvalAttribute():
               class WrongEvalAttribute(Expression):
                    def eval(values, x):
                         pass

          def wrongEvalDataAttribute():
               class WrongEvalDataAttribute(Expression):
                    def eval_cell(values, data):
                         pass

          def noEvalAttribute():
               class NoEvalAttribute(Expression):
                    def evaluate(self, values, data):
                         pass

          def wrongArgs():
               class WrongArgs(Expression):
                    def eval(self, values, x):
                         pass
               e = WrongArgs(V)

          def deprecationWarning():
               class Deprecated(Expression):
                    def eval(self, values, x):
                         pass
                    def dim(self):
                         return 2
               e = Deprecated()

          def noDefaultValues():
               Expression("a")

          def wrongDefaultType():
               Expression("a", a="1")

          self.assertRaises(TypeError, noAttributes)
          self.assertRaises(TypeError, noEvalAttribute)
          self.assertRaises(TypeError, wrongEvalAttribute)
          self.assertRaises(TypeError, wrongEvalDataAttribute)
          self.assertRaises(TypeError, wrongArgs)
          self.assertRaises(DeprecationWarning, deprecationWarning)
          self.assertRaises(RuntimeError, noDefaultValues)
          self.assertRaises(TypeError, wrongDefaultType)

     def test_element_instantiation(self):
          class F0(Expression):
               def eval(self, values, x):
                    values[0] = 1.0
          class F1(Expression):
               def eval(self, values, x):
                    values[0] = 1.0
                    values[1] = 1.0
               def value_shape(self):
                    return (2,)

          class F2(Expression):
               def eval(self, values, x):
                    values[0] = 1.0
                    values[1] = 1.0
                    values[2] = 1.0
                    values[3] = 1.0
               def value_shape(self):
                    return (2,2)

          e0 = Expression("1")
          self.assertTrue(e0.ufl_element().cell().is_undefined())

          e1 = Expression("1", cell=triangle)
          self.assertFalse(e1.ufl_element().cell().is_undefined())

          e2 = Expression("1", cell=triangle, degree=2)
          self.assertEqual(e2.ufl_element().degree(), 2)

          e3 = Expression(["1", "1"], cell=triangle)
          self.assertTrue(isinstance(e3.ufl_element(), VectorElement))

          e4 = Expression((("1", "1"), ("1", "1")), cell=triangle)
          self.assertTrue(isinstance(e4.ufl_element(), TensorElement))

          f0 = F0()
          self.assertTrue(f0.ufl_element().cell().is_undefined())

          f1 = F0(cell=triangle)
          self.assertFalse(f1.ufl_element().cell().is_undefined())

          f2 = F0(cell=triangle, degree=2)
          self.assertEqual(f2.ufl_element().degree(), 2)

          f3 = F1(cell=triangle)
          self.assertTrue(isinstance(f3.ufl_element(), VectorElement))

          f4 = F2(cell=triangle)
          self.assertTrue(isinstance(f4.ufl_element(), TensorElement))

     def test_exponent_init(self):
          e0 = Expression("1e10")
          self.assertEqual(e0(0,0,0), 1e10)

          e1 = Expression("1e-10")
          self.assertEqual(e1(0,0,0), 1e-10)

          e2 = Expression("1e+10")
          self.assertEqual(e2(0,0,0), 1e+10)

     def test_name_space_usage(self):
          e0 = Expression("std::sin(x[0])*cos(x[1])")
          e1 = Expression("sin(x[0])*std::cos(x[1])")
          self.assertAlmostEqual(assemble(e0*dx, mesh=mesh), \
                                 assemble(e1*dx, mesh=mesh))


if __name__ == "__main__":
    unittest.main()
