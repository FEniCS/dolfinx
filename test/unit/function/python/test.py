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

     def testArbitraryEval(self):
          class F0(Expression):
               def eval(self, values, x):
                    values[0] = sin(3.0*x[0])*sin(3.0*x[1])*sin(3.0*x[2])

          f0 = F0()
          f1 = Expression("a*sin(3.0*x[0])*sin(3.0*x[1])*sin(3.0*x[2])", \
                          degree=2, a=1.)
          f2, f3 = Expressions("a*sin(3.0*x[0])*sin(3.0*x[1])*sin(3.0*x[2])", \
                    dict(a=1.), "1.0 + 3.0*x[0] + 4.0*x[1] + 0.5*x[2]", degree=2)

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
          f2(0.31, 0.32, 0.33, values = u20)
          u21 = f2(0.31, 0.32, 0.33)
          self.assertAlmostEqual(u20[0], u21)

          # Test Point evaluation
          p0 = Point(0.31, 0.32, 0.33)
          u21 = f2(p0)
          self.assertAlmostEqual(u20[0], u21)

          same_result = sin(3.0*x[0])*sin(3.0*x[1])*sin(3.0*x[2])
          self.assertAlmostEqual(u00[0], same_result)
          self.assertAlmostEqual(u11, same_result)
          self.assertAlmostEqual(u21, same_result)

          # Test ufl evalutation:
          def N(x):
               return x[0]**2 + x[1]

          self.assertEqual(f0((2.0, 1.0), { f0: N }),5)

          a = f0**2
          b = a((2.0, 1.0), { f0: N })
          self.assertEqual(b, 25)

          # Projection requires CGAL
          if not has_cgal():
               return
          
          # FIXME: eval does not work in parallel yet
          if MPI.num_processes() == 1:
              V2 = FunctionSpace(mesh, 'CG', 2)
              g = project(f3, V2)
              u3 = f3(x)
              u4 = g(x)
              self.assertAlmostEqual(u3, u4, places=5)
              self.assertRaises(TypeError, g, [0,0,0,0])
              self.assertRaises(TypeError, g, Point(0,0))

     def testOverLoadAndCallBack(self):
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

     def testWrongEval(self):
          # Test wrong evaluation
          class F0(Expression):
               def eval(self, values, x):
                    values[0] = sin(3.0*x[0])*sin(3.0*x[1])*sin(3.0*x[2])

          f0 = F0()
          f1 = Expression("sin(3.0*x[0])*sin(3.0*x[1])*sin(3.0*x[2])", degree=2)
          f2, f3 = Expressions("sin(3.0*x[0])*sin(3.0*x[1])*sin(3.0*x[2])",
                               "1.0 + 3.0*x[0] + 4.0*x[1] + 0.5*x[2]", degree=2)

          for f in [f0,f1,f2,f3]:
               self.assertRaises(TypeError, f, "s")
               self.assertRaises(TypeError, f, [])
               self.assertRaises(TypeError, f, 0.5, 0.5, 0.5, values = zeros(3,'i'))
               self.assertRaises(TypeError, f, [0.3, 0.2, []])
               self.assertRaises(TypeError, f, 0.3, 0.2, {})
               self.assertRaises(TypeError, f, zeros(3), values = zeros(4))
               self.assertRaises(TypeError, f, zeros(4), values = zeros(3))

     def testNoWriteToConstArray(self):
          class F1(Expression):
               def eval(self, values, x):
                    x[0] = 1.0
                    values[0] = sin(3.0*x[0])*sin(3.0*x[1])*sin(3.0*x[2])

          mesh = UnitCube(3,3,3)
          f1 = F1()
          self.assertRaises(RuntimeError, lambda : assemble(f1*dx, mesh=mesh))

class Instantiation(unittest.TestCase):

     def testWrongSubClassing(self):

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

          def noMemberValues():
               Expression("1", a=1.0)

          def wrongDefaultType():
               Expression("a", a="1")
               
          self.assertRaises(TypeError, noAttributes)
          self.assertRaises(TypeError, noEvalAttribute)
          self.assertRaises(TypeError, wrongEvalAttribute)
          self.assertRaises(TypeError, wrongEvalDataAttribute)
          self.assertRaises(TypeError, wrongArgs)
          self.assertRaises(DeprecationWarning, deprecationWarning)
          self.assertRaises(RuntimeError, noDefaultValues)
          self.assertRaises(RuntimeError, noMemberValues)
          self.assertRaises(TypeError, wrongDefaultType)

     def testElementInstantiation(self):
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
          
     def testExponentInit(self):
          e0 = Expression("1e10")
          self.assertEqual(e0(0,0,0), 1e10)

          e1 = Expression("1e-10")
          self.assertEqual(e1(0,0,0), 1e-10)

          e2 = Expression("1e+10")
          self.assertEqual(e2(0,0,0), 1e+10)


class Interpolate(unittest.TestCase):

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

        # Interpolation not working in parallel yet
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

class Constants(unittest.TestCase):
     
     def testConstantInit(self):
          c0 = Constant(1.)
          c1 = Constant([2,3], interval)
          c2 = Constant([[2,3], [3,4]], triangle)
          c3 = Constant(array([2,3]), tetrahedron)

          self.assertTrue(c0.cell().is_undefined())
          self.assertTrue(c1.cell() == interval)
          self.assertTrue(c2.cell() == triangle)
          self.assertTrue(c3.cell() == tetrahedron)
          
          self.assertTrue(c0.shape() == ())
          self.assertTrue(c1.shape() == (2,))
          self.assertTrue(c2.shape() == (2,2))
          self.assertTrue(c3.shape() == (2,))

     def testGrad(self):
          import ufl
          zero = ufl.constantvalue.Zero((2,3))
          c0 = Constant(1.)
          c3 = Constant(array([2,3]), tetrahedron)
          def gradient(c):
               return grad(c)
          self.assertRaises(UFLException, gradient, c0)
          self.assertEqual(zero, gradient(c3))

if __name__ == "__main__":
    unittest.main()
