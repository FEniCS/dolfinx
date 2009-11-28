"""Unit tests for the function library"""

__author__ = "Anders Logg (logg@simula.no)"
__date__ = "2007-05-24 -- 2009-10-31"
__copyright__ = "Copyright (C) 2007 Anders Logg"
__license__  = "GNU LGPL Version 2.1"

import unittest
from dolfin import *
from math   import sin
from numpy  import array, zeros

mesh = UnitCube(8, 8, 8)
V = FunctionSpace(mesh,'CG',1)

class Eval(unittest.TestCase):

     def testArbitraryEval(self):
          class F0(Expression):
               def eval(self,values,x):
                    values[0] = sin(3.0*x[0])*sin(3.0*x[1])*sin(3.0*x[2])

          f0 = F0(V)
          f1 = Expression("sin(3.0*x[0])*sin(3.0*x[1])*sin(3.0*x[2])", degree=2)
          f2, f3 = Expressions("sin(3.0*x[0])*sin(3.0*x[1])*sin(3.0*x[2])",
                              "1.0 + 3.0*x[0] + 4.0*x[1] + 0.5*x[2]", degree=2)

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

          # Projection requires gts
          if not has_gts():
               return

          V2 = FunctionSpace(mesh, 'CG', 2)
          g = project(f3, V2)
          u3 = f3(x)
          u4 = g(x)
          self.assertAlmostEqual(u3,u4, places=5)

     def testWrongEval(self):
          # Test wrong evaluation
          class F0(Expression):
               def eval(self, values, x):
                    values[0] = sin(3.0*x[0])*sin(3.0*x[1])*sin(3.0*x[2])

          f0 = F0(V)
          f1 = Expression("sin(3.0*x[0])*sin(3.0*x[1])*sin(3.0*x[2])", degree=2)
          f2, f3 = Expressions("sin(3.0*x[0])*sin(3.0*x[1])*sin(3.0*x[2])",
                               "1.0 + 3.0*x[0] + 4.0*x[1] + 0.5*x[2]", degree=2)

          for f in [f0,f1,f2,f3]:
               self.assertRaises(TypeError, f, "s")
               self.assertRaises(TypeError, f, [])
               #self.assertRaises(TypeError, f, zeros(2))
               self.assertRaises(TypeError, f, 0.5, 0.5, 0.5, values = zeros(3,'i'))
               #self.assertRaises(TypeError, f, zeros(4))
               self.assertRaises(ValueError,f, [0.3, 0.2, []])
               self.assertRaises(TypeError, f, 0.3, 0.2, {})
               #self.assertRaises(TypeError, f, 0.3, 0.2)
               #self.assertRaises(TypeError, f, [0.5, 0.2, 0.1, 0.2])
               self.assertRaises(TypeError, f, zeros(3), values = zeros(4))
               self.assertRaises(TypeError, f, zeros(4), values = zeros(3))

class Instantiation(unittest.TestCase):

     def testSameBases(self):

          # FIXME: Hake: What should this unit test do?

          return

          f0 = Expression("2", degree=2)

          class MyConstant(Expression):
               cpparg = "2"

          f1 = MyConstant(V)

          self.assertNotEqual(type(f0).__bases__,type(f1).__bases__)

          #f2, f3 = Expressions("2", V, "3", V)
          f4, f5 = Expressions("2", "3", V = V)
          self.assertEqual(type(f2).__bases__, type(f4).__bases__)
          self.assertEqual(type(f3).__bases__, type(f5).__bases__)

     def testWrongSubClassing(self):

          def noAttributes():
               class NoAttributes(Expression):pass

          def wrongEvalAttribute():
               class WrongEvalAttribute(Expression):
                    def eval(values, x):
                         pass

          def wrongEvalDataAttribute():
               class WrongEvalDataAttribute(Expression):
                    def eval_data(values, data):
                         pass

          def noEvalAttribute():
               class NoEvalAttribute(Expression):
                    def evaluate(self, values, data):
                         pass

          def bothCompileAndPythonSubClassing0():
               class bothCompileAndPythonSubClassing(Expression):
                    def eval(self, values, x):pass
                    cpparg = "2"

          def bothCompileAndPythonSubClassing1():
               class bothCompileAndPythonSubClassing(Expression):
                    def eval_data(self, values, data):pass
                    cpparg = "2"

          def wrongCppargType():
               class WrongCppargType(Expression):
                    cpparg = 2

          self.assertRaises(TypeError,noAttributes)
          self.assertRaises(TypeError,noEvalAttribute)
          self.assertRaises(TypeError,wrongEvalAttribute)
          self.assertRaises(TypeError,wrongEvalDataAttribute)
          self.assertRaises(TypeError,bothCompileAndPythonSubClassing0)
          self.assertRaises(TypeError,bothCompileAndPythonSubClassing1)
          self.assertRaises(TypeError,wrongCppargType)

if __name__ == "__main__":
    unittest.main()
