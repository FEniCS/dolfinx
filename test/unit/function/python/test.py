"""Unit tests for the function library"""

__author__ = "Anders Logg (logg@simula.no)"
__date__ = "2007-05-24 -- 2007-05-24"
__copyright__ = "Copyright (C) 2007 Anders Logg"
__license__  = "GNU LGPL Version 2.1"

import unittest
from dolfin import *
from math   import sin
from numpy  import array, zeros

mesh = UnitCube(8, 8, 8)
V = FunctionSpace(mesh,'CG',1)

try:
     IntersectionDetector(mesh)
     HAS_GTS = True
except:
     HAS_GTS = False
     
class Eval(unittest.TestCase):

     def testArbitraryEval(self):
          class F0(Function):
               def eval(self,values,x):
                    values[0] = sin(3.0*x[0])*sin(3.0*x[1])*sin(3.0*x[2])
          
          f0 = F0(V)
          f1 = Function(V,"sin(3.0*x[0])*sin(3.0*x[1])*sin(3.0*x[2])")
          f2,f3 = Functions("sin(3.0*x[0])*sin(3.0*x[1])*sin(3.0*x[2])",
                            "1.0 + 3.0*x[0] + 4.0*x[1] + 0.5*x[2]",V=V)
          
          x = array([0.31, 0.32, 0.33])
          u0 = zeros(1);u1 = zeros(1);u2 = zeros(1)
          f0.eval(u0,x);f1.eval(u1,x);f2.eval(u2,x)

          same_result = sin(3.0*x[0])*sin(3.0*x[1])*sin(3.0*x[2])
          self.assertAlmostEqual(u0[0],same_result)
          self.assertAlmostEqual(u1[0],same_result)
          self.assertAlmostEqual(u2[0],same_result)
          
          if not HAS_GTS:
               return
          
          V2 = FunctionSpace(mesh,'CG',2)
          g = project(f3,V2)
          f3.eval(u0,x)
          g.eval(u1,x)
          self.assertAlmostEqual(u0[0],u1[0])

class Instantiation(unittest.TestCase):
     def testSameBases(self):
          c = Constant(mesh,2)
          f0 = Function(V,"2")
          class MyConstant(Function):
               cpparg = "2"
          f1 = MyConstant(V)
          self.assertEqual(type(f0).__bases__,type(c).__bases__)
          self.assertEqual(type(f0).__bases__,type(f1).__bases__)

          f2, f3 = Functions(V,"2",V,"3")
          f4, f5 = Functions("2","3",V=V)
          self.assertEqual(type(f2).__bases__,type(f4).__bases__)
          self.assertEqual(type(f3).__bases__,type(f5).__bases__)
          

     def testWrongSubClassing(self):

          def noAttributes():
               class NoAttributes(Function):pass

          def wrongEvalAttribute():
               class WrongEvalAttribute(Function):
                    def eval(values,x):
                         pass
          
          def wrongEvalDataAttribute():
               class WrongEvalDataAttribute(Function):
                    def eval(values,data):
                         pass
          
          def noEvalAttribute():
               class NoEvalAttribute(Function):
                    def evaluate(self,values,data):
                         pass
          
          def bothCompileAndPythonSubClassing0():
               class bothCompileAndPythonSubClassing(Function):
                    def eval(self,values,x):pass
                    cpparg = "2"
          
          def bothCompileAndPythonSubClassing1():
               class bothCompileAndPythonSubClassing(Function):
                    def eval_data(self,values,data):pass
                    cpparg = "2"
          
          def wrongCppargType():
               class WrongCppargType(Function):
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
