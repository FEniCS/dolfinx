"""Unit tests for the linear algebra interface"""

__author__ = "Johan Hake (hake@simula.no)"
__date__ = "2008-09-30"
__copyright__ = "Copyright (C) 2008 Johan Hake"
__license__  = "GNU LGPL Version 2.1"

import unittest
import dolfin
from dolfin import *

class AbstractBaseTest(object):

    def assemble_matrix(self):
        " Assemble a simple matrix"
        element = FiniteElement("Lagrange", "triangle", 1)
        
        u = TrialFunction(element)
        v = TestFunction(element)
        
        mesh = UnitSquare(3,3)
        A = self.MatrixType()
        return assemble(dot(grad(u),grad(v))*dx,mesh,tensor=A)
    
    def assemble_vector(self):
        " Assemble a simple matrix"
        element = FiniteElement("Lagrange", "triangle", 1)
        v = TestFunction(element)
        f = Function(element)
        mesh = UnitSquare(3,3)
        vec = self.VectorType()
        return assemble(v*f*dx,mesh,["1"],tensor=vec)

    def test_matrix(self):
        from numpy import ndarray
        org = self.assemble_matrix()
        A = org.copy()
        B = org.copy()
        self.assertEqual(A(5,5),B(5,5))
        
        B *= 0.5
        A *= 2
        self.assertEqual(A(5,5),4*B(5,5))
        
        B /= 2
        A /= 0.5
        self.assertEqual(A(5,5),16*B(5,5))
        
        A += B
        self.assertEqual(A(5,5),17)
        
        A -= B
        self.assertEqual(A(5,5),16)
        
        C = 16*B
        self.assertEqual(A(5,5),C(5,5))
        
        D = (C+B)*5
        self.assertEqual(D(5,5),85)
        
        F = (A-B)/4
        self.assertEqual(F(5,5),3.75)
        
        A.axpy(100,B)
        self.assertEqual(A(5,5),116)
        
        A2 = A.array()
        self.assertTrue(isinstance(A2,ndarray))
        self.assertEqual(A2.shape,(16,16))
        self.assertEqual(A2[5,5],A(5,5))

    def test_vector(self):
        from numpy import ndarray
        org = self.assemble_vector()
        
        A = org.copy()
        B = org.copy()
        self.assertEqual(A[5],B[5])
        
        B *= 0.5
        A *= 2
        self.assertEqual(A[5],4*B[5])
        
        B /= 2
        A /= 0.5
        self.assertEqual(A[5],16*B[5])
        
        val1 = 0.44444444444444375
        val2 = 0.027777777777777735
        A += B
        self.assertEqual(A[5],val1+val2)
        
        A -= B
        self.assertEqual(A[5],val1)
        
        C = 16*B
        self.assertEqual(A[5],C[5])
        
        D = (C+B)*5
        self.assertEqual(D[5],(val1+val2)*5)
        
        F = (A-B)/4
        self.assertEqual(F[5],(val1-val2)/4)
        
        A.axpy(100,B)
        self.assertEqual(A[5],val1+val2*100)
        
        A2 = A.array()
        self.assertTrue(isinstance(A2,ndarray))
        self.assertEqual(A2.shape,(16,))
        self.assertEqual(A2[5],A[5])

    def test_matrix_vector(self):
        from numpy import dot, absolute
        v = self.assemble_vector()
        A = self.assemble_matrix()
        
        u = A*v
        self.assertTrue(isinstance(u,type(v)))
        self.assertEqual(u.size(),v.size())

        v2 = v.array()
        A2 = A.array()
        u2 = dot(A2,v2)
        self.assertEqual(absolute(u.array()).sum(),absolute(u2).sum())

class MatrixTester(AbstractBaseTest,unittest.TestCase):
    MatrixType = Matrix
    VectorType = Vector

class uBLAS1Tester(AbstractBaseTest,unittest.TestCase):
    MatrixType = uBLASSparseMatrix
    VectorType = uBLASVector

class uBLAS2Tester(AbstractBaseTest,unittest.TestCase):
    MatrixType = uBLASDenseMatrix
    VectorType = uBLASVector

if hasattr(dolfin,"PETScMatrix"):
    class PETScTester(AbstractBaseTest,unittest.TestCase):
        MatrixType = PETScMatrix
        VectorType = PETScVector
    
if hasattr(dolfin,"EpetraMatrix"):
    class EpetraTester(AbstractBaseTest,unittest.TestCase):
        MatrixType = EpetraMatrix
        VectorType = EpetraVector

if hasattr(dolfin,"MTL4Matrix"):
    class MTL4Tester(AbstractBaseTest,unittest.TestCase):
        MatrixType = MTL4Matrix
        VectorType = MTL4Vector

if __name__ == "__main__":
    unittest.main()
