"""Unit tests for the linear algebra interface"""

__author__ = "Johan Hake (hake@simula.no)"
__date__ = "2008-09-30 -- 2008-10-06"
__copyright__ = "Copyright (C) 2008 Johan Hake"
__license__  = "GNU LGPL Version 2.1"

import unittest
from dolfin import *

class AbstractBaseTest(object):
    count = 0
    def setUp(self):
        type(self).count += 1
        if type(self).count == 1:
            # Only print this message once per class instance
            print "\nRunning:",type(self).__name__

    def assemble_matrix(self):
        " Assemble a simple matrix"
        mesh = UnitSquare(3,3)

        V = FunctionSpace(mesh,"Lagrange", 1)
        
        u = TrialFunction(V)
        v = TestFunction(V)
        
        A = self.MatrixType()
        return assemble(dot(grad(u),grad(v))*dx,tensor=A)
    
    def get_vector(self):
        " Assemble a simple matrix"
        from numpy import random, linspace
        vec = self.VectorType(16)
        vec.set(random.rand(vec.size()))
        return vec

    def test_matrix(self):
        from numpy import ndarray
        org = self.assemble_matrix()
        A = org.copy()
        B = org.copy()
        self.assertAlmostEqual(A(5,5),B(5,5))
        
        B *= 0.5
        A *= 2
        self.assertAlmostEqual(A(5,5),4*B(5,5))
        
        B /= 2
        A /= 0.5
        self.assertAlmostEqual(A(5,5),16*B(5,5))
        
        A += B
        self.assertAlmostEqual(A(5,5),17)
        
        A -= B
        self.assertAlmostEqual(A(5,5),16)
        
        C = 16*B
        self.assertAlmostEqual(A(5,5),C(5,5))
        
        D = (C+B)*5
        self.assertAlmostEqual(D(5,5),85)
        
        F = (A-B)/4
        self.assertAlmostEqual(F(5,5),3.75)
        
        A.axpy(100,B)
        self.assertAlmostEqual(A(5,5),116)
        
        A2 = A.array()
        self.assertTrue(isinstance(A2,ndarray))
        self.assertEqual(A2.shape,(16,16))
        self.assertAlmostEqual(A2[5,5],A(5,5))

    def test_vector(self):
        from numpy import ndarray, linspace
        org = self.get_vector()
        
        A = org.copy()
        B = org.copy()
        self.assertAlmostEqual(A[5],B[5])
        
        B *= 0.5
        A *= 2
        self.assertAlmostEqual(A[5],4*B[5])
        
        B /= 2
        A /= 0.5
        self.assertAlmostEqual(A[5],16*B[5])
        
        val1 = A[5]
        val2 = B[5]
        A += B
        self.assertAlmostEqual(A[5],val1+val2)
        
        A -= B
        self.assertAlmostEqual(A[5],val1)
        
        C = 16*B
        self.assertAlmostEqual(A[5],C[5])
        
        D = (C+B)*5
        self.assertAlmostEqual(D[5],(val1+val2)*5)
        
        F = (A-B)/4
        self.assertAlmostEqual(F[5],(val1-val2)/4)
        
        A.axpy(100,B)
        self.assertAlmostEqual(A[5],val1+val2*100)
        
        A2 = A.array()
        self.assertTrue(isinstance(A2,ndarray))
        self.assertEqual(A2.shape,(16,))
        self.assertAlmostEqual(A2[5],A[5])

    def test_matrix_vector(self):
        from numpy import dot, absolute
        v = self.get_vector()
        A = self.assemble_matrix()

        u = A*v
        self.assertTrue(isinstance(u,type(v)))
        self.assertEqual(u.size(),v.size())

        u2 = 2*u - A*v
        self.assertAlmostEqual(u2[4],u[4])
        
        u3 = 2*u + -1.0*(A*v)
        self.assertAlmostEqual(u3[4],u[4])
        
        v_numpy = v.array()
        A_numpy = A.array()
        
        u_numpy = dot(A_numpy,v_numpy)
        u_numpy2 = A*v_numpy

        self.assertTrue(absolute(u.array()-u_numpy).sum() < DOLFIN_EPS*v.size())
        self.assertTrue(absolute(u_numpy2-u_numpy).sum() < DOLFIN_EPS*v.size())

class MatrixTester(AbstractBaseTest,unittest.TestCase):
    MatrixType = Matrix
    VectorType = Vector

class uBLASSparseTester(AbstractBaseTest,unittest.TestCase):
    MatrixType = uBLASSparseMatrix
    VectorType = uBLASVector

class uBLASDenseTester(AbstractBaseTest,unittest.TestCase):
    MatrixType = uBLASDenseMatrix
    VectorType = uBLASVector

if hasattr(cpp,"PETScMatrix"):
    class PETScTester(AbstractBaseTest,unittest.TestCase):
        MatrixType = PETScMatrix
        VectorType = PETScVector

if hasattr(cpp,"EpetraMatrix"):
    class EpetraTester(AbstractBaseTest,unittest.TestCase):
        MatrixType = EpetraMatrix
        VectorType = EpetraVector

if hasattr(cpp,"MTL4Matrix"):
    class MTL4Tester(AbstractBaseTest,unittest.TestCase):
        MatrixType = MTL4Matrix
        VectorType = MTL4Vector

if __name__ == "__main__":
    print ""
    print "Testing basic PyDOLFIN linear algebra operations"
    print "------------------------------------------------"
    unittest.main()
