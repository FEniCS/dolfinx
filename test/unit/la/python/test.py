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
        if self.backend != "default":
            dolfin_set("linear algebra backend",self.backend)
        type(self).count += 1
        if type(self).count == 1:
            # Only print this message once per class instance
            print "\nRunning:",type(self).__name__
        

    def assemble_matrix(self, use_backend=False):
        " Assemble a simple matrix"
        mesh = UnitSquare(3,3)

        V = FunctionSpace(mesh,"Lagrange", 1)
        
        u = TrialFunction(V)
        v = TestFunction(V)

        if use_backend:
            if self.backend == "uBLAS":
                backend = globals()[self.backend+self.sub_backend+'Factory_instance']()
            else:
                backend = globals()[self.backend+'Factory_instance']()
            return assemble(dot(grad(u),grad(v))*dx, backend=backend)
        else:
            return assemble(dot(grad(u),grad(v))*dx)
    
    def get_Vector(self):
        from numpy import random, linspace
        vec = Vector(16)
        vec.set(random.rand(vec.size()))
        return vec

    def run_matrix_test(self,use_backend):
        from numpy import ndarray
        org = self.assemble_matrix(use_backend)
        A = org.copy()
        B = org.copy()

        def wrong_getitem(type):
            if type == 0:
                A["0,1"]
            elif type == 1:
                A[0]
            elif type == 2:
                A[0,0,0]
            elif type == 3:
                A("0,1")
            elif type == 4:
                A(0)
            else:
                A(0,0,0)
        
        self.assertRaises(TypeError,wrong_getitem,0)
        self.assertRaises(TypeError,wrong_getitem,1)
        self.assertRaises(TypeError,wrong_getitem,2)
        self.assertRaises(TypeError,wrong_getitem,3)
        self.assertRaises(TypeError,wrong_getitem,4)
        self.assertRaises(TypeError,wrong_getitem,5)
        
        self.assertAlmostEqual(A[5,5],A(5,5))
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
        
    def test_matrix_with_backend(self):
        self.run_matrix_test(True)

    def test_matrix_without_backend(self):
        self.run_matrix_test(False)

    def test_vector(self):
        from numpy import ndarray, linspace
        org = self.get_Vector()

        A = org.copy()
        B = down_cast(org.copy())
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
        self.assertAlmostEqual(A2.sum(),A.sum())

        B2 = B.array()
        A[1:16:2] = B[1:16:2]
        A2[1:16:2] = B2[1:16:2]
        self.assertAlmostEqual(A2[1],A[1])
        
        ind = [1,3,6,9,15]
        G  = A[ind]
        G2 = A2[ind]
        self.assertAlmostEqual(G2.sum(),G.sum())
        self.assertAlmostEqual(A[-1],A[15])
        self.assertAlmostEqual(A[-16],A[0])
        
        C[:] = 2
        D.assign(2)
        self.assertAlmostEqual(C[0],2)
        self.assertAlmostEqual(C[-1],2)
        self.assertAlmostEqual(C.sum(),D.sum())
        
        C[ind] = 3
        self.assertAlmostEqual(C[ind].sum(),3*len(ind))
        
        def wrong_index(ind):
            A[ind]
        
        self.assertRaises(RuntimeError,wrong_index,(-17))
        self.assertRaises(RuntimeError,wrong_index,(16))
        self.assertRaises(RuntimeError,wrong_index,("jada"))
        self.assertRaises(RuntimeError,wrong_index,([-17,2]))
        self.assertRaises(RuntimeError,wrong_index,([16,2]))
        
        def wrong_dim(ind0,ind1):
            A[ind0] = B[ind1]
        
        self.assertRaises(RuntimeError,wrong_dim,[0,2],[0,2,4])
        self.assertRaises(RuntimeError,wrong_dim,[0,2],slice(0,4,1))
        self.assertRaises(RuntimeError,wrong_dim,0,slice(0,4,1))

    def test_matrix_vector(self):
        from numpy import dot, absolute
        v = self.get_Vector()
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

# A DataTester class that test the acces of the raw data through pointers
# This is only available for uBLAS and MTL4 backends
class DataTester(object):
    def test_matrix_data(self):
        # Test for ordinary Matrix
        A = self.assemble_matrix()
        array = A.array()
        rows, cols, values = A.data()
        i = 0
        for row in xrange(A.size(0)):
            for col in xrange(rows[row],rows[row+1]):
                self.assertEqual(array[row,cols[col]],values[i])
                i += 1

        # Test for down_casted Matrix
        A = down_cast(A)
        rows, cols, values = A.data()
        for row in xrange(A.size(0)):
            for k in xrange(rows[row],rows[row+1]):
                self.assertEqual(array[row,cols[k]],values[k])

    def test_vector_data(self):
        # Test for ordinary Vector
        v = self.get_Vector()
        array = v.array()
        data = v.data()
        self.assertTrue((data==array).all())

        # Test for down_casted Vector
        v = down_cast(v)
        data = v.data()
        self.assertTrue((data==array).all())

class DataNotWorkingTester(object):
    def test_matrix_data(self):
        A = self.assemble_matrix()
        self.assertRaises(RuntimeError,A.data)

        A = down_cast(A)
        self.assertRaises(RuntimeError,A.data)

    def test_vector_data(self):
        v = self.get_Vector()
        self.assertRaises(RuntimeError,v.data)

        v = down_cast(v)
        def no_attribute():
            v.data()
        self.assertRaises(AttributeError,no_attribute)
        

class uBLASSparseTester(AbstractBaseTest,DataTester,unittest.TestCase):
    backend     = "uBLAS"
    sub_backend = "Sparse"

class uBLASDenseTester(AbstractBaseTest,DataTester,unittest.TestCase):
    backend     = "uBLAS"
    sub_backend = "Dense"

if has_linear_algebra_backend("PETSc"):
    class PETScTester(AbstractBaseTest,DataNotWorkingTester,unittest.TestCase):
        backend    = "PETSc"

if has_linear_algebra_backend("Epetra"):
    class EpetraTester(AbstractBaseTest,DataNotWorkingTester,unittest.TestCase):
        backend    = "Epetra"

if has_linear_algebra_backend("MTL4"):
    class MTL4Tester(AbstractBaseTest,DataTester,unittest.TestCase):
        backend    = "MTL4"

if __name__ == "__main__":
    print ""
    print "Testing basic PyDOLFIN linear algebra operations"
    print "------------------------------------------------"
    unittest.main()
