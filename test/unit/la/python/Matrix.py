"""Unit tests for the Matrix interface"""

__author__ = "Garth N. Wells (gnw20@cam.ac.uk)"
__date__ = "2011-03-03"
__copyright__ = "Copyright (C) 2011 Garth N. Wells"
__license__  = "GNU LGPL Version 2.1"

import unittest
from dolfin import *

class AbstractBaseTest(object):
    count = 0
    def setUp(self):
        if self.backend != "default":
            parameters.linear_algebra_backend = self.backend
        type(self).count += 1
        if type(self).count == 1:
            # Only print this message once per class instance
            print "\nRunning:",type(self).__name__

    def assemble_matrices(self, use_backend=False):
        " Assemble a pair of matrices, one (square) MxM and one MxN"
        mesh = UnitSquare(21, 23)

        V = FunctionSpace(mesh, "Lagrange", 2)
        W = FunctionSpace(mesh, "Lagrange", 1)

        v = TestFunction(V)
        u = TrialFunction(V)
        s = TrialFunction(W)

        # Forms
        a = dot(grad(u),grad(v))*dx
        b = v*s*dx

        if use_backend:
            if self.backend == "uBLAS":
                backend = globals()[self.backend+self.sub_backend+'Factory_instance']()
            else:
                backend = globals()[self.backend+'Factory_instance']()
            return assemble(a, backend=backend), assemble(b, backend=backend)
        else:
            return assemble(a), assemble(b)

    #def assemble_vectors(self):
    #    mesh = UnitSquare(3,3)
    #
    #    V = FunctionSpace(mesh, "Lagrange", 2)
    #    W = FunctionSpace(mesh, "Lagrange", 1)

    #    v = TestFunction(V)
    #    t = TestFunction(W)

    #    return assemble(v*dx), assemble(t*dx)


    def test_create_empty_matrix(self):
        A = Matrix()
        self.assertEqual(A.size(0), 0)
        self.assertEqual(A.size(1), 0)

    def test_copy_empty_matrix(self):
        A = Matrix()
        B = Matrix(A)
        self.assertEqual(B.size(0), 0)
        self.assertEqual(B.size(1), 0)

    def test_copy_matrix(self):
        A0, B0 = self.assemble_matrices()

        A1 = Matrix(A0)
        self.assertEqual(A0.size(0), A1.size(0))
        self.assertEqual(A0.size(1), A1.size(1))
        self.assertEqual(A0.norm("frobenius"), A1.norm("frobenius"))

        B1 = Matrix(B0)
        self.assertEqual(B0.size(0), B1.size(0))
        self.assertEqual(B0.size(1), B1.size(1))
        self.assertEqual(B0.norm("frobenius"), B1.norm("frobenius"))

    #def test_create_from_sparsity_pattern(self):

    #def test_size(self):

    #def test_local_range(self):

    #def test_zero(self):

    #def test_apply(self):

    #def test_str(self):

    #def test_resize(self):


# A DataTester class that test the acces of the raw data through pointers
# This is only available for uBLAS and MTL4 backends
class DataTester(AbstractBaseTest):
    def test_vector_data(self):
        # Test for ordinary Vector
        v = Vector(301)
        array = v.array()
        data = v.data()
        self.assertTrue((data == array).all())

        # Test for down_casted Vector
        v = down_cast(v)
        data = v.data()
        self.assertTrue((data==array).all())

class DataNotWorkingTester(AbstractBaseTest):
    def test_vector_data(self):
        v = Vector(301)
        self.assertRaises(RuntimeError, v.data)

        v = down_cast(v)
        def no_attribute():
            v.data()
        self.assertRaises(AttributeError,no_attribute)


if MPI.num_processes() <= 1:
    class uBLASSparseTester(DataTester, unittest.TestCase):
        backend     = "uBLAS"
        sub_backend = "Sparse"

    class uBLASDenseTester(DataTester, unittest.TestCase):
        backend     = "uBLAS"
        sub_backend = "Dense"

    if has_la_backend("MTL4"):
        class MTL4Tester(DataTester, unittest.TestCase):
            backend    = "MTL4"

if has_la_backend("PETSc"):
    class PETScTester(DataNotWorkingTester, unittest.TestCase):
        backend    = "PETSc"

if has_la_backend("Epetra"):
    class EpetraTester(DataNotWorkingTester, unittest.TestCase):
        backend    = "Epetra"

if __name__ == "__main__":
    # Turn of DOLFIN output
    logging(False)

    print ""
    print "Testing DOLFIN Vector class"
    print "------------------------------------------------"
    unittest.main()
