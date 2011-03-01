"""Unit tests for the Vector interface"""

__author__ = "Garth N. Wells (gnw20@cam.ac.uk)"
__date__ = "2011-03-01"
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

    #def assemble_matrices(self, use_backend=False):
    #    " Assemble a pair of matrices, one (square) MxM and one MxN"
    #    mesh = UnitSquare(3,3)
    #
    #    V = FunctionSpace(mesh, "Lagrange", 2)
    #    W = FunctionSpace(mesh, "Lagrange", 1)
    #
    #    v = TestFunction(V)
    #    u = TrialFunction(V)
    #    s = TrialFunction(W)

    #    # Forms
    #    a = dot(grad(u),grad(v))*dx
    #    b = v*s*dx

    #    if use_backend:
    #        if self.backend == "uBLAS":
    #            backend = globals()[self.backend+self.sub_backend+'Factory_instance']()
    #        else:
    #            backend = globals()[self.backend+'Factory_instance']()
    #        return assemble(a, backend=backend), assemble(b, backend=backend)
    #    else:
    #        return assemble(a), assemble(b)

    #def assemble_vectors(self):
    #    mesh = UnitSquare(3,3)
    #
    #    V = FunctionSpace(mesh, "Lagrange", 2)
    #    W = FunctionSpace(mesh, "Lagrange", 1)

    #    v = TestFunction(V)
    #    t = TestFunction(W)

    #    return assemble(v*dx), assemble(t*dx)


    def test_create_empty_vector(self):
        v0 = Vector()
        self.assertEqual(v0.size(), 0)

    def test_create_vector(self):
        n = 301
        v1 = Vector(n)
        self.assertEqual(v1.size(), n)

    def test_copy_vector(self):
        n = 301
        v0 = Vector(n)
        v1 = Vector(v0)
        self.assertEqual(v0.size(), n)
        del v0
        self.assertEqual(v1.size(), n)

    def test_assign_and_copy_vector(self):
        n = 301
        v0 = Vector(n)
        v0[:] = 1.0
        self.assertEqual(v0.sum(), n)
        v1 = Vector(v0)
        del v0
        self.assertEqual(v1.sum(), n)

    def test_zero(self):
        v0 = Vector(301)
        v0.zero()
        self.assertEqual(v0.sum(), 0.0)

    def test_apply(self):
        v0 = Vector(301)
        v0.apply("insert")
        v0.apply("add")

    def test_str(self):
        v0 = Vector(13)
        tmp = v0.str(False)
        tmp = v0.str(True)

    def test_resize(self):
        m, n = 301, 409
        v0 = Vector()
        v0.resize(m)
        self.assertEqual(v0.size(), m)
        v0.resize(n)
        self.assertEqual(v0.size(), n)

    def test_resize_range(self):
        n = 301
        local_range = MPI.local_range(n)
        v0 = Vector()
        v0.resize(local_range)
        self.assertEqual(v0.local_range(), local_range)

    def test_size(self):
        n = 301
        v0 = Vector(301)
        self.assertEqual(v0.size(), n)

    def test_local_size(self):
        n = 301
        local_range = MPI.local_range(n)
        v0 = Vector()
        v0.resize(local_range)
        self.assertEqual(v0.local_size(), local_range[1] - local_range[0])

    def test_owns_index(self):
        m, n = 301, 25
        v0 = Vector(m)
        local_range = v0.local_range()
        in_range = local_range[0] <= n < local_range[1]
        self.assertEqual(v0.owns_index(n), in_range)


# A DataTester class that test the acces of the raw data through pointers
# This is only available for uBLAS and MTL4 backends
class DataTester(AbstractBaseTest):
    def test_vector_data(self):
        print "Test data"
        # Test for ordinary Vector
        #v,w = self.assemble_vectors()
        #array = v.array()
        #data = v.data()
        #self.assertTrue((data==array).all())

        # Test for down_casted Vector
        #v = down_cast(v)
        #data = v.data()
        #self.assertTrue((data==array).all())

class DataNotWorkingTester(AbstractBaseTest):
    def test_vector_data(self):
        print "Test data"
        #v, w = self.assemble_vectors()
        #self.assertRaises(RuntimeError,v.data)

        #v = down_cast(v)
        #def no_attribute():
        #    v.data()
        #self.assertRaises(AttributeError,no_attribute)

#
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
