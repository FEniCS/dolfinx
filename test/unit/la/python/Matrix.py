"""Unit tests for the Matrix interface"""

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
# Modified by Anders Logg 2011
# Modified by Mikael Mortensen 2011
# Modified by Jan Blechta 2013
#
# First added:  2011-03-03
# Last changed: 2013-03-22

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


    def assemble_matrices(self, use_backend=False, keep_diagonal=False):
        " Assemble a pair of matrices, one (square) MxM and one MxN"
        mesh = UnitSquareMesh(21, 23)

        V = FunctionSpace(mesh, "Lagrange", 2)
        W = FunctionSpace(mesh, "Lagrange", 1)

        v = TestFunction(V)
        u = TrialFunction(V)
        s = TrialFunction(W)

        # Forms
        a = dot(grad(u), grad(v))*ds
        b = v*s*dx

        if use_backend:
            if self.backend == "uBLAS":
                backend = globals()[self.backend + self.sub_backend + 'Factory'].instance()
            else:
                backend = globals()[self.backend + 'Factory'].instance()
            return assemble(a, backend=backend, keep_diagonal=keep_diagonal), \
                   assemble(b, backend=backend, keep_diagonal=keep_diagonal)
        else:
            return assemble(a, keep_diagonal=keep_diagonal), \
                   assemble(b, keep_diagonal=keep_diagonal)


    def test_basic_la_operations(self, use_backend=False):
        from numpy import ndarray, array, ones, sum

        # Tests bailout for this choice
        if self.backend == "uBLAS" and not use_backend:
            return

        A, B = self.assemble_matrices(use_backend)
        unit_norm = A.norm('frobenius')

        def wrong_getitem(type):
            if type == 0:
                A["0,1"]
            elif type == 1:
                A[0]
            elif type == 2:
                A[0, 0, 0]

        # Test wrong getitem
        self.assertRaises(TypeError, wrong_getitem, 0)
        self.assertRaises(TypeError, wrong_getitem, 1)
        self.assertRaises(TypeError, wrong_getitem, 2)

        # Test __imul__ operator
        A *= 2
        self.assertAlmostEqual(A.norm('frobenius'), 2*unit_norm)

        # Test __idiv__ operator
        A /= 2
        self.assertAlmostEqual(A.norm('frobenius'), unit_norm)

        # Test __mul__ operator
        C = 4*A
        self.assertAlmostEqual(C.norm('frobenius'), 4*unit_norm)

        # Test __iadd__ operator
        A += C
        self.assertAlmostEqual(A.norm('frobenius'), 5*unit_norm)

        # Test __isub__ operator
        A -= C
        self.assertAlmostEqual(A.norm('frobenius'), unit_norm)

        # Test __mul__ and __add__ operator
        D = (C+A)*0.2
        self.assertAlmostEqual(D.norm('frobenius'), unit_norm)

        # Test __div__ and __sub__ operator
        F = (C-A)/3
        self.assertAlmostEqual(F.norm('frobenius'), unit_norm)

        # Test axpy
        A.axpy(10,C,True)
        self.assertAlmostEqual(A.norm('frobenius'), 41*unit_norm)

        # Test to NumPy array
        if MPI.num_processes() == 1:
            A2 = A.array()
            self.assertTrue(isinstance(A2,ndarray))
            self.assertEqual(A2.shape, (2021, 2021))
            self.assertAlmostEqual(sqrt(sum(A2**2)), A.norm('frobenius'))

            if self.backend == 'uBLAS' and self.sub_backend == 'Sparse':
                try:
                    import scipy.sparse
                    import numpy.linalg
                    A3 = A.sparray()
                    self.assertTrue(isinstance(A3, scipy.sparse.csr_matrix))
                    self.assertAlmostEqual(numpy.linalg.norm(A3.todense() - A2), 0.0)
                except ImportError:
                    pass

        # Test expected size of rectangular array
        self.assertEqual(A.size(0), B.size(0))
        self.assertEqual(B.size(1), 528)

        # Test setitem/getitem
        #A[5,5] = 15
        #self.assertEqual(A[5,5],15)


    def test_basic_la_operations_with_backend(self):
        self.test_basic_la_operations(True)

    #def create_sparsity_pattern(self):
    #    "Create a sparsity pattern"
    #    mesh = UnitSquareMesh(34, 33)
    #
    #    V = FunctionSpace(mesh, "Lagrange", 2)
    #    W = FunctionSpace(mesh, "Lagrange", 1)
    #
    #    v = TestFunction(V)
    #    u = TrialFunction(V)
    #    s = TrialFunction(W)
    #
    #    # Forms
    #    a = dot(grad(u), grad(v))*dx
    #    b = v*s*dx

    def test_create_empty_matrix(self):
        A = Matrix()
        self.assertEqual(A.size(0), 0)
        self.assertEqual(A.size(1), 0)
        info(A)
        info(A, True)

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
        self.assertAlmostEqual(B0.norm("frobenius"), B1.norm("frobenius"))

    def test_compress_matrix(self):

        A0, B0 = self.assemble_matrices()
        A0_norm_0 = A0.norm('frobenius')
        A0.compress()
        A0_norm_1 = A0.norm('frobenius')
        self.assertAlmostEqual(A0_norm_0, A0_norm_1)

    def test_ident_zeros_AND_keep_diagonal(self):

        A, B = self.assemble_matrices()
        if self.backend[0:5] == "PETSc":
          # FIXME: how to supress catched PETSc error output?
          self.assertRaises(RuntimeError, A.ident_zeros)
        else:
          A.ident_zeros()

        A, B = self.assemble_matrices(keep_diagonal=True)
        A.ident_zeros()

    #def test_create_from_sparsity_pattern(self):

    #def test_size(self):

    #def test_local_range(self):

    #def test_zero(self):

    #def test_apply(self):

    #def test_str(self):

    #def test_resize(self):


# A DataTester class that test the acces of the raw data through pointers
# This is only available for uBLAS and MTL4 backends
class DataTester:
    def test_matrix_data(self, use_backend=False):
        """ Test for ordinary Matrix"""
        # Tests bailout for this choice
        if self.backend == "uBLAS" and \
               (not use_backend or self.sub_backend =="Dense"):
            return

        A, B = self.assemble_matrices(use_backend)
        array = A.array()
        rows, cols, values = A.data()
        i = 0
        for row in xrange(A.size(0)):
            for col in xrange(rows[row], rows[row+1]):
                self.assertEqual(array[row, cols[col]],values[i])
                i += 1

        # Test none writeable of a shallow copy of the data
        rows, cols, values = A.data(False)
        def write_data(data):
            data[0] = 1
        self.assertRaises(StandardError, write_data, rows)
        self.assertRaises(StandardError, write_data, cols)
        self.assertRaises(StandardError, write_data, values)

        # Test for as_backend_typeed Matrix
        A = as_backend_type(A)
        rows, cols, values = A.data()
        for row in xrange(A.size(0)):
            for k in xrange(rows[row], rows[row+1]):
                self.assertEqual(array[row,cols[k]], values[k])

    def test_matrix_data_use_backend(self):
        self.test_matrix_data(True)

class DataNotWorkingTester:
    def test_matrix_data(self):
        A, B = self.assemble_matrices()
        self.assertRaises(RuntimeError, A.data)

        A = as_backend_type(A)
        self.assertRaises(RuntimeError, A.data)

if MPI.num_processes() == 1:
    class uBLASSparseTester(DataTester, AbstractBaseTest, unittest.TestCase):
        backend     = "uBLAS"
        sub_backend = "Sparse"

    class uBLASDenseTester(DataTester, AbstractBaseTest, unittest.TestCase):
        backend     = "uBLAS"
        sub_backend = "Dense"

    if has_linear_algebra_backend("MTL4"):
        class MTL4Tester(DataTester, AbstractBaseTest, unittest.TestCase):
            backend    = "MTL4"

    if has_linear_algebra_backend("PETScCusp"):
        class PETScCuspTester(DataNotWorkingTester, AbstractBaseTest, unittest.TestCase):
            backend    = "PETScCusp"

if has_linear_algebra_backend("PETSc"):
    class PETScTester(DataNotWorkingTester, AbstractBaseTest, unittest.TestCase):
        backend    = "PETSc"

if has_linear_algebra_backend("Epetra"):
    class EpetraTester(DataNotWorkingTester, AbstractBaseTest, unittest.TestCase):
        backend    = "Epetra"

#class STLTester(DataNotWorkingTester, AbstractBaseTest, unittest.TestCase):
#    backend    = "STL"

if __name__ == "__main__":

    # Turn off DOLFIN output
    set_log_active(False)

    print ""
    print "Testing DOLFIN Matrix classes"
    print "------------------------------------------------"
    unittest.main()
