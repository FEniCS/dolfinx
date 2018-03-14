"""Unit tests for the Matrix interface"""

# Copyright (C) 2011-2014 Garth N. Wells
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import pytest
from dolfin import *
from dolfin_utils.test import *

# TODO: Reuse this fixture setup code between matrix and vector tests:

# Lists of backends supporting or not supporting FooMatrix::data()
# access
data_backends = []
no_data_backends = [("PETSc", ""), ("Tpetra", "")]

# Add serial only backends
if MPI.size(MPI.comm_world) == 1:
    # TODO: What about "Dense" and "Sparse"? The sub_backend wasn't
    # used in the old test.
    data_backends += [("Eigen", "")]

# Remove backends we haven't built with
data_backends = [b for b in data_backends if has_linear_algebra_backend(b[0])]
no_data_backends = [b for b in no_data_backends if has_linear_algebra_backend(b[0])]
any_backends = data_backends + no_data_backends


# Fixtures setting up and resetting the global linear algebra backend
# for a list of backends
any_backend = set_parameters_fixture("linear_algebra_backend", any_backends,
                                     lambda x: x[0])
data_backend = set_parameters_fixture("linear_algebra_backend", data_backends,
                                      lambda x: x[0])
no_data_backend = set_parameters_fixture("linear_algebra_backend",
                                         no_data_backends, lambda x: x[0])

# With and without explicit backend choice
use_backend = true_false_fixture

class TestMatrixForAnyBackend:

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
            backend = globals()[self.backend + self.sub_backend + 'Factory'].instance()
        else:
            backend = None

        # Build square matrix with some 'empty' diagonals
        A = assemble(a, backend=backend, keep_diagonal=keep_diagonal)

        # Build non-square matrix
        B = assemble(b, backend=backend, keep_diagonal=keep_diagonal)

        return A, B

    def test_basic_la_operations(self, use_backend, any_backend):
        # Hack to make old tests work in new framework. The original
        # setup was a bit exoteric...
        # TODO: Removing use of self in this class will make it
        # clearer what happens in this test.
        self.backend, self.sub_backend = any_backend

        from numpy import ndarray, array, ones, sum

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
        with pytest.raises(TypeError):
            wrong_getitem(0)
        with pytest.raises(TypeError):
            wrong_getitem(1)
        with pytest.raises(TypeError):
            wrong_getitem(2)

        # Test __imul__ operator
        A *= 2
        assert round(A.norm('frobenius') - 2*unit_norm, 7) == 0

        # Test __idiv__ operator
        A /= 2
        assert round(A.norm('frobenius') - unit_norm, 7) == 0

        # Test __mul__ operator
        C = 4*A
        assert round(C.norm('frobenius') - 4*unit_norm, 7) == 0

        # Test __iadd__ operator
        A += C
        assert round(A.norm('frobenius') - 5*unit_norm, 7) == 0

        # Test __isub__ operator
        A -= C
        assert round(A.norm('frobenius') - unit_norm, 7) == 0

        # Test __mul__ and __add__ operator
        D = (C+A)*0.2
        assert round(D.norm('frobenius') - unit_norm, 7) == 0

        # Test __div__ and __sub__ operator
        F = (C-A)/3
        assert round(F.norm('frobenius') - unit_norm, 7) == 0

        # Test axpy
        A.axpy(10,C,True)
        assert round(A.norm('frobenius') - 41*unit_norm, 7) == 0

        # Test expected size of rectangular array
        assert A.size(0) == B.size(0)
        assert B.size(1) == 528

        # Test setitem/getitem
        #A[5,5] = 15
        #assert A[5,5] == 15

    @skip_in_parallel
    def test_numpy_array(self, use_backend, any_backend):
        self.backend, self.sub_backend = any_backend

        from numpy import ndarray, array, ones, sum

        # Assemble matrices
        A, B = self.assemble_matrices(use_backend)

        # Test to NumPy array
        A2 = A.array()
        assert isinstance(A2,ndarray)
        assert A2.shape == (2021, 2021)
        assert round(sqrt(sum(A2**2)) - A.norm('frobenius'), 7) == 0

        if self.backend == 'Eigen':
            try:
                import scipy.sparse
                import numpy.linalg
                A = as_backend_type(A)
                A3 = A.sparray()
                assert isinstance(A3, scipy.sparse.csr_matrix)
                assert round(numpy.linalg.norm(A3.todense() - A2) - 0.0, 7) == 0

                row, col, val = A.data()
                A_scipy = scipy.sparse.csr_matrix((val, col, row))
                assert round(numpy.linalg.norm(A_scipy.todense(), 'fro') \
                             - A.norm("frobenius"), 7) == 0.0

            except ImportError:
                pass

    def test_create_empty_matrix(self, any_backend):
        A = Matrix()
        assert A.size(0) == 0
        assert A.size(1) == 0
        info(A)
        info(A, True)

    def test_copy_empty_matrix(self, any_backend):
        A = Matrix()
        B = Matrix(A)
        assert B.size(0) == 0
        assert B.size(1) == 0

    def test_copy_matrix(self, any_backend):
        A0, B0 = self.assemble_matrices()

        A1 = Matrix(A0)
        assert A0.size(0) == A1.size(0)
        assert A0.size(1) == A1.size(1)
        assert A0.norm("frobenius") == A1.norm("frobenius")

        B1 = Matrix(B0)
        assert B0.size(0) == B1.size(0)
        assert B0.size(1) == B1.size(1)
        assert round(B0.norm("frobenius") - B1.norm("frobenius"), 7) == 0

    def test_ident_zeros(self, use_backend, any_backend):
        self.backend, self.sub_backend = any_backend

        # Check that PETScMatrix::ident_zeros() rethrows PETSc error
        if self.backend[0:5] == "PETSc":
            try:
                from petsc4py import PETSc
            except ImportError:
                # Can't detect PETSc version. Skip the test.
                pass
            else:
                petsc_version = PETSc.Sys.getVersion()
                if petsc_version[0:2] == (3, 7) and petsc_version[2] >= 6:
                    # Skip the test. PETSc 3.7.(>=6) is doing the diagonal
                    # check only in debug mode so that DOLFIN might not
                    # rethrow the error. Fixed in
                    # https://bitbucket.org/petsc/petsc/commits/a21198abcdd10db88d217ac122e897fcbe3179cd
                    pass
                else:
                    # NOTE: Throw try-except-else,if blocks above and do the
                    #       test everytime when PETSc requirement is bumped
                    #       to 3.8
                    A, B = self.assemble_matrices(use_backend=use_backend)
                    with pytest.raises(RuntimeError):
                        A.ident_zeros()

        # Assemble matrix A with diagonal entries
        A, B = self.assemble_matrices(use_backend=use_backend,
                                      keep_diagonal=True)

        # Find zero rows
        zero_rows = []
        for i in range(A.local_range(0)[0], A.local_range(0)[1]):
            row = A.getrow(i)[1]
            if sum(abs(row)) < DOLFIN_EPS:
                zero_rows.append(i)

        # Set zero rows to (0,...,0, 1, 0,...,0)
        A.ident_zeros()

        # Check it
        for i in zero_rows:
            cols = A.getrow(i)[0]
            row  = A.getrow(i)[1]
            for j in range(cols.size + 1):
                if i == cols[j]:
                    assert round(row[j] - 1.0, 7) == 0
                    break
            assert j < cols.size
            assert round(sum(abs(row)) - 1.0, 7) == 0

    @skip_in_parallel
    def test_ident(self, use_backend, any_backend):
        self.backend, self.sub_backend = any_backend
        if self.backend == 'Tpetra':
            pytest.skip()
        A, B = self.assemble_matrices(use_backend)
        N, M = A.size(0), A.size(1)

        # Make sure rows are not identity from before
        import numpy
        ROWS = [2, 4]
        for row in ROWS:
            block = numpy.array([[42.0]], dtype=float)
            A.set(block, numpy.array([row], dtype=numpy.intc),
                  numpy.array([row], dtype=numpy.intc))
        A.apply("insert")

        def check_row(A, row, idented):
            cols, vals = A.getrow(row)
            i = list(cols).index(row)
            if idented:
                assert vals[i] == 1
                vals[i] = 0
                for c in cols:
                    if c != row:
                        assert (vals == 0).all()
            else:
                assert vals[i] == 42.0

        for row in ROWS:
            check_row(A, row, False)
        A.ident(numpy.array(ROWS, dtype=numpy.intc))
        A.apply("insert")
        for row in ROWS:
            check_row(A, row, True)

    def test_setting_getting_diagonal(self, use_backend, any_backend):
        self.backend, self.sub_backend = any_backend

        mesh = UnitSquareMesh(21, 23)

        V = FunctionSpace(mesh, "Lagrange", 2)
        v = TestFunction(V)
        u = TrialFunction(V)
        w = Function(V)

        if use_backend:
            backend = globals()[self.backend + self.sub_backend + 'Factory'].instance()
        else:
            backend = None

        B = assemble(u*v*dx(), backend=backend, keep_diagonal=True)

        b = assemble(action(u*v*dx(), Constant(1)))
        A = B.copy()
        A.zero()
        A.set_diagonal(b)

        resultsA = Vector()
        resultsB = Vector()
        A.init_vector(resultsA, 1)
        B.init_vector(resultsB, 1)

        ones = b.copy()
        ones[:] = 1.0

        A.mult(ones, resultsA)
        B.mult(ones, resultsB)
        assert round(resultsA.norm("l2") - resultsB.norm("l2"), 7) == 0

        A.get_diagonal(w.vector())
        w.vector()[:] -= b
        assert round(w.vector().norm("l2"), 14) == 0

    @skip_in_parallel
    def test_get_set(self, use_backend, any_backend):
        self.backend, self.sub_backend = any_backend
        if self.backend == 'Tpetra':
            pytest.skip()
        A, B = self.assemble_matrices(use_backend)
        N, M = A.size(0), A.size(1)

        import numpy
        rows = numpy.array([1, 2], dtype=numpy.intc)
        cols = numpy.array([1, 2], dtype=numpy.intc)

        block_in = numpy.ones((2, 2), dtype=float) * 42
        Anp0 = A.array()
        assert (Anp0[1:3,1:3] != block_in).any()

        A.set(block_in, rows, cols)
        A.apply("insert")

        Anp1 = A.array()
        assert (Anp1[1:3,1:3] == block_in).all()

        block_out = numpy.zeros_like(block_in)
        A.get(block_out, rows, cols)
        assert (block_out == block_in).all()

    # def test_create_from_sparsity_pattern(self):

    # def test_size(self):

    # def test_local_range(self):

    # def test_zero(self):

    # def test_apply(self):

    # def test_str(self):

    # def test_resize(self):


    # Test the access of the raw data through pointers
    # This is only available for the Eigen backend
    def test_matrix_data(self, use_backend, data_backend):
        """ Test for ordinary Matrix"""
        self.backend, self.sub_backend = data_backend

        A, B = self.assemble_matrices(use_backend)
        A = as_backend_type(A)
        B = as_backend_type(B)

        array = A.array()
        rows, cols, values = A.data()
        i = 0
        for row in range(A.size(0)):
            for col in range(rows[row], rows[row+1]):
                assert array[row, cols[col]] == values[i]
                i += 1

        # Test for as_backend_typeed Matrix
        A = as_backend_type(A)
        rows, cols, values = A.data()
        for row in range(A.size(0)):
            for k in range(rows[row], rows[row+1]):
                assert array[row,cols[k]] == values[k]


    def test_matrix_nnz(self, any_backend):
        A, B = self.assemble_matrices()
        assert A.nnz() == 2992
        assert B.nnz() == 9398

        A, B = self.assemble_matrices(keep_diagonal=True)
        assert A.nnz() == 4589
        # NOTE: Following should never be tested because diagonal is not
        #       invariant w.r.t. different row and column dof reordering!
        #assert B.nnz() == ??
