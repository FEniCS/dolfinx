#!/usr/bin/env py.test

"""Unit tests for the linear algebra interface"""

# Copyright (C) 2008 Johan Hake
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
# First added:  2008-09-30
# Last changed: 2014-05-30

from __future__ import print_function
import pytest
from six import integer_types
from dolfin import *
import sys

from dolfin_utils.test import *

# Lists of backends supporting or not supporting data access
data_backends = []
no_data_backends = [("PETSc", ""), ("Tpetra", "")]

# Add serial only backends
if MPI.size(mpi_comm_world()) == 1:
    # TODO: What about "Dense" and "Sparse"? The sub_backend wasn't
    # used in the old test.
    data_backends += [("Eigen", "")]
    no_data_backends += [("PETScCusp", "")]

# TODO: STL tests were disabled in old test framework, and do not work now:
# If we have PETSc, STL Vector gets typedefed to one of these and data
# test will not work. If none of these backends are available
# STLVector defaults to EigenVEctor, which data will work
#if has_linear_algebra_backend("PETSc"):
#    no_data_backends += [("STL", "")]
#else:
#    data_backends += [("STL", "")]

# Remove backends we haven't built with
data_backends = [b for b in data_backends if has_linear_algebra_backend(b[0])]
no_data_backends = [b for b in no_data_backends if has_linear_algebra_backend(b[0])]
any_backends = data_backends + no_data_backends

# Fixtures setting up and resetting the global linear algebra backend
# for a list of backends
any_backend     = set_parameters_fixture("linear_algebra_backend",
                                         any_backends, lambda x: x[0])
data_backend    = set_parameters_fixture("linear_algebra_backend",
                                         data_backends, lambda x: x[0])
no_data_backend = set_parameters_fixture("linear_algebra_backend",
                                         no_data_backends, lambda x: x[0])

# With and without explicit backend choice
use_backend = true_false_fixture


def xtest_deterministic_partition():
    # Works with parmetis, not with scotch with mpirun -np 3
    mesh1 = UnitSquareMesh(3, 3)
    mesh2 = UnitSquareMesh(3, 3)
    V1 = FunctionSpace(mesh1, "Lagrange", 1)
    V2 = FunctionSpace(mesh2, "Lagrange", 1)
    assert V1.dofmap().ownership_range() == V2.dofmap().ownership_range()


def assemble_vectors(mesh):
    V = FunctionSpace(mesh, "Lagrange", 2)
    W = FunctionSpace(mesh, "Lagrange", 1)
    v = TestFunction(V)
    t = TestFunction(W)
    return assemble(v*dx), assemble(t*dx)


def get_forms(mesh):
    V = FunctionSpace(mesh, "Lagrange", 2)
    W = FunctionSpace(mesh, "Lagrange", 1)

    v = TestFunction(V)
    u = TrialFunction(V)
    s = TrialFunction(W)

    # Forms
    a = dot(grad(u),grad(v))*dx
    b = v*s*dx
    return a, b


class TestBasicLaOperations:
    def test_vector(self, any_backend):
        self.backend, self.sub_backend = any_backend
        from numpy import ndarray, linspace, array, fromiter
        from numpy import int,int0,int16,int32,int64
        from numpy import uint,uint0,uint16,uint32,uint64
        mesh = UnitSquareMesh(3, 3)
        v, w = assemble_vectors(mesh)

        # Get local ownership range (relevant for parallel vectors)
        n0, n1 = v.local_range()
        distributed = True
        if (n1 - n0) == v.size():
            distributed = False

        # Test set and access with different integers
        lind = 2
        for T in [int,int16,int32,int64,uint,uint0,uint16,uint32,uint64,\
                  int0,integer_types[-1]]:
            v[T(lind)] = 2.0
            assert round(sum(v[T(lind)] - 2.0), 7) == 0

        A = v.copy()
        B = as_backend_type(v.copy())
        gind = 5
        lind = gind-n0

        # Test global index access
        if A.owns_index(gind):
            assert round(sum(A[lind] - B[lind]), 7) == 0

        lind0 = 5
        round(sum(A[lind0] - B[lind0]), 7) == 0

        B *= 0.5
        A *= 2
        assert round(sum(A[lind0] - 4*B[lind0]), 7) == 0

        B /= 2
        A /= 0.5
        assert round(sum(A[lind0] - 16*B[lind0]), 7) == 0

        val1 = A[lind0]
        val2 = B[lind0]

        A += B
        assert round(sum(A[lind0] - val1-val2), 7) == 0

        A -= B
        assert round(sum(A[lind0] - val1), 7) == 0

        C = 16*B
        assert round(sum(A[lind0] - C[lind0]), 7) == 0

        D = (C + B)*5
        assert round(sum(D[lind0] - (val1 + val2)*5), 7) == 0

        F = (A-B)/4
        assert round(sum(F[lind0] - (val1 - val2)/4), 7) == 0

        A.axpy(100, B)
        assert round(sum(A[lind0] - val1 - val2*100), 7) == 0

        A2 = A.array()
        assert isinstance(A2,ndarray)
        assert A2.shape == (n1 - n0, )

        assert round(sum(A2[lind0] - A[lind0]), 7) == 0
        assert round(MPI.sum(A.mpi_comm(), A2.sum()) - A.sum(), 7) == 0

        B2 = B.array()

        inds = [1,3,6,9,15,20,24,28,32,40,50,60,70,100000]

        # Extract owned local indices
        inds0  = array([i for i in inds if v.owns_index(i)])
        linds0 = [i-n0 for i in inds0]
        linds1 = array(linds0, 'i')
        linds2 = array(linds0, 'I')

        A[linds2] = inds0
        A2[linds0] = inds0

        G  = A[linds0]
        G1 = A[linds1]
        G2 = A2[linds2]

        G3 = A[A > 1]
        G4 = A2[A2 > 1]

        A3 = fromiter(A, "d")

        if A.owns_index(15):
            a = A[15-n0]

        b = 1.e10

        assert round(G1.sum() - G.sum(), 7) == 0
        assert round(G2.sum() - G.sum(), 7) == 0
        assert len(G3) == len(G4)
        assert round(G3.sum() - G4.sum(), 7) == 0
        assert all(val==G[i] for i, val in enumerate(G))
        assert (G==G1).all()
        assert (G<=G1).all()
        assert (G>=G1).all()
        assert not (G<G1).any()
        assert not (G>G1).any()
        if A.owns_index(15): assert a in A
        assert b not in A

        assert (A3==A2).all()

        A[:] = A2
        assert (A.array()==A2).all()

        H  = A.copy()
        H._assign(0.0)
        H[linds0] = G

        C[:] = 2
        D._assign(2)
        assert round(sum(C[0] - 2), 7) == 0
        assert round(sum(C[len(linds0)-1] - 2), 7) == 0
        assert round(C.sum() - D.sum(), 7) == 0

        C[linds0] = 3
        assert round(C[linds0].sum() - 3*len(linds0), 7) == 0

        def wrong_index(ind):
            A[ind]

        with pytest.raises(IndexError):
            wrong_index(-len(A)-1)
        with pytest.raises(IndexError):
            wrong_index(A[-1])
        with pytest.raises(IndexError):
            wrong_index(len(A)+1)
        with pytest.raises(TypeError):
            wrong_index("jada")
        with pytest.raises(TypeError):
            wrong_index(.5)
        with pytest.raises(IndexError):
            wrong_index([-len(A)-1, 2])
        with pytest.raises(IndexError):
            wrong_index([len(A), 2])

        def wrong_dim(ind0, ind1):
            A[ind0] = B[ind1]

        with pytest.raises(IndexError):
            wrong_dim([0,2], [0,2,4])
        with pytest.raises(IndexError):
            wrong_dim([0,2], slice(0,4,1))

        A2 = A.array()
        A *= B
        A2 *= B2
        I = A*B
        I2 = A2*B2

        assert round(A.sum() - MPI.sum(A.mpi_comm(), A2.sum()), 7) == 0
        assert round(I.sum() - MPI.sum(A.mpi_comm(), I2.sum()), 7) == 0

        def wrong_assign(A, ind):
            A[linds0[::2]] = linds0[::2]

        with pytest.raises(TypeError):
            wrong_assign(A, linds2)


    def test_matrix_vector(self, any_backend, use_backend):
        self.backend, self.sub_backend = any_backend
        from numpy import dot, absolute

        mesh = UnitSquareMesh(3, 3)

        a, b = get_forms(mesh)
        if use_backend:
            backend = getattr(cpp, self.backend+self.sub_backend+'Factory').instance()
        else:
            backend = None

        A = assemble(a, backend=backend)
        B = assemble(b, backend=backend)

        v, w = assemble_vectors(mesh)

        # Get local ownership range (relevant for parallel vectors)
        n0, n1 = v.local_range()

        distributed = True
        if (n1 - n0) == v.size():
            distributed = False

        # Reference values
        v_norm  = 0.181443684651
        w_norm  = 0.278394377377
        A_norm  = 31.947874212
        B_norm  = 0.11052313564
        Av_norm = 0.575896483442
        Bw_norm = 0.0149136743079
        Cv_norm = 0.00951459156865

        assert round(v.norm('l2') - v_norm, 7) == 0
        assert round(w.norm('l2') - w_norm, 7) == 0
        assert round(A.norm('frobenius') - A_norm, 7) == 0
        assert round(B.norm('frobenius') - B_norm, 7) == 0

        u = A*v

        assert isinstance(u, type(v))
        assert len(u) == len(v)

        # Test basic square matrix multiply results
        assert round(v.norm('l2') - v_norm, 7) == 0
        assert round(u.norm('l2') - Av_norm, 7) == 0

        # Test rectangular matrix multiply results
        assert round((B*w).norm('l2') - Bw_norm, 7) == 0

        # Test transpose multiply (rectangular)
        x = Vector()
        B.transpmult(v, x)
        assert round(x.norm('l2') - Cv_norm, 7) == 0

        # Miscellaneous tests
        u2 = 2*u - A*v
        assert round(sum(u2[4] - u[4]), 7) == 0

        u3 = 2*u + -1.0*(A*v)
        assert round(sum(u3[4] - u[4]), 7) == 0

        # Numpy arrays are not alligned in parallel
        if distributed:
            return

        v_numpy = v.array()
        A_numpy = A.array()

        u_numpy = dot(A_numpy, v_numpy)
        u_numpy2 = A*v_numpy

        assert absolute(u.array() - u_numpy).sum() < DOLFIN_EPS*len(v)
        assert absolute(u_numpy2 - u_numpy).sum() < DOLFIN_EPS*len(v)


    def test_vector_data(self, no_data_backend):
        mesh = UnitSquareMesh(3, 3)
        v, w = assemble_vectors(mesh)
        v = as_backend_type(v)
        def no_attribute():
            v.data()

        with pytest.raises(AttributeError):
            no_attribute()
