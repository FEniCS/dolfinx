#!/usr/bin/env py.test

"""Unit tests for TensorLayout and SparsityPattern interface"""

# Copyright (C) 2015 Jan Blechta
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

from __future__ import print_function
import pytest
from dolfin import *
import numpy as np

from dolfin_utils.test import *


backends = list(linear_algebra_backends().keys())
if 'STL' in backends:
    backends.remove('STL')
if MPI.size(mpi_comm_world()) > 1 and 'Eigen' in backends:
    backends.remove('Eigen')
backend = set_parameters_fixture("linear_algebra_backend", backends)

@fixture
def mesh():
    return UnitSquareMesh(10, 10)

@fixture
def V(mesh):
    return FunctionSpace(mesh, 'Lagrange', 1)

@fixture
def VV(mesh):
    return VectorFunctionSpace(mesh, 'Lagrange', 1)

@fixture
def TH(mesh):
    P2 = VectorElement('Lagrange', mesh.ufl_cell(), 2)
    P1 = FiniteElement('Lagrange', mesh.ufl_cell(), 1)
    return FunctionSpace(mesh, P2*P1)

@fixture
def VR(mesh):
    P1 = FiniteElement('Lagrange', mesh.ufl_cell(), 1)
    R = FiniteElement('Real', mesh.ufl_cell(), 0)
    return FunctionSpace(mesh, P1*R)


def test_layout_and_pattern_interface(backend, V, VV, TH, VR):
    for V in [V, VV, TH]:
        m = V.mesh()
        c = m.mpi_comm()
        d = V.dofmap()
        i = d.index_map()

        # Poisson problem
        u = TrialFunction(V)
        v = TestFunction(V)
        a = inner(grad(u), grad(v))*dx
        f = Expression(np.full(v.ufl_shape, "x[0]*x[1]", dtype=object).tolist(), degree=2)
        L = inner(f, v)*dx

        # Test ghosted vector (for use as dofs of FE function)
        t0 = TensorLayout(0, TensorLayout.Sparsity_DENSE)
        t0.init(c, [i], TensorLayout.Ghosts_GHOSTED)
        x = Vector()
        x.init(t0)
        u = Function(V, x)

        # Test unghosted vector (for assembly of rhs)
        t1 = TensorLayout(0, TensorLayout.Sparsity_DENSE)
        t1.init(c, [i], TensorLayout.Ghosts_UNGHOSTED)
        b = Vector()
        b.init(t1)
        assemble(L, tensor=b)

        # Build sparse tensor layout (for assembly of matrix)
        t2 = TensorLayout(0, TensorLayout.Sparsity_SPARSE)
        t2.init(c, [i, i], TensorLayout.Ghosts_UNGHOSTED)
        s2 = t2.sparsity_pattern()
        s2.init(c, [i, i])
        SparsityPatternBuilder.build(s2, m, [d, d],
                                     True, False, False, False,
                                     False, init=False)
        A = Matrix()
        A.init(t2)
        assemble(a, tensor=A)

        # Test sparsity pattern consistency
        diag = s2.num_nonzeros_diagonal()
        off_diag = s2.num_nonzeros_off_diagonal()
        # Sequential pattern returns just empty off_diagonal
        off_diag = off_diag if off_diag.any() else np.zeros(diag.shape, diag.dtype)
        local = s2.num_local_nonzeros()
        assert (local == diag + off_diag).all()
        assert local.sum() == s2.num_nonzeros()

        # Check that solve passes smoothly
        ud = np.full(v.ufl_shape, 0, dtype=object).tolist()
        bc = DirichletBC(V, ud, lambda x, b: b)
        bc.apply(A)
        bc.apply(b)
        solve(A, x, b)
