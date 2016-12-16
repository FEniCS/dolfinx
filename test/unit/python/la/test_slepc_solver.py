#!/usr/bin/env py.test

"""Unit tests for the KrylovSolver interface"""

# Copyright (C) 2016 Nate Sime
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
from dolfin_utils.test import skip_if_not_PETsc_or_not_slepc, fixture

# Stiffness and mass bilinear formulations

def k(u, v):
    return inner(grad(u), grad(v))*dx

def m(u, v):
    return dot(u, v)*dx


# Fixtures

@fixture
def mesh():
    return UnitSquareMesh(32, 32)

@fixture
def V(mesh):
    return FunctionSpace(mesh, "CG", 1)

@fixture
def V_vec(mesh):
    return VectorFunctionSpace(mesh, "CG", 1)

@fixture
def K_M(V):
    u, v = TrialFunction(V), TestFunction(V)
    K_mat, M_mat = PETScMatrix(), PETScMatrix()
    x0 = PETScVector()
    L = Constant(0.0)*v*dx
    assemble_system(k(u, v), L, bcs=[], A_tensor=K_mat, b_tensor=x0)
    assemble_system(m(u, v), L, bcs=[], A_tensor=M_mat, b_tensor=x0)
    return K_mat, M_mat

@fixture
def K_M_vec(V_vec):
    u, v = TrialFunction(V_vec), TestFunction(V_vec)
    K_mat, M_mat = PETScMatrix(), PETScMatrix()
    x0 = PETScVector()
    L = dot(Constant([0.0]*V_vec.mesh().geometry().dim()), v)*dx
    assemble_system(k(u, v), L, bcs=[], A_tensor=K_mat, b_tensor=x0)
    assemble_system(m(u, v), L, bcs=[], A_tensor=M_mat, b_tensor=x0)
    return K_mat, M_mat


# Tests

@skip_if_not_PETsc_or_not_slepc
def test_slepc_eigensolver_gen_hermitian(K_M):
    "Test SLEPc eigen solver"

    K, M = K_M
    esolver = SLEPcEigenSolver(K, M)

    esolver.parameters["solver"] = "krylov-schur"
    esolver.parameters["spectral_transform"] = 'shift-and-invert'
    esolver.parameters['spectral_shift'] = 0.0
    esolver.parameters["problem_type"] = 'gen_hermitian'

    nevs = 20
    esolver.solve(nevs)

    # Test default eigenvalue
    re_0, im_0 = esolver.get_eigenvalue()
    assert near(re_0, 0.0, eps=1e-12)
    assert near(im_0, 0.0)

    re_0, im_0, v_re_0, v_im_0 = esolver.get_eigenpair()
    assert near(re_0, 0.0, eps=1e-12)
    assert near(im_0, 0.0)
    assert v_re_0.norm("l2") > 0.0
    assert near(v_im_0.norm("l2"), 0.0)

    # Test remaining eigenvalues and eigenpairs
    for j in range(1, nevs):
        re, im = esolver.get_eigenvalue(j)
        assert re > 0.0
        assert near(im, 0.0)

    for j in range(1, nevs):
        re, im, v_re, v_im = esolver.get_eigenpair(j)
        assert re > 0.0
        assert near(im, 0.0)
        assert v_re.norm("l2") > 0.0
        assert near(v_im.norm("l2"), 0.0)


@skip_if_not_PETsc_or_not_slepc
def test_slepc_null_space(K_M, V):
    "Test SLEPc eigen solver with nullspace as PETScVector"

    K, M = K_M
    esolver = SLEPcEigenSolver(K, M)

    esolver.parameters["solver"] = "jacobi-davidson"
    esolver.parameters["problem_type"] = 'gen_hermitian'
    
    u0 = Function(V)
    nullspace_basis = as_backend_type(u0.vector().copy())
    V.dofmap().set(nullspace_basis, 1.0)
    esolver.set_deflation_space(nullspace_basis)

    nevs = 20
    esolver.solve(nevs)

    for j in range(1, nevs):
        re, im, v_re, v_im = esolver.get_eigenpair(j)
        assert re > 0.0
        assert near(im, 0.0)
        assert v_re.norm("l2") > 0.0
        assert near(v_im.norm("l2"), 0.0)


@skip_if_not_PETsc_or_not_slepc
def test_slepc_vector_null_space(K_M_vec, V_vec):
    "Test SLEPc eigen solver with nullspace as VectorSpaceBasis"

    def build_nullspace(V, x):
        nullspace_basis = [x.copy() for i in range(2)]

        V.sub(0).dofmap().set(nullspace_basis[0], 1.0)
        V.sub(1).dofmap().set(nullspace_basis[1], 1.0)

        for x in nullspace_basis:
            x.apply("insert")

        # Create vector space basis and orthogonalize
        basis = VectorSpaceBasis(nullspace_basis)
        basis.orthonormalize()

        return basis

    K, M = K_M_vec
    esolver = SLEPcEigenSolver(K, M)

    esolver.parameters["solver"] = "jacobi-davidson"
    esolver.parameters["problem_type"] = 'gen_hermitian'
    
    u0 = Function(V_vec)
    nullspace_basis = build_nullspace(V_vec, u0.vector())
    esolver.set_deflation_space(nullspace_basis)

    nevs = 20
    esolver.solve(nevs)

    for j in range(1, nevs):
        re, im, v_re, v_im = esolver.get_eigenpair(j)
        assert re > 0.0
        assert near(im, 0.0)
        assert v_re.norm("l2") > 0.0
        assert near(v_im.norm("l2"), 0.0)

