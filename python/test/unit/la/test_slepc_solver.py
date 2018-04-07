"""Unit tests for the KrylovSolver interface"""

# Copyright (C) 2016 Nate Sime
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import pytest
from dolfin import *
from dolfin_utils.test import skip_if_not_PETsc_or_not_slepc, fixture

# Stiffness and mass bilinear formulations

def k(u, v):
    return inner(grad(u), grad(v))*dx

def m(u, v):
    return dot(u, v)*dx

# Wrappers around SLEPcEigenSolver for test_slepc_eigensolver_gen_hermitian
def SLEPcEigenSolverOperatorsFromInit(K, M):
    return SLEPcEigenSolver(K, M)

def SLEPcEigenSolverOperatorsFromSetOperators(K, M):
    slepc_eigen_solver = SLEPcEigenSolver(K.mpi_comm())
    slepc_eigen_solver.set_operators(K, M)
    return slepc_eigen_solver

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
    L = dot(Constant([0.0]*V_vec.mesh().geometry.dim), v)*dx
    assemble_system(k(u, v), L, bcs=[], A_tensor=K_mat, b_tensor=x0)
    assemble_system(m(u, v), L, bcs=[], A_tensor=M_mat, b_tensor=x0)
    return K_mat, M_mat


# Tests

@skip_if_not_PETsc_or_not_slepc
def test_set_from_options():
    "Test SLEPc options prefixes"

    prefix = "my_slepc_"
    solver = SLEPcEigenSolver(MPI.comm_world)
    solver.set_options_prefix(prefix)
    solver.set_from_options()

    assert solver.get_options_prefix() == prefix

@skip_if_not_PETsc_or_not_slepc
@pytest.mark.parametrize("SLEPcEigenSolverWrapper", (SLEPcEigenSolverOperatorsFromInit, SLEPcEigenSolverOperatorsFromSetOperators))
def test_slepc_eigensolver_gen_hermitian(K_M, SLEPcEigenSolverWrapper):
    "Test SLEPc eigen solver"

    K, M = K_M
    esolver = SLEPcEigenSolverWrapper(K, M)

    esolver.parameters["solver"] = "krylov-schur"
    esolver.parameters["spectral_transform"] = 'shift-and-invert'
    esolver.parameters['spectral_shift'] = 0.0
    esolver.parameters["problem_type"] = "gen_hermitian"

    nevs = 20
    esolver.solve(nevs)

    # Test default eigenvalue
    re_0, im_0 = esolver.get_eigenvalue(0)
    assert near(re_0, 0.0, eps=1e-12)
    assert near(im_0, 0.0)

    re_0, im_0, v_re_0, v_im_0 = esolver.get_eigenpair(0)
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
    esolver.parameters["problem_type"] = "gen_hermitian"

    u0 = Function(V)
    nullspace_basis = as_backend_type(u0.vector().copy())
    V.dofmap().set(nullspace_basis, 1.0)
    esolver.set_deflation_space(VectorSpaceBasis([nullspace_basis]))

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
    esolver.parameters["problem_type"] = "gen_hermitian"

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


@skip_if_not_PETsc_or_not_slepc
def test_slepc_initial_space(K_M, V):
    "Test SLEPc eigen solver with inital space as PETScVector"

    K, M = K_M
    esolver = SLEPcEigenSolver(K, M)

    esolver.parameters["solver"] = "jacobi-davidson"
    esolver.parameters["problem_type"] = "gen_hermitian"

    u0 = as_backend_type(interpolate(Constant(2.0), V).vector())
    esolver.set_initial_space(VectorSpaceBasis([u0]))

    nevs = 20
    esolver.solve(nevs)

    for j in range(1, nevs):
        re, im, v_re, v_im = esolver.get_eigenpair(j)
        assert re > 0.0
        assert near(im, 0.0)
        assert v_re.norm("l2") > 0.0
        assert near(v_im.norm("l2"), 0.0)


@skip_if_not_PETsc_or_not_slepc
def test_slepc_vector_initial_space(K_M_vec, V_vec):
    "Test SLEPc eigen solver with initial space as VectorSpaceBasis"

    K, M = K_M_vec
    esolver = SLEPcEigenSolver(K, M)

    esolver.parameters["solver"] = "jacobi-davidson"
    esolver.parameters["problem_type"] = "gen_hermitian"

    u0 = as_backend_type(interpolate(Constant((2.0, 1.0)), V_vec).vector())
    u1 = as_backend_type(interpolate(Constant((0.0, -4.0)), V_vec).vector())
    initial_space = VectorSpaceBasis([u0, u1])
    esolver.set_initial_space(initial_space)

    nevs = 20
    esolver.solve(nevs)

    for j in range(1, nevs):
        re, im, v_re, v_im = esolver.get_eigenpair(j)
        assert re > 0.0
        assert near(im, 0.0)
        assert v_re.norm("l2") > 0.0
        assert near(v_im.norm("l2"), 0.0)
