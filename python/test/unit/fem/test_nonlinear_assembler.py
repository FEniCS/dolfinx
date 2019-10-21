# Copyright (C) 2019 Nathan Sime
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for assembly"""

import math

import numpy
import pytest
from petsc4py import PETSc

import dolfin
import ufl
from dolfin import constant, functionspace
from dolfin.specialfunctions import SpatialCoordinate
from ufl import derivative, ds, dx, inner


def nest_matrix_norm(A):
    """Return norm of a MatNest matrix"""
    assert A.getType() == "nest"
    norm = 0.0
    nrows, ncols = A.getNestSize()
    for row in range(nrows):
        for col in range(ncols):
            A_sub = A.getNestSubMatrix(row, col)
            if A_sub:
                _norm = A_sub.norm()
                norm += _norm * _norm
    return math.sqrt(norm)


def test_matrix_assembly_block():
    """Test assembly of block matrices and vectors into (a) monolithic
    blocked structures, PETSc Nest structures, and monolithic structures
    in the nonlinear setting
    """
    mesh = dolfin.generation.UnitSquareMesh(dolfin.MPI.comm_world, 4, 8)

    p0, p1 = 1, 2
    P0 = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), p0)
    P1 = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), p1)

    V0 = dolfin.function.functionspace.FunctionSpace(mesh, P0)
    V1 = dolfin.function.functionspace.FunctionSpace(mesh, P1)

    def boundary(x):
        return numpy.logical_or(x[:, 0] < 1.0e-6, x[:, 0] > 1.0 - 1.0e-6)


    initial_guess_value = 1.0
    bc_value = 0.0

    u_bc = dolfin.function.Function(V1)
    with u_bc.vector.localForm() as u_local:
        u_local.set(bc_value)
    bc = dolfin.fem.dirichletbc.DirichletBC(V1, u_bc, boundary)

    # Define variational problem
    du, dp = dolfin.function.TrialFunction(V0), dolfin.function.TrialFunction(V1)
    u, p = dolfin.function.Function(V0), dolfin.function.Function(V1)
    v, q = dolfin.function.TestFunction(V0), dolfin.function.TestFunction(V1)

    zero = dolfin.Function(V0)
    f = 1.0
    g = -3.0

    F0 = inner(u, v) * dx + inner(p, v) * dx - inner(f, v)*dx
    F1 = inner(u, q) * dx + inner(p, q) * dx - inner(g, q)*dx

    a_block = [[ufl.derivative(F0, u, du), ufl.derivative(F0, p, dp)],
               [ufl.derivative(F1, u, du), ufl.derivative(F1, p, dp)]]
    L_block = [F0, F1]

    # Monolithic blocked
    x0 = dolfin.fem.create_vector_block(L_block)
    with x0.localForm() as x0_local:
        x0_local.set(initial_guess_value)

    # # Copy initial guess vector x0 into FE functions
    # offset = 0
    # for soln_vec in [u, p]:
    #     soln_vec = soln_vec.vector
    #     size_local = soln_vec.getLocalSize()
    #     iset = PETSc.IS().createGeneral(list(range(offset, offset + size_local)))
    #     x0_subvec = x0.getSubVector(iset)
    #     x0_subvec.copy(soln_vec)
    #     x0.restoreSubVector(iset, x0_subvec)
    #     soln_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    #     offset += size_local

    with u.vector.localForm() as ul:
        ul.set(initial_guess_value)
    with p.vector.localForm() as pl:
        pl.set(initial_guess_value)

    # Ghosts are updated inside assemble_vector_block
    A0 = dolfin.fem.assemble_matrix_block(a_block, [bc])
    b0 = dolfin.fem.assemble_vector_block(L_block, a_block, [bc], x0=x0, scale=-1.0)
    assert A0.getType() != "nest"
    Anorm0 = A0.norm()
    bnorm0 = b0.norm()

    # # Nested (MatNest)
    # A1 = dolfin.fem.assemble_matrix_nest(a_block, [bc])
    # Anorm1 = nest_matrix_norm(A1)
    # assert Anorm0 == pytest.approx(Anorm1, 1.0e-12)
    # b1 = dolfin.fem.assemble_vector_nest(L_block, a_block, [bc])
    # bnorm1 = math.sqrt(sum([x.norm()**2 for x in b1.getNestSubVecs()]))
    # assert bnorm0 == pytest.approx(bnorm1, 1.0e-12)

    # Monolithic version
    E = P0 * P1
    W = dolfin.function.functionspace.FunctionSpace(mesh, E)
    dU = dolfin.function.TrialFunction(W)
    U = dolfin.function.Function(W)
    u0, u1 = ufl.split(U)
    v0, v1 = dolfin.function.TestFunctions(W)

    with U.vector.localForm() as Ulocal:
        Ulocal.set(initial_guess_value)

    F = inner(u0, v0) * dx + inner(u1, v0) * dx \
        + inner(u0, v1) * dx + inner(u1, v1) * dx \
        - inner(f, v0) * ufl.dx - inner(g, v1) * dx

    J = ufl.derivative(F, U, dU)

    bc = dolfin.fem.dirichletbc.DirichletBC(W.sub(1), u_bc, boundary)
    A2 = dolfin.fem.assemble_matrix(J, [bc])
    A2.assemble()
    b2 = dolfin.fem.assemble_vector(F)
    dolfin.fem.apply_lifting(b2, [J], bcs=[[bc]], x0=[U.vector], scale=-1.0)
    b2.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    dolfin.fem.set_bc(b2, [bc], x0=U.vector, scale=-1.0)
    assert A2.getType() != "nest"
    assert A2.norm() == pytest.approx(Anorm0, 1.0e-9)
    assert b2.norm() == pytest.approx(bnorm0, 1.0e-9)


# def test_assembly_solve_block():
#     """Solve a two-field mass-matrix like problem with block matrix approaches
#     and test that solution is the same.
#     """
#     mesh = dolfin.generation.UnitSquareMesh(dolfin.MPI.comm_world, 32, 31)
#     p0, p1 = 1, 1
#     P0 = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), p0)
#     P1 = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), p1)
#     V0 = dolfin.function.functionspace.FunctionSpace(mesh, P0)
#     V1 = dolfin.function.functionspace.FunctionSpace(mesh, P1)
#
#     def boundary(x):
#         return numpy.logical_or(x[:, 0] < 1.0e-6, x[:, 0] > 1.0 - 1.0e-6)
#
#     u_bc0 = dolfin.function.Function(V0)
#     u_bc0.vector.set(50.0)
#     u_bc0.vector.ghostUpdate(
#         addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
#     u_bc1 = dolfin.function.Function(V1)
#     u_bc1.vector.set(20.0)
#     u_bc1.vector.ghostUpdate(
#         addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
#     bcs = [
#         dolfin.fem.dirichletbc.DirichletBC(V0, u_bc0, boundary),
#         dolfin.fem.dirichletbc.DirichletBC(V1, u_bc1, boundary)
#     ]
#
#     # Variational problem
#     u, p = dolfin.function.TrialFunction(V0), dolfin.function.TrialFunction(V1)
#     v, q = dolfin.function.TestFunction(V0), dolfin.function.TestFunction(V1)
#     f = 1.0
#     g = -3.0
#     zero = dolfin.Function(V0)
#
#     a00 = inner(u, v) * dx
#     a01 = zero * inner(p, v) * dx
#     a10 = zero * inner(u, q) * dx
#     a11 = inner(p, q) * dx
#     L0 = inner(f, v) * dx
#     L1 = inner(g, q) * dx
#
#     def monitor(ksp, its, rnorm):
#         pass
#         # print("Norm:", its, rnorm)
#
#     A0 = dolfin.fem.assemble_matrix_block([[a00, a01], [a10, a11]], bcs)
#     b0 = dolfin.fem.assemble_vector_block([L0, L1], [[a00, a01], [a10, a11]],
#                                           bcs)
#     A0norm = A0.norm()
#     b0norm = b0.norm()
#     x0 = A0.createVecLeft()
#     ksp = PETSc.KSP()
#     ksp.create(mesh.mpi_comm())
#     ksp.setOperators(A0)
#     ksp.setMonitor(monitor)
#     ksp.setType('cg')
#     ksp.setTolerances(rtol=1.0e-14)
#     ksp.setFromOptions()
#     ksp.solve(b0, x0)
#     x0norm = x0.norm()
#
#     # Nested (MatNest)
#     A1 = dolfin.fem.assemble_matrix_nest([[a00, a01], [a10, a11]], bcs)
#     b1 = dolfin.fem.assemble_vector_nest([L0, L1], [[a00, a01], [a10, a11]],
#                                          bcs)
#     b1norm = b1.norm()
#     assert b1norm == pytest.approx(b0norm, 1.0e-12)
#     A1norm = nest_matrix_norm(A1)
#     assert A0norm == pytest.approx(A1norm, 1.0e-12)
#
#     x1 = b1.copy()
#     ksp = PETSc.KSP()
#     ksp.create(mesh.mpi_comm())
#     ksp.setMonitor(monitor)
#     ksp.setOperators(A1)
#     ksp.setType('cg')
#     ksp.setTolerances(rtol=1.0e-12)
#     ksp.setFromOptions()
#     ksp.solve(b1, x1)
#     x1norm = x1.norm()
#     assert x1norm == pytest.approx(x0norm, rel=1.0e-12)
#
#     # Monolithic version
#     E = P0 * P1
#     W = dolfin.function.functionspace.FunctionSpace(mesh, E)
#     u0, u1 = dolfin.function.TrialFunctions(W)
#     v0, v1 = dolfin.function.TestFunctions(W)
#     a = inner(u0, v0) * dx + inner(u1, v1) * dx
#     L = inner(f, v0) * ufl.dx + inner(g, v1) * dx
#
#     V0 = dolfin.function.functionspace.FunctionSpace(mesh, P0)
#     V1 = dolfin.function.functionspace.FunctionSpace(mesh, P1)
#
#     u0_bc = dolfin.function.Function(V0)
#     u0_bc.vector.set(50.0)
#     u0_bc.vector.ghostUpdate(
#         addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
#     u1_bc = dolfin.function.Function(V1)
#     u1_bc.vector.set(20.0)
#     u1_bc.vector.ghostUpdate(
#         addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
#
#     bcs = [
#         dolfin.fem.dirichletbc.DirichletBC(W.sub(0), u0_bc, boundary),
#         dolfin.fem.dirichletbc.DirichletBC(W.sub(1), u1_bc, boundary)
#     ]
#
#     A2 = dolfin.fem.assemble_matrix(a, bcs)
#     A2.assemble()
#     b2 = dolfin.fem.assemble_vector(L)
#     dolfin.fem.apply_lifting(b2, [a], [bcs])
#     b2.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
#     dolfin.fem.set_bc(b2, bcs)
#     A2norm = A2.norm()
#     b2norm = b2.norm()
#     assert A2norm == pytest.approx(A0norm, 1.0e-12)
#     assert b2norm == pytest.approx(b0norm, 1.0e-12)
#
#     x2 = b2.copy()
#     ksp = PETSc.KSP()
#     ksp.create(mesh.mpi_comm())
#     ksp.setMonitor(monitor)
#     ksp.setOperators(A2)
#     ksp.setType('cg')
#     ksp.getPC().setType('jacobi')
#     ksp.setTolerances(rtol=1.0e-12)
#     ksp.setFromOptions()
#     ksp.solve(b2, x2)
#     x2norm = x2.norm()
#     assert x2norm == pytest.approx(x0norm, 1.0e-10)
#
#
# @pytest.mark.parametrize("mesh", [
#     dolfin.generation.UnitSquareMesh(dolfin.MPI.comm_world, 32, 31),
#     dolfin.generation.UnitCubeMesh(dolfin.MPI.comm_world, 3, 7, 8)
# ])
# def test_assembly_solve_taylor_hood(mesh):
#     """Assemble Stokes problem with Taylor-Hood elements and solve."""
#     P2 = functionspace.VectorFunctionSpace(mesh, ("Lagrange", 2))
#     P1 = functionspace.FunctionSpace(mesh, ("Lagrange", 1))
#
#     def boundary0(x):
#         """Define boundary x = 0"""
#         return x[:, 0] < 10 * numpy.finfo(float).eps
#
#     def boundary1(x):
#         """Define boundary x = 1"""
#         return x[:, 0] > (1.0 - 10 * numpy.finfo(float).eps)
#
#     u0 = dolfin.Function(P2)
#     u0.vector.set(1.0)
#     u0.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
#     bc0 = dolfin.DirichletBC(P2, u0, boundary0)
#     bc1 = dolfin.DirichletBC(P2, u0, boundary1)
#
#     u, p = dolfin.TrialFunction(P2), dolfin.TrialFunction(P1)
#     v, q = dolfin.TestFunction(P2), dolfin.TestFunction(P1)
#
#     a00 = inner(ufl.grad(u), ufl.grad(v)) * dx
#     a01 = ufl.inner(p, ufl.div(v)) * dx
#     a10 = ufl.inner(ufl.div(u), q) * dx
#     a11 = None
#
#     p00 = a00
#     p01, p10 = None, None
#     p11 = inner(p, q) * dx
#
#     # FIXME
#     # We need zero function for the 'zero' part of L
#     p_zero = dolfin.Function(P1)
#     f = dolfin.Function(P2)
#     L0 = ufl.inner(f, v) * dx
#     L1 = ufl.inner(p_zero, q) * dx
#
#     # -- Blocked and nested
#
#     A0 = dolfin.fem.assemble_matrix_nest([[a00, a01], [a10, a11]], [bc0, bc1])
#     A0norm = nest_matrix_norm(A0)
#     P0 = dolfin.fem.assemble_matrix_nest([[p00, p01], [p10, p11]], [bc0, bc1])
#     P0norm = nest_matrix_norm(P0)
#     b0 = dolfin.fem.assemble_vector_nest([L0, L1], [[a00, a01], [a10, a11]],
#                                          [bc0, bc1])
#     b0norm = b0.norm()
#
#     ksp = PETSc.KSP()
#     ksp.create(mesh.mpi_comm())
#     ksp.setOperators(A0, P0)
#     nested_IS = P0.getNestISs()
#     ksp.setType("minres")
#     pc = ksp.getPC()
#     pc.setType("fieldsplit")
#     pc.setFieldSplitIS(["u", nested_IS[0][0]], ["p", nested_IS[1][1]])
#     ksp_u, ksp_p = pc.getFieldSplitSubKSP()
#     ksp_u.setType("preonly")
#     ksp_u.getPC().setType('lu')
#     ksp_u.getPC().setFactorSolverType('mumps')
#     ksp_p.setType("preonly")
#
#     def monitor(ksp, its, rnorm):
#         # print("Num it, rnorm:", its, rnorm)
#         pass
#
#     ksp.setTolerances(rtol=1.0e-8, max_it=50)
#     ksp.setMonitor(monitor)
#     ksp.setFromOptions()
#     x0 = b0.copy()
#     ksp.solve(b0, x0)
#     assert ksp.getConvergedReason() > 0
#
#     # -- Blocked and monolithic
#
#     A1 = dolfin.fem.assemble_matrix_block([[a00, a01], [a10, a11]], [bc0, bc1])
#     assert A1.norm() == pytest.approx(A0norm, 1.0e-12)
#     P1 = dolfin.fem.assemble_matrix_block([[p00, p01], [p10, p11]], [bc0, bc1])
#     assert P1.norm() == pytest.approx(P0norm, 1.0e-12)
#     b1 = dolfin.fem.assemble_vector_block([L0, L1], [[a00, a01], [a10, a11]],
#                                           [bc0, bc1])
#     assert b1.norm() == pytest.approx(b0norm, 1.0e-12)
#
#     ksp = PETSc.KSP()
#     ksp.create(mesh.mpi_comm())
#     ksp.setOperators(A1, P1)
#     ksp.setType("minres")
#     pc = ksp.getPC()
#     pc.setType('lu')
#     pc.setFactorSolverType('mumps')
#     ksp.setTolerances(rtol=1.0e-8, max_it=50)
#     ksp.setFromOptions()
#     x1 = A1.createVecRight()
#     ksp.solve(b1, x1)
#     assert ksp.getConvergedReason() > 0
#     assert x1.norm() == pytest.approx(x0.norm(), 1e-8)
#
#     # -- Monolithic
#
#     P2 = ufl.VectorElement("Lagrange", mesh.ufl_cell(), 2)
#     P1 = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
#     TH = P2 * P1
#     W = dolfin.FunctionSpace(mesh, TH)
#     (u, p) = dolfin.TrialFunctions(W)
#     (v, q) = dolfin.TestFunctions(W)
#     a00 = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
#     a01 = ufl.inner(p, ufl.div(v)) * dx
#     a10 = ufl.inner(ufl.div(u), q) * dx
#     a = a00 + a01 + a10
#
#     p00 = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
#     p11 = ufl.inner(p, q) * dx
#     p_form = p00 + p11
#
#     f = dolfin.Function(W.sub(0).collapse())
#     p_zero = dolfin.Function(W.sub(1).collapse())
#     L0 = inner(f, v) * dx
#     L1 = inner(p_zero, q) * dx
#     L = L0 + L1
#
#     bc0 = dolfin.DirichletBC(W.sub(0), u0, boundary0)
#     bc1 = dolfin.DirichletBC(W.sub(0), u0, boundary1)
#
#     A2 = dolfin.fem.assemble_matrix(a, [bc0, bc1])
#     A2.assemble()
#     assert A2.norm() == pytest.approx(A0norm, 1.0e-12)
#     P2 = dolfin.fem.assemble_matrix(p_form, [bc0, bc1])
#     P2.assemble()
#     assert P2.norm() == pytest.approx(P0norm, 1.0e-12)
#
#     b2 = dolfin.fem.assemble_vector(L)
#     dolfin.fem.apply_lifting(b2, [a], [[bc0, bc1]])
#     b2.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
#     dolfin.fem.set_bc(b2, [bc0, bc1])
#     b2norm = b2.norm()
#     assert b2norm == pytest.approx(b0norm, 1.0e-12)
#
#     ksp = PETSc.KSP()
#     ksp.create(mesh.mpi_comm())
#     ksp.setOperators(A2, P2)
#     ksp.setType("minres")
#     pc = ksp.getPC()
#     pc.setType('lu')
#     pc.setFactorSolverType('mumps')
#
#     def monitor(ksp, its, rnorm):
#         # print("Num it, rnorm:", its, rnorm)
#         pass
#
#     ksp.setTolerances(rtol=1.0e-8, max_it=50)
#     ksp.setMonitor(monitor)
#     ksp.setFromOptions()
#     x2 = A2.createVecRight()
#     ksp.solve(b2, x2)
#     assert ksp.getConvergedReason() > 0
#     assert x0.norm() == pytest.approx(x2.norm(), 1e-8)
#
