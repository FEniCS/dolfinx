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
from ufl import derivative, dx, inner


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

    initial_guess_u = lambda x: numpy.sin(numpy.pi*x[:, 0])*numpy.sin(numpy.pi*x[:, 1])
    initial_guess_p = lambda x: -x[:, 0]**2 - x[:, 1]**3
    bc_value = lambda x: numpy.cos(numpy.pi*x[:, 0])*numpy.cos(numpy.pi*x[:, 1])

    u_bc = dolfin.function.Function(V1)
    u_bc.interpolate(bc_value)
    bc = dolfin.fem.dirichletbc.DirichletBC(V1, u_bc, boundary)

    # Define variational problem
    du, dp = dolfin.function.TrialFunction(V0), dolfin.function.TrialFunction(V1)
    u, p = dolfin.function.Function(V0), dolfin.function.Function(V1)
    v, q = dolfin.function.TestFunction(V0), dolfin.function.TestFunction(V1)

    u.interpolate(initial_guess_u)
    p.interpolate(initial_guess_p)

    f = 1.0
    g = -3.0

    F0 = inner(u, v) * dx + inner(p, v) * dx - inner(f, v) * dx
    F1 = inner(u, q) * dx + inner(p, q) * dx - inner(g, q) * dx

    a_block = [[derivative(F0, u, du), derivative(F0, p, dp)],
               [derivative(F1, u, du), derivative(F1, p, dp)]]
    L_block = [F0, F1]

    # Monolithic blocked
    x0 = dolfin.fem.create_vector_block(L_block)
    dolfin.cpp.la.scatter_local_vectors(x0, [u.vector.array_r, p.vector.array_r], [u.function_space.dofmap.index_map, p.function_space.dofmap.index_map])

    # Ghosts are updated inside assemble_vector_block
    A0 = dolfin.fem.assemble_matrix_block(a_block, [bc])
    b0 = dolfin.fem.assemble_vector_block(L_block, a_block, [bc], x0=x0, scale=-1.0)
    assert A0.getType() != "nest"
    Anorm0 = A0.norm()
    bnorm0 = b0.norm()

    # Nested (MatNest)
    x1 = dolfin.fem.create_vector_nest(L_block)
    for x1_soln_pair in zip(x1.getNestSubVecs(), (u, p)):
        x1_sub, soln_sub = x1_soln_pair
        soln_sub.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        soln_sub.vector.copy(result=x1_sub)
        x1_sub.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    A1 = dolfin.fem.assemble_matrix_nest(a_block, [bc])
    b1 = dolfin.fem.assemble_vector_nest(L_block, a_block, [bc], x0=x1, scale=-1.0)

    assert A1.getType() == "nest"
    assert nest_matrix_norm(A1) == pytest.approx(Anorm0, 1.0e-12)
    assert b1.norm() == pytest.approx(bnorm0, 1.0e-12)

    # Monolithic version
    E = P0 * P1
    W = dolfin.function.functionspace.FunctionSpace(mesh, E)
    dU = dolfin.function.TrialFunction(W)
    U = dolfin.function.Function(W)
    u0, u1 = ufl.split(U)
    v0, v1 = dolfin.function.TestFunctions(W)

    U.interpolate(lambda x: numpy.column_stack(
        (initial_guess_u(x), initial_guess_p(x))))

    F = inner(u0, v0) * dx + inner(u1, v0) * dx \
        + inner(u0, v1) * dx + inner(u1, v1) * dx \
        - inner(f, v0) * ufl.dx - inner(g, v1) * dx

    J = derivative(F, U, dU)

    bc = dolfin.fem.dirichletbc.DirichletBC(W.sub(1), u_bc, boundary)
    A2 = dolfin.fem.assemble_matrix(J, [bc])
    A2.assemble()
    b2 = dolfin.fem.assemble_vector(F)
    dolfin.fem.apply_lifting(b2, [J], bcs=[[bc]], x0=[U.vector], scale=-1.0)
    b2.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    dolfin.fem.set_bc(b2, [bc], x0=U.vector, scale=-1.0)
    assert A2.getType() != "nest"
    assert A2.norm() == pytest.approx(Anorm0, 1.0e-12)
    assert b2.norm() == pytest.approx(bnorm0, 1.0e-12)


class NonlinearPDE_SNESProblem():
    def __init__(self, F, J, soln_vars, bcs, P=None):
        self.L = F
        self.a = J
        self.a_precon = P
        self.bcs = bcs
        self.soln_vars = soln_vars

    def F_mono(self, snes, x, F):
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        with x.localForm() as _x, self.soln_vars.vector.localForm() as _u:
            _u[:] = _x
        with F.localForm() as f_local:
            f_local.set(0.0)
        dolfin.fem.assemble_vector(F, self.L)
        dolfin.fem.apply_lifting(F, [self.a], [self.bcs], x0=[x], scale=-1.0)
        F.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        dolfin.fem.set_bc(F, self.bcs, x, -1.0)

    def J_mono(self, snes, x, J, P):
        J.zeroEntries()
        dolfin.fem.assemble_matrix(J, self.a, self.bcs, diagonal=1.0)
        J.assemble()
        if self.a_precon is not None:
            P.zeroEntries()
            dolfin.fem.assemble_matrix(P, self.a_precon, self.bcs, diagonal=1.0)
            P.assemble()

    def F_block(self, snes, x, F):
        assert x.getType() != "nest"
        assert F.getType() != "nest"
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        offset = 0
        x_array = x.getArray(readonly=True)
        for var in self.soln_vars:
            size_local = var.vector.getLocalSize()
            var.vector.array[:] = x_array[offset:offset + size_local]
            var.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
            offset += size_local

        dolfin.fem.assemble_vector_block(F, self.L, self.a, self.bcs, x0=x, scale=-1.0)

    def J_block(self, snes, x, J, P):
        assert x.getType() != "nest" and J.getType() != "nest" and P.getType() != "nest"
        J.zeroEntries()
        dolfin.fem.assemble_matrix_block(J, self.a, self.bcs, diagonal=1.0)
        J.assemble()
        if self.a_precon is not None:
            P.zeroEntries()
            dolfin.fem.assemble_matrix_block(P, self.a_precon, self.bcs, diagonal=1.0)
            P.assemble()

    def F_nest(self, snes, x, F):
        assert x.getType() == "nest" and F.getType() == "nest"
        for x_sub, var_sub in zip(x.getNestSubVecs(), self.soln_vars):
            x_sub.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
            with x_sub.localForm() as _x, var_sub.vector.localForm() as _u:
                _u[:] = _x
        dolfin.fem.assemble_vector_nest(F, self.L, self.a, self.bcs, x0=x, scale=-1.0)
        # Must assemble F here in the case of nest matrices
        F.assemble()

    def J_nest(self, snes, x, J, P):
        assert x.getType() == "nest" and J.getType() == "nest" and P.getType() == "nest"
        J.zeroEntries()
        dolfin.fem.assemble_matrix_nest(J, self.a, self.bcs, diagonal=1.0)
        J.assemble()
        if self.a_precon is not None:
            P.zeroEntries()
            dolfin.fem.assemble_matrix_nest(P, self.a_precon, self.bcs, diagonal=1.0)
            P.assemble()


def test_assembly_solve_block():
    """Solve a two-field nonlinear diffusion like problem with block matrix
    approaches and test that solution is the same.
    """
    mesh = dolfin.generation.UnitSquareMesh(dolfin.MPI.comm_world, 12, 11)
    p0, p1 = 1, 1
    P0 = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), p0)
    P1 = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), p1)
    V0 = dolfin.function.functionspace.FunctionSpace(mesh, P0)
    V1 = dolfin.function.functionspace.FunctionSpace(mesh, P1)

    bc_val_0 = 1.0
    bc_val_1 = 2.0

    initial_guess = 1.0

    def boundary(x):
        return numpy.logical_or(x[:, 0] < 1.0e-6, x[:, 0] > 1.0 - 1.0e-6)

    u_bc0 = dolfin.function.Function(V0)
    u_bc0.vector.set(bc_val_0)
    u_bc0.vector.ghostUpdate(
        addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    u_bc1 = dolfin.function.Function(V1)
    u_bc1.vector.set(bc_val_1)
    u_bc1.vector.ghostUpdate(
        addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    bcs = [
        dolfin.fem.dirichletbc.DirichletBC(V0, u_bc0, boundary),
        dolfin.fem.dirichletbc.DirichletBC(V1, u_bc1, boundary)
    ]

    # Block and Nest variational problem
    u, p = dolfin.function.Function(V0), dolfin.function.Function(V1)
    du, dp = dolfin.function.TrialFunction(V0), dolfin.function.TrialFunction(V1)
    v, q = dolfin.function.TestFunction(V0), dolfin.function.TestFunction(V1)

    f = 1.0
    g = -3.0

    F = [inner((u**2 + 1) * ufl.grad(u), ufl.grad(v)) * dx - inner(f, v) * dx,
         inner((p**2 + 1) * ufl.grad(p), ufl.grad(q)) * dx - inner(g, q) * dx]

    J = [[derivative(F[0], u, du), derivative(F[0], p, dp)],
         [derivative(F[1], u, du), derivative(F[1], p, dp)]]

    # -- Blocked version
    Jmat0 = dolfin.fem.create_matrix_block(J)
    Fvec0 = dolfin.fem.create_vector_block(F)

    snes = PETSc.SNES().create(dolfin.MPI.comm_world)
    snes.setTolerances(rtol=1.0e-15, max_it=10)

    snes.getKSP().setType("preonly")
    snes.getKSP().getPC().setType("lu")
    snes.getKSP().getPC().setFactorSolverType("mumps")

    problem = NonlinearPDE_SNESProblem(F, J, [u, p], bcs)
    snes.setFunction(problem.F_block, Fvec0)
    snes.setJacobian(problem.J_block, J=Jmat0, P=None)

    x0 = dolfin.fem.create_vector_block(F)
    with x0.localForm() as x0l:
        x0l.set(initial_guess)
    snes.solve(None, x0)

    assert snes.getKSP().getConvergedReason() > 0
    assert snes.getConvergedReason() > 0

    J0norm = Jmat0.norm()
    F0norm = Fvec0.norm()
    x0norm = x0.norm()

    # -- Nested (MatNest)
    Jmat1 = dolfin.fem.create_matrix_nest(J)
    Fvec1 = dolfin.fem.create_vector_nest(F)

    snes = PETSc.SNES().create(dolfin.MPI.comm_world)
    snes.setTolerances(rtol=1.0e-15, max_it=10)

    nested_IS = Jmat1.getNestISs()

    snes.getKSP().setType("fgmres")
    snes.getKSP().setTolerances(rtol=1e-12)
    snes.getKSP().getPC().setType("fieldsplit")
    snes.getKSP().getPC().setFieldSplitIS(["u", nested_IS[0][0]], ["p", nested_IS[1][1]])

    ksp_u, ksp_p = snes.getKSP().getPC().getFieldSplitSubKSP()
    ksp_u.setType("preonly")
    ksp_u.getPC().setType('lu')
    ksp_u.getPC().setFactorSolverType('mumps')
    ksp_p.setType("preonly")
    ksp_p.getPC().setType('lu')
    ksp_p.getPC().setFactorSolverType('mumps')

    problem = NonlinearPDE_SNESProblem(F, J, [u, p], bcs)
    snes.setFunction(problem.F_nest, Fvec1)
    snes.setJacobian(problem.J_nest, J=Jmat1, P=None)

    x1 = dolfin.fem.create_vector_nest(F)
    x1.set(initial_guess)
    for x1_sub in x1.getNestSubVecs():
        x1_sub.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    u.vector.zeroEntries()
    p.vector.zeroEntries()

    snes.solve(None, x1)

    assert snes.getKSP().getConvergedReason() > 0
    assert snes.getConvergedReason() > 0
    assert x1.getType() == "nest"
    assert Jmat1.getType() == "nest"
    assert Fvec1.getType() == "nest"

    J1norm = nest_matrix_norm(Jmat1)
    F1norm = Fvec1.norm()
    x1norm = x1.norm()

    assert J1norm == pytest.approx(J0norm, 1.0e-12)
    assert F1norm == pytest.approx(F0norm, 1.0e-12)
    assert x1norm == pytest.approx(x0norm, 1.0e-12)

    # -- Monolithic version
    E = P0 * P1
    W = dolfin.function.functionspace.FunctionSpace(mesh, E)
    U = dolfin.function.Function(W)
    dU = dolfin.function.TrialFunction(W)
    u0, u1 = ufl.split(U)
    v0, v1 = dolfin.function.TestFunctions(W)

    F = inner((u0**2 + 1) * ufl.grad(u0), ufl.grad(v0)) * dx \
        + inner((u1**2 + 1) * ufl.grad(u1), ufl.grad(v1)) * dx \
        - inner(f, v0) * ufl.dx - inner(g, v1) * dx
    J = derivative(F, U, dU)

    u0_bc = dolfin.function.Function(V0)
    u0_bc.vector.set(bc_val_0)
    u0_bc.vector.ghostUpdate(
        addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    u1_bc = dolfin.function.Function(V1)
    u1_bc.vector.set(bc_val_1)
    u1_bc.vector.ghostUpdate(
        addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    bcs = [
        dolfin.fem.dirichletbc.DirichletBC(W.sub(0), u0_bc, boundary),
        dolfin.fem.dirichletbc.DirichletBC(W.sub(1), u1_bc, boundary)
    ]

    Jmat2 = dolfin.fem.create_matrix(J)
    Fvec2 = dolfin.fem.create_vector(F)

    snes = PETSc.SNES().create(dolfin.MPI.comm_world)
    snes.setTolerances(rtol=1.0e-15, max_it=10)

    snes.getKSP().setType("preonly")
    snes.getKSP().getPC().setType("lu")
    snes.getKSP().getPC().setFactorSolverType("mumps")

    problem = NonlinearPDE_SNESProblem(F, J, U, bcs)
    snes.setFunction(problem.F_mono, Fvec2)
    snes.setJacobian(problem.J_mono, J=Jmat2, P=None)

    x2 = dolfin.fem.create_vector(F)
    with x2.localForm() as x2l:
        x2l.set(initial_guess)
    snes.solve(None, x2)

    assert snes.getKSP().getConvergedReason() > 0
    assert snes.getConvergedReason() > 0

    J2norm = Jmat2.norm()
    F2norm = Fvec2.norm()
    x2norm = x2.norm()

    assert J2norm == pytest.approx(J0norm, 1.0e-12)
    assert F2norm == pytest.approx(F0norm, 1.0e-12)
    assert x2norm == pytest.approx(x0norm, 1.0e-12)


@pytest.mark.parametrize("mesh", [
    dolfin.generation.UnitSquareMesh(dolfin.MPI.comm_world, 12, 11),
    dolfin.generation.UnitCubeMesh(dolfin.MPI.comm_world, 3, 5, 4)
])
def test_assembly_solve_taylor_hood(mesh):
    """Assemble Stokes problem with Taylor-Hood elements and solve."""
    P2 = dolfin.function.functionspace.VectorFunctionSpace(mesh, ("Lagrange", 2))
    P1 = dolfin.function.functionspace.FunctionSpace(mesh, ("Lagrange", 1))

    def boundary0(x):
        """Define boundary x = 0"""
        return x[:, 0] < 10 * numpy.finfo(float).eps

    def boundary1(x):
        """Define boundary x = 1"""
        return x[:, 0] > (1.0 - 10 * numpy.finfo(float).eps)

    u0 = dolfin.Function(P2)
    with u0.vector.localForm() as x:
        x.set(1.0)
    bcs = [dolfin.DirichletBC(P2, u0, boundary0),
           dolfin.DirichletBC(P2, u0, boundary1)]

    u, p = dolfin.Function(P2), dolfin.Function(P1)
    du, dp = dolfin.TrialFunction(P2), dolfin.TrialFunction(P1)
    v, q = dolfin.TestFunction(P2), dolfin.TestFunction(P1)

    F = [inner(ufl.grad(u), ufl.grad(v)) * dx + inner(p, ufl.div(v)) * dx,
         inner(ufl.div(u), q) * dx]
    J = [[derivative(F[0], u, du), derivative(F[0], p, dp)],
         [derivative(F[1], u, du), derivative(F[1], p, dp)]]
    P = [[J[0][0], None],
         [None, inner(dp, q) * dx]]

    # -- Blocked and monolithic

    Jmat0 = dolfin.fem.create_matrix_block(J)
    Pmat0 = dolfin.fem.create_matrix_block(P)
    Fvec0 = dolfin.fem.create_vector_block(F)

    snes = PETSc.SNES().create(dolfin.MPI.comm_world)
    snes.setTolerances(rtol=1.0e-15, max_it=10)

    snes.getKSP().setType("minres")
    snes.getKSP().getPC().setType("lu")
    snes.getKSP().getPC().setFactorSolverType("mumps")

    problem = NonlinearPDE_SNESProblem(F, J, [u, p], bcs, P=P)
    snes.setFunction(problem.F_block, Fvec0)
    snes.setJacobian(problem.J_block, J=Jmat0, P=Pmat0)

    x0 = dolfin.fem.create_vector_block(F)
    with x0.localForm() as x0l:
        x0l.set(0.0)
    snes.solve(None, x0)

    assert snes.getConvergedReason() > 0

    # -- Blocked and nested

    Jmat1 = dolfin.fem.create_matrix_nest(J)
    Pmat1 = dolfin.fem.create_matrix_nest(P)
    Fvec1 = dolfin.fem.create_vector_nest(F)

    snes = PETSc.SNES().create(dolfin.MPI.comm_world)
    snes.setTolerances(rtol=1.0e-15, max_it=10)

    nested_IS = Jmat1.getNestISs()

    snes.getKSP().setType("minres")
    snes.getKSP().setTolerances(rtol=1e-12)
    snes.getKSP().getPC().setType("fieldsplit")
    snes.getKSP().getPC().setFieldSplitIS(["u", nested_IS[0][0]], ["p", nested_IS[1][1]])

    ksp_u, ksp_p = snes.getKSP().getPC().getFieldSplitSubKSP()
    ksp_u.setType("preonly")
    ksp_u.getPC().setType('lu')
    ksp_u.getPC().setFactorSolverType('mumps')
    ksp_p.setType("preonly")
    ksp_p.getPC().setType('lu')
    ksp_p.getPC().setFactorSolverType('mumps')

    problem = NonlinearPDE_SNESProblem(F, J, [u, p], bcs, P=P)
    snes.setFunction(problem.F_nest, Fvec1)
    snes.setJacobian(problem.J_nest, J=Jmat1, P=Pmat1)

    x1 = dolfin.fem.create_vector_nest(F)
    x1.zeroEntries()
    snes.solve(None, x1)

    assert snes.getConvergedReason() > 0
    assert nest_matrix_norm(Jmat1) == pytest.approx(Jmat0.norm(), 1.0e-12)
    assert Fvec1.norm() == pytest.approx(Fvec0.norm(), 1.0e-12)
    assert x1.norm() == pytest.approx(x0.norm(), 1.0e-12)

    # -- Monolithic

    P2 = ufl.VectorElement("Lagrange", mesh.ufl_cell(), 2)
    P1 = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    TH = P2 * P1
    W = dolfin.FunctionSpace(mesh, TH)
    U = dolfin.Function(W)
    dU = dolfin.TrialFunction(W)
    u, p = ufl.split(U)
    du, dp = ufl.split(dU)
    v, q = dolfin.TestFunctions(W)

    F = inner(ufl.grad(u), ufl.grad(v)) * dx \
        + inner(p, ufl.div(v)) * dx \
        + inner(ufl.div(u), q) * dx
    J = derivative(F, U, dU)
    P = inner(ufl.grad(du), ufl.grad(v)) * dx \
        + inner(dp, q) * dx

    bcs = [dolfin.DirichletBC(W.sub(0), u0, boundary0),
           dolfin.DirichletBC(W.sub(0), u0, boundary1)]

    Jmat2 = dolfin.fem.create_matrix(J)
    Pmat2 = dolfin.fem.create_matrix(P)
    Fvec2 = dolfin.fem.create_vector(F)

    snes = PETSc.SNES().create(dolfin.MPI.comm_world)
    snes.setTolerances(rtol=1.0e-15, max_it=10)

    snes.getKSP().setType("minres")
    snes.getKSP().getPC().setType("lu")
    snes.getKSP().getPC().setFactorSolverType("mumps")

    problem = NonlinearPDE_SNESProblem(F, J, U, bcs, P=P)
    snes.setFunction(problem.F_mono, Fvec2)
    snes.setJacobian(problem.J_mono, J=Jmat2, P=Pmat2)

    x2 = dolfin.fem.create_vector(F)
    with x2.localForm() as x2l:
        x2l.set(0.0)
    snes.solve(None, x2)

    assert snes.getConvergedReason() > 0
    assert Jmat2.norm() == pytest.approx(Jmat0.norm(), 1.0e-12)
    assert Fvec2.norm() == pytest.approx(Fvec0.norm(), 1.0e-12)
    assert x2.norm() == pytest.approx(x0.norm(), 1.0e-12)
