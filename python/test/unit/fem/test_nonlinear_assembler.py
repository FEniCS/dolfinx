# Copyright (C) 2019 Nathan Sime
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for assembly"""

import math

import numpy as np
import pytest

import ufl
from dolfinx.cpp.la.petsc import scatter_local_vectors
from dolfinx.fem import (Function, FunctionSpace, VectorFunctionSpace,
                         bcs_by_block, dirichletbc, extract_function_spaces,
                         form, locate_dofs_topological)
from dolfinx.fem.petsc import (apply_lifting, apply_lifting_nest,
                               assemble_matrix, assemble_matrix_block,
                               assemble_matrix_nest, assemble_vector,
                               assemble_vector_block, assemble_vector_nest,
                               create_matrix, create_matrix_block,
                               create_matrix_nest, create_vector,
                               create_vector_block, create_vector_nest, set_bc,
                               set_bc_nest)
from dolfinx.mesh import (GhostMode, create_unit_cube, create_unit_square,
                          locate_entities_boundary)
from ufl import derivative, dx, inner

from mpi4py import MPI
from petsc4py import PETSc


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


def test_matrix_assembly_block_nl():
    """Test assembly of block matrices and vectors into (a) monolithic
    blocked structures, PETSc Nest structures, and monolithic structures
    in the nonlinear setting
    """
    mesh = create_unit_square(MPI.COMM_WORLD, 4, 8)
    p0, p1 = 1, 2
    P0 = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), p0)
    P1 = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), p1)
    V0 = FunctionSpace(mesh, P0)
    V1 = FunctionSpace(mesh, P1)

    def initial_guess_u(x):
        return np.sin(x[0]) * np.sin(x[1])

    def initial_guess_p(x):
        return -x[0]**2 - x[1]**3

    def bc_value(x):
        return np.cos(x[0]) * np.cos(x[1])

    facetdim = mesh.topology.dim - 1
    bndry_facets = locate_entities_boundary(mesh, facetdim, lambda x: np.logical_or(np.isclose(x[0], 0.0),
                                                                                    np.isclose(x[0], 1.0)))

    u_bc = Function(V1)
    u_bc.interpolate(bc_value)
    bdofs = locate_dofs_topological(V1, facetdim, bndry_facets)
    bc = dirichletbc(u_bc, bdofs)

    # Define variational problem
    du, dp = ufl.TrialFunction(V0), ufl.TrialFunction(V1)
    u, p = Function(V0), Function(V1)
    v, q = ufl.TestFunction(V0), ufl.TestFunction(V1)

    u.interpolate(initial_guess_u)
    p.interpolate(initial_guess_p)

    f = 1.0
    g = -3.0

    F0 = inner(u, v) * dx + inner(p, v) * dx - inner(f, v) * dx
    F1 = inner(u, q) * dx + inner(p, q) * dx - inner(g, q) * dx

    a_block = form([[derivative(F0, u, du), derivative(F0, p, dp)],
                    [derivative(F1, u, du), derivative(F1, p, dp)]])
    L_block = form([F0, F1])

    # Monolithic blocked
    x0 = create_vector_block(L_block)
    scatter_local_vectors(x0, [u.vector.array_r, p.vector.array_r],
                          [(u.function_space.dofmap.index_map, u.function_space.dofmap.index_map_bs),
                           (p.function_space.dofmap.index_map, p.function_space.dofmap.index_map_bs)])
    x0.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    # Ghosts are updated inside assemble_vector_block
    A0 = assemble_matrix_block(a_block, bcs=[bc])
    b0 = assemble_vector_block(L_block, a_block, bcs=[bc], x0=x0, scale=-1.0)
    A0.assemble()
    assert A0.getType() != "nest"
    Anorm0 = A0.norm()
    bnorm0 = b0.norm()

    # Nested (MatNest)
    x1 = create_vector_nest(L_block)
    for x1_soln_pair in zip(x1.getNestSubVecs(), (u, p)):
        x1_sub, soln_sub = x1_soln_pair
        soln_sub.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        soln_sub.vector.copy(result=x1_sub)
        x1_sub.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    A1 = assemble_matrix_nest(a_block, bcs=[bc])
    b1 = assemble_vector_nest(L_block)
    apply_lifting_nest(b1, a_block, bcs=[bc], x0=x1, scale=-1.0)
    for b_sub in b1.getNestSubVecs():
        b_sub.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    bcs0 = bcs_by_block([L.function_spaces[0] for L in L_block], [bc])

    set_bc_nest(b1, bcs0, x1, scale=-1.0)
    A1.assemble()

    assert A1.getType() == "nest"
    assert nest_matrix_norm(A1) == pytest.approx(Anorm0, 1.0e-12)
    assert b1.norm() == pytest.approx(bnorm0, 1.0e-12)

    # Monolithic version
    E = P0 * P1
    W = FunctionSpace(mesh, E)
    dU = ufl.TrialFunction(W)
    U = Function(W)
    u0, u1 = ufl.split(U)
    v0, v1 = ufl.TestFunctions(W)

    U.sub(0).interpolate(initial_guess_u)
    U.sub(1).interpolate(initial_guess_p)

    F = inner(u0, v0) * dx + inner(u1, v0) * dx + inner(u0, v1) * dx + inner(u1, v1) * dx \
        - inner(f, v0) * ufl.dx - inner(g, v1) * dx
    J = derivative(F, U, dU)
    F, J = form(F), form(J)

    bdofsW_V1 = locate_dofs_topological((W.sub(1), V1), facetdim, bndry_facets)

    bc = dirichletbc(u_bc, bdofsW_V1, W.sub(1))
    A2 = assemble_matrix(J, bcs=[bc])
    A2.assemble()
    b2 = assemble_vector(F)
    apply_lifting(b2, [J], bcs=[[bc]], x0=[U.vector], scale=-1.0)
    b2.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b2, [bc], x0=U.vector, scale=-1.0)
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
        with x.localForm() as _x:
            self.soln_vars.x.array[:] = _x.array_r
        with F.localForm() as f_local:
            f_local.set(0.0)
        assemble_vector(F, self.L)
        apply_lifting(F, [self.a], bcs=[self.bcs], x0=[x], scale=-1.0)
        F.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        set_bc(F, self.bcs, x, -1.0)

    def J_mono(self, snes, x, J, P):
        J.zeroEntries()
        assemble_matrix(J, self.a, bcs=self.bcs, diagonal=1.0)
        J.assemble()
        if self.a_precon is not None:
            P.zeroEntries()
            assemble_matrix(P, self.a_precon, bcs=self.bcs, diagonal=1.0)
            P.assemble()

    def F_block(self, snes, x, F):
        assert x.getType() != "nest"
        assert F.getType() != "nest"
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        with F.localForm() as f_local:
            f_local.set(0.0)

        offset = 0
        x_array = x.getArray(readonly=True)
        for var in self.soln_vars:
            size_local = var.vector.getLocalSize()
            var.vector.array[:] = x_array[offset: offset + size_local]
            var.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
            offset += size_local

        assemble_vector_block(F, self.L, self.a, bcs=self.bcs, x0=x, scale=-1.0)

    def J_block(self, snes, x, J, P):
        assert x.getType() != "nest" and J.getType() != "nest" and P.getType() != "nest"
        J.zeroEntries()
        assemble_matrix_block(J, self.a, bcs=self.bcs, diagonal=1.0)
        J.assemble()
        if self.a_precon is not None:
            P.zeroEntries()
            assemble_matrix_block(P, self.a_precon, bcs=self.bcs, diagonal=1.0)
            P.assemble()

    def F_nest(self, snes, x, F):
        assert x.getType() == "nest" and F.getType() == "nest"
        # Update solution
        x = x.getNestSubVecs()
        for x_sub, var_sub in zip(x, self.soln_vars):
            x_sub.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
            with x_sub.localForm() as _x:
                var_sub.x.array[:] = _x.array_r

        # Assemble
        bcs1 = bcs_by_block(extract_function_spaces(self.a, 1), self.bcs)
        for L, F_sub, a in zip(self.L, F.getNestSubVecs(), self.a):
            with F_sub.localForm() as F_sub_local:
                F_sub_local.set(0.0)
            assemble_vector(F_sub, L)
            apply_lifting(F_sub, a, bcs=bcs1, x0=x, scale=-1.0)
            F_sub.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

        # Set bc value in RHS
        bcs0 = bcs_by_block(extract_function_spaces(self.L), self.bcs)
        for F_sub, bc, x_sub in zip(F.getNestSubVecs(), bcs0, x):
            set_bc(F_sub, bc, x_sub, -1.0)

        # Must assemble F here in the case of nest matrices
        F.assemble()

    def J_nest(self, snes, x, J, P):
        assert J.getType() == "nest" and P.getType() == "nest"
        J.zeroEntries()
        assemble_matrix_nest(J, self.a, bcs=self.bcs, diagonal=1.0)
        J.assemble()
        if self.a_precon is not None:
            P.zeroEntries()
            assemble_matrix_nest(P, self.a_precon, bcs=self.bcs, diagonal=1.0)
            P.assemble()


def test_assembly_solve_block_nl():
    """Solve a two-field nonlinear diffusion like problem with block matrix
    approaches and test that solution is the same.
    """
    mesh = create_unit_square(MPI.COMM_WORLD, 12, 11)
    p = 1
    P = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), p)
    V0 = FunctionSpace(mesh, P)
    V1 = V0.clone()

    def bc_val_0(x):
        return x[0]**2 + x[1]**2

    def bc_val_1(x):
        return np.sin(x[0]) * np.cos(x[1])

    def initial_guess_u(x):
        return np.sin(x[0]) * np.sin(x[1])

    def initial_guess_p(x):
        return -x[0]**2 - x[1]**3

    facetdim = mesh.topology.dim - 1
    bndry_facets = locate_entities_boundary(mesh, facetdim, lambda x: np.logical_or(np.isclose(x[0], 0.0),
                                                                                    np.isclose(x[0], 1.0)))

    u_bc0 = Function(V0)
    u_bc0.interpolate(bc_val_0)
    u_bc1 = Function(V1)
    u_bc1.interpolate(bc_val_1)
    bdofs0 = locate_dofs_topological(V0, facetdim, bndry_facets)
    bdofs1 = locate_dofs_topological(V1, facetdim, bndry_facets)
    bcs = [dirichletbc(u_bc0, bdofs0), dirichletbc(u_bc1, bdofs1)]

    # Block and Nest variational problem
    u, p = Function(V0), Function(V1)
    du, dp = ufl.TrialFunction(V0), ufl.TrialFunction(V1)
    v, q = ufl.TestFunction(V0), ufl.TestFunction(V1)

    f = 1.0
    g = -3.0

    F = [inner((u**2 + 1) * ufl.grad(u), ufl.grad(v)) * dx - inner(f, v) * dx,
         inner((p**2 + 1) * ufl.grad(p), ufl.grad(q)) * dx - inner(g, q) * dx]
    J = [[derivative(F[0], u, du), derivative(F[0], p, dp)],
         [derivative(F[1], u, du), derivative(F[1], p, dp)]]

    F, J = form(F), form(J)

    def blocked_solve():
        """Blocked version"""
        Jmat = create_matrix_block(J)
        Fvec = create_vector_block(F)

        snes = PETSc.SNES().create(MPI.COMM_WORLD)
        snes.setTolerances(rtol=1.0e-15, max_it=10)

        problem = NonlinearPDE_SNESProblem(F, J, [u, p], bcs)
        snes.setFunction(problem.F_block, Fvec)
        snes.setJacobian(problem.J_block, J=Jmat, P=None)

        u.interpolate(initial_guess_u)
        p.interpolate(initial_guess_p)

        x = create_vector_block(F)
        scatter_local_vectors(x, [u.vector.array_r, p.vector.array_r],
                              [(u.function_space.dofmap.index_map, u.function_space.dofmap.index_map_bs),
                               (p.function_space.dofmap.index_map, p.function_space.dofmap.index_map_bs)])
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        snes.solve(None, x)
        assert snes.getKSP().getConvergedReason() > 0
        assert snes.getConvergedReason() > 0
        return x.norm()

    def nested_solve():
        """Nested version"""
        Jmat = create_matrix_nest(J)
        assert Jmat.getType() == "nest"
        Fvec = create_vector_nest(F)
        assert Fvec.getType() == "nest"

        snes = PETSc.SNES().create(MPI.COMM_WORLD)
        snes.setTolerances(rtol=1.0e-15, max_it=10)

        nested_IS = Jmat.getNestISs()

        snes.getKSP().setType("gmres")
        snes.getKSP().setTolerances(rtol=1e-12)
        snes.getKSP().getPC().setType("fieldsplit")
        snes.getKSP().getPC().setFieldSplitIS(["u", nested_IS[0][0]], ["p", nested_IS[1][1]])

        problem = NonlinearPDE_SNESProblem(F, J, [u, p], bcs)
        snes.setFunction(problem.F_nest, Fvec)
        snes.setJacobian(problem.J_nest, J=Jmat, P=None)

        u.interpolate(initial_guess_u)
        p.interpolate(initial_guess_p)

        x = create_vector_nest(F)
        assert x.getType() == "nest"
        for x_soln_pair in zip(x.getNestSubVecs(), (u, p)):
            x_sub, soln_sub = x_soln_pair
            soln_sub.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
            soln_sub.vector.copy(result=x_sub)
            x_sub.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        snes.solve(None, x)
        assert snes.getKSP().getConvergedReason() > 0
        assert snes.getConvergedReason() > 0
        return x.norm()

    def monolithic_solve():
        """Monolithic version"""
        E = P * P
        W = FunctionSpace(mesh, E)
        U = Function(W)
        dU = ufl.TrialFunction(W)
        u0, u1 = ufl.split(U)
        v0, v1 = ufl.TestFunctions(W)

        F = inner((u0**2 + 1) * ufl.grad(u0), ufl.grad(v0)) * dx \
            + inner((u1**2 + 1) * ufl.grad(u1), ufl.grad(v1)) * dx \
            - inner(f, v0) * ufl.dx - inner(g, v1) * dx
        J = derivative(F, U, dU)

        F, J = form(F), form(J)

        u0_bc = Function(V0)
        u0_bc.interpolate(bc_val_0)
        u1_bc = Function(V1)
        u1_bc.interpolate(bc_val_1)
        bdofsW0_V0 = locate_dofs_topological((W.sub(0), V0), facetdim, bndry_facets)
        bdofsW1_V1 = locate_dofs_topological((W.sub(1), V1), facetdim, bndry_facets)
        bcs = [dirichletbc(u0_bc, bdofsW0_V0, W.sub(0)), dirichletbc(u1_bc, bdofsW1_V1, W.sub(1))]

        Jmat = create_matrix(J)
        Fvec = create_vector(F)

        snes = PETSc.SNES().create(MPI.COMM_WORLD)
        snes.setTolerances(rtol=1.0e-15, max_it=10)

        problem = NonlinearPDE_SNESProblem(F, J, U, bcs)
        snes.setFunction(problem.F_mono, Fvec)
        snes.setJacobian(problem.J_mono, J=Jmat, P=None)

        U.sub(0).interpolate(initial_guess_u)
        U.sub(1).interpolate(initial_guess_p)

        x = create_vector(F)
        x.array = U.vector.array_r

        snes.solve(None, x)
        assert snes.getKSP().getConvergedReason() > 0
        assert snes.getConvergedReason() > 0
        return x.norm()

    norm0 = blocked_solve()
    norm1 = nested_solve()
    norm2 = monolithic_solve()
    assert norm1 == pytest.approx(norm0, 1.0e-12)
    assert norm2 == pytest.approx(norm0, 1.0e-12)


@pytest.mark.parametrize("mesh", [
    create_unit_square(MPI.COMM_WORLD, 12, 11, ghost_mode=GhostMode.none),
    create_unit_square(MPI.COMM_WORLD, 12, 11, ghost_mode=GhostMode.shared_facet),
    create_unit_cube(MPI.COMM_WORLD, 3, 5, 4, ghost_mode=GhostMode.none),
    create_unit_cube(MPI.COMM_WORLD, 3, 5, 4, ghost_mode=GhostMode.shared_facet)
])
def test_assembly_solve_taylor_hood_nl(mesh):
    """Assemble Stokes problem with Taylor-Hood elements and solve."""
    gdim = mesh.geometry.dim
    P2 = VectorFunctionSpace(mesh, ("Lagrange", 2))
    P1 = FunctionSpace(mesh, ("Lagrange", 1))

    def boundary0(x):
        """Define boundary x = 0"""
        return np.isclose(x[0], 0.0)

    def boundary1(x):
        """Define boundary x = 1"""
        return np.isclose(x[0], 1.0)

    def initial_guess_u(x):
        u_init = np.row_stack((np.sin(x[0]) * np.sin(x[1]),
                               np.cos(x[0]) * np.cos(x[1])))
        if gdim == 3:
            u_init = np.row_stack((u_init, np.cos(x[2])))
        return u_init

    def initial_guess_p(x):
        return -x[0]**2 - x[1]**3

    u_bc_0 = Function(P2)
    u_bc_0.interpolate(lambda x: np.row_stack(tuple(x[j] + float(j) for j in range(gdim))))

    u_bc_1 = Function(P2)
    u_bc_1.interpolate(lambda x: np.row_stack(tuple(np.sin(x[j]) for j in range(gdim))))

    facetdim = mesh.topology.dim - 1
    bndry_facets0 = locate_entities_boundary(mesh, facetdim, boundary0)
    bndry_facets1 = locate_entities_boundary(mesh, facetdim, boundary1)

    bdofs0 = locate_dofs_topological(P2, facetdim, bndry_facets0)
    bdofs1 = locate_dofs_topological(P2, facetdim, bndry_facets1)

    bcs = [dirichletbc(u_bc_0, bdofs0), dirichletbc(u_bc_1, bdofs1)]

    u, p = Function(P2), Function(P1)
    du, dp = ufl.TrialFunction(P2), ufl.TrialFunction(P1)
    v, q = ufl.TestFunction(P2), ufl.TestFunction(P1)

    F = [inner(ufl.grad(u), ufl.grad(v)) * dx + inner(p, ufl.div(v)) * dx,
         inner(ufl.div(u), q) * dx]
    J = [[derivative(F[0], u, du), derivative(F[0], p, dp)],
         [derivative(F[1], u, du), derivative(F[1], p, dp)]]
    P = [[J[0][0], None],
         [None, inner(dp, q) * dx]]

    F, J, P = form(F), form(J), form(P)

    # -- Blocked and monolithic

    Jmat0 = create_matrix_block(J)
    Pmat0 = create_matrix_block(P)
    Fvec0 = create_vector_block(F)

    snes = PETSc.SNES().create(MPI.COMM_WORLD)
    snes.setTolerances(rtol=1.0e-15, max_it=10)
    snes.getKSP().setType("minres")

    problem = NonlinearPDE_SNESProblem(F, J, [u, p], bcs, P=P)
    snes.setFunction(problem.F_block, Fvec0)
    snes.setJacobian(problem.J_block, J=Jmat0, P=Pmat0)

    u.interpolate(initial_guess_u)
    p.interpolate(initial_guess_p)

    x0 = create_vector_block(F)
    with u.vector.localForm() as _u, p.vector.localForm() as _p:
        scatter_local_vectors(x0, [_u.array_r, _p.array_r],
                              [(u.function_space.dofmap.index_map, u.function_space.dofmap.index_map_bs),
                               (p.function_space.dofmap.index_map, p.function_space.dofmap.index_map_bs)])
    x0.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    snes.solve(None, x0)

    assert snes.getConvergedReason() > 0

    # -- Blocked and nested

    Jmat1 = create_matrix_nest(J)
    Pmat1 = create_matrix_nest(P)
    Fvec1 = create_vector_nest(F)

    snes = PETSc.SNES().create(MPI.COMM_WORLD)
    snes.setTolerances(rtol=1.0e-15, max_it=10)

    nested_IS = Jmat1.getNestISs()

    snes.getKSP().setType("minres")
    snes.getKSP().setTolerances(rtol=1e-12)
    snes.getKSP().getPC().setType("fieldsplit")
    snes.getKSP().getPC().setFieldSplitIS(["u", nested_IS[0][0]], ["p", nested_IS[1][1]])

    problem = NonlinearPDE_SNESProblem(F, J, [u, p], bcs, P=P)
    snes.setFunction(problem.F_nest, Fvec1)
    snes.setJacobian(problem.J_nest, J=Jmat1, P=Pmat1)

    u.interpolate(initial_guess_u)
    p.interpolate(initial_guess_p)

    x1 = create_vector_nest(F)
    for x1_soln_pair in zip(x1.getNestSubVecs(), (u, p)):
        x1_sub, soln_sub = x1_soln_pair
        soln_sub.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        soln_sub.vector.copy(result=x1_sub)
        x1_sub.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    x1.set(0.0)
    snes.solve(None, x1)

    assert snes.getConvergedReason() > 0
    assert nest_matrix_norm(Jmat1) == pytest.approx(Jmat0.norm(), 1.0e-12)
    assert Fvec1.norm() == pytest.approx(Fvec0.norm(), 1.0e-12)
    assert x1.norm() == pytest.approx(x0.norm(), 1.0e-12)

    # -- Monolithic

    P2_el = ufl.VectorElement("Lagrange", mesh.ufl_cell(), 2)
    P1_el = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    TH = P2_el * P1_el
    W = FunctionSpace(mesh, TH)
    U = Function(W)
    dU = ufl.TrialFunction(W)
    u, p = ufl.split(U)
    du, dp = ufl.split(dU)
    v, q = ufl.TestFunctions(W)

    F = inner(ufl.grad(u), ufl.grad(v)) * dx + inner(p, ufl.div(v)) * dx \
        + inner(ufl.div(u), q) * dx
    J = derivative(F, U, dU)
    P = inner(ufl.grad(du), ufl.grad(v)) * dx + inner(dp, q) * dx

    F, J, P = form(F), form(J), form(P)

    bdofsW0_P2_0 = locate_dofs_topological((W.sub(0), P2), facetdim, bndry_facets0)
    bdofsW0_P2_1 = locate_dofs_topological((W.sub(0), P2), facetdim, bndry_facets1)

    bcs = [dirichletbc(u_bc_0, bdofsW0_P2_0, W.sub(0)), dirichletbc(u_bc_1, bdofsW0_P2_1, W.sub(0))]

    Jmat2 = create_matrix(J)
    Pmat2 = create_matrix(P)
    Fvec2 = create_vector(F)

    snes = PETSc.SNES().create(MPI.COMM_WORLD)
    snes.setTolerances(rtol=1.0e-15, max_it=10)
    snes.getKSP().setType("minres")

    problem = NonlinearPDE_SNESProblem(F, J, U, bcs, P=P)
    snes.setFunction(problem.F_mono, Fvec2)
    snes.setJacobian(problem.J_mono, J=Jmat2, P=Pmat2)

    U.sub(0).interpolate(initial_guess_u)
    U.sub(1).interpolate(initial_guess_p)

    x2 = create_vector(F)
    x2.array = U.vector.array_r

    snes.solve(None, x2)

    assert snes.getConvergedReason() > 0
    assert Jmat2.norm() == pytest.approx(Jmat0.norm(), 1.0e-12)
    assert Fvec2.norm() == pytest.approx(Fvec0.norm(), 1.0e-12)
    assert x2.norm() == pytest.approx(x0.norm(), 1.0e-12)
