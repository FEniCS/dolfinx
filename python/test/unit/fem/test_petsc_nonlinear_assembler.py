# Copyright (C) 2019 Nathan Sime
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for assembly."""

import itertools
import math

from mpi4py import MPI

import numpy as np
import pytest

import ufl
from basix.ufl import element, mixed_element
from dolfinx import default_real_type
from dolfinx.fem import (
    Function,
    bcs_by_block,
    dirichletbc,
    extract_function_spaces,
    form,
    functionspace,
    locate_dofs_topological,
)
from dolfinx.mesh import GhostMode, create_unit_cube, create_unit_square, locate_entities_boundary
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


class NonlinearPDE_SNESProblem:
    def __init__(self, F, J, soln_vars, bcs, P=None):
        self.L = F
        self.a = J
        self.a_precon = P
        self.bcs = bcs
        self.soln_vars = soln_vars

    def F_mono(self, snes, x, F):
        from petsc4py import PETSc

        from dolfinx.fem.petsc import apply_lifting, assemble_vector, assign, set_bc

        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        assign(x, self.soln_vars)

        with F.localForm() as f_local:
            f_local.set(0.0)
        assemble_vector(F, self.L)
        apply_lifting(F, [self.a], bcs=[self.bcs], x0=[x], alpha=-1.0)
        F.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        set_bc(F, self.bcs, x, -1.0)

    def J_mono(self, snes, x, J, P):
        from dolfinx.fem.petsc import assemble_matrix

        J.zeroEntries()
        assemble_matrix(J, self.a, bcs=self.bcs, diag=1.0)
        J.assemble()
        if self.a_precon is not None:
            P.zeroEntries()
            assemble_matrix(P, self.a_precon, bcs=self.bcs, diag=1.0)
            P.assemble()

    def F_block(self, snes, x, F):
        from petsc4py import PETSc

        from dolfinx.fem.petsc import apply_lifting, assemble_vector, assign, set_bc

        assert x.getType() != "nest"
        assert F.getType() != "nest"

        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        assign(x, self.soln_vars)

        with F.localForm() as f_local:
            f_local.set(0.0)

        maps = [
            (
                form.function_spaces[0].dofmaps(0).index_map,
                form.function_spaces[0].dofmaps(0).index_map_bs,
            )
            for form in self.L
        ]
        off_owned = tuple(
            itertools.accumulate(maps, lambda off, m: off + m[0].size_local * m[1], initial=0)
        )
        off_ghost = tuple(
            itertools.accumulate(
                maps, lambda off, m: off + m[0].num_ghosts * m[1], initial=off_owned[-1]
            )
        )

        with F.localForm() as f_local:
            f_local.set(0.0)

        F.setAttr("_blocks", (off_owned, off_ghost))
        x.setAttr("_blocks", (off_owned, off_ghost))

        assemble_vector(F, self.L)
        bcs1 = bcs_by_block(extract_function_spaces(self.a, 1), bcs=self.bcs)
        apply_lifting(F, self.a, bcs=bcs1, x0=x, alpha=-1.0)
        F.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        bcs0 = bcs_by_block(extract_function_spaces(self.L), self.bcs)
        set_bc(F, bcs0, x0=x, alpha=-1)

    def J_block(self, snes, x, J, P):
        from dolfinx.fem.petsc import assemble_matrix

        assert x.getType() != "nest" and J.getType() != "nest" and P.getType() != "nest"
        J.zeroEntries()
        assemble_matrix(J, self.a, bcs=self.bcs, diag=1.0)
        J.assemble()
        if self.a_precon is not None:
            P.zeroEntries()
            assemble_matrix(P, self.a_precon, bcs=self.bcs, diag=1.0)
            P.assemble()

    def F_nest(self, snes, x, F):
        from petsc4py import PETSc

        from dolfinx.fem.petsc import apply_lifting, assemble_vector, assign, set_bc

        assert x.getType() == "nest" and F.getType() == "nest"

        # Update solution
        for x_sub in x.getNestSubVecs():
            x_sub.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        assign(x, self.soln_vars)

        # Assemble
        x = x.getNestSubVecs()
        bcs1 = bcs_by_block(extract_function_spaces(self.a, 1), self.bcs)
        for L, F_sub, a in zip(self.L, F.getNestSubVecs(), self.a):
            with F_sub.localForm() as F_sub_local:
                F_sub_local.set(0.0)
            assemble_vector(F_sub, L)
            apply_lifting(F_sub, a, bcs=bcs1, x0=x, alpha=-1.0)
            F_sub.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

        # Set bc value in RHS
        bcs0 = bcs_by_block(extract_function_spaces(self.L), self.bcs)
        for F_sub, bc, x_sub in zip(F.getNestSubVecs(), bcs0, x):
            set_bc(F_sub, bc, x_sub, -1.0)

        # Must assemble F here in the case of nest matrices
        F.assemble()

    def J_nest(self, snes, x, J, P):
        from dolfinx.fem.petsc import assemble_matrix

        assert J.getType() == "nest" and P.getType() == "nest"
        J.zeroEntries()
        assemble_matrix(J, self.a, bcs=self.bcs, diag=1.0)
        J.assemble()
        if self.a_precon is not None:
            P.zeroEntries()
            assemble_matrix(P, self.a_precon, bcs=self.bcs, diag=1.0)
            P.assemble()


@pytest.mark.petsc4py
class TestNLSPETSc:
    def test_matrix_assembly_block_nl(self):
        """Test assembly of block matrices and vectors into (a) monolithic
        blocked structures, PETSc Nest structures, and monolithic structures
        in the nonlinear setting."""
        from petsc4py import PETSc

        from dolfinx.fem.petsc import (
            apply_lifting,
            assemble_matrix,
            assemble_vector,
            assign,
            create_vector,
            set_bc,
        )

        mesh = create_unit_square(MPI.COMM_WORLD, 4, 8)
        p0, p1 = 1, 2
        P0 = element("Lagrange", mesh.basix_cell(), p0, dtype=default_real_type)
        P1 = element("Lagrange", mesh.basix_cell(), p1, dtype=default_real_type)
        V0 = functionspace(mesh, P0)
        V1 = functionspace(mesh, P1)

        def initial_guess_u(x):
            return np.sin(x[0]) * np.sin(x[1])

        def initial_guess_p(x):
            return -(x[0] ** 2) - x[1] ** 3

        def bc_value(x):
            return np.cos(x[0]) * np.cos(x[1])

        facetdim = mesh.topology.dim - 1
        bndry_facets = locate_entities_boundary(
            mesh, facetdim, lambda x: np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0)
        )

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

        a_block = form(
            [
                [derivative(F0, u, du), derivative(F0, p, dp)],
                [derivative(F1, u, du), derivative(F1, p, dp)],
            ]
        )
        L_block = form([F0, F1])

        def blocked():
            """Monolithic blocked"""
            x = create_vector(L_block, kind="mpi")

            assign((u, p), x)

            # Ghosts are updated inside assemble_vector_block
            A = assemble_matrix(a_block, bcs=[bc])
            A.assemble()

            b = assemble_vector(L_block, kind="mpi")
            bcs1 = bcs_by_block(extract_function_spaces(a_block, 1), bcs=[bc])
            apply_lifting(b, a_block, bcs=bcs1, x0=x, alpha=-1.0)
            b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            bcs0 = bcs_by_block(extract_function_spaces(L_block), [bc])
            set_bc(b, bcs0, x0=x, alpha=-1)

            assert A.getType() != "nest"
            Anorm = A.norm()
            bnorm = b.norm()
            A.destroy()
            b.destroy()
            x.destroy()
            return Anorm, bnorm

        # Nested (MatNest)
        def nested():
            """Nested (MatNest)"""
            x = create_vector(L_block, kind=PETSc.Vec.Type.NEST)

            assign((u, p), x)

            A = assemble_matrix(a_block, bcs=[bc], kind="nest")
            b = assemble_vector(L_block, kind="nest")
            bcs1 = bcs_by_block(extract_function_spaces(a_block, 1), bcs=[bc])
            apply_lifting(b, a_block, bcs=bcs1, x0=x, alpha=-1.0)
            for b_sub in b.getNestSubVecs():
                b_sub.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            bcs0 = bcs_by_block([L.function_spaces[0] for L in L_block], [bc])

            set_bc(b, bcs0, x, alpha=-1.0)
            A.assemble()
            assert A.getType() == "nest"
            Anorm = nest_matrix_norm(A)
            bnorm = b.norm()
            A.destroy()
            b.destroy()
            x.destroy()
            return Anorm, bnorm

        def monolithic():
            """Monolithic version"""
            E = mixed_element([P0, P1])
            W = functionspace(mesh, E)
            dU = ufl.TrialFunction(W)
            U = Function(W)
            u0, u1 = ufl.split(U)
            v0, v1 = ufl.TestFunctions(W)

            U.sub(0).interpolate(initial_guess_u)
            U.sub(1).interpolate(initial_guess_p)

            F = (
                inner(u0, v0) * dx
                + inner(u1, v0) * dx
                + inner(u0, v1) * dx
                + inner(u1, v1) * dx
                - inner(f, v0) * ufl.dx
                - inner(g, v1) * dx
            )
            J = derivative(F, U, dU)
            F, J = form(F), form(J)

            bdofsW_V1 = locate_dofs_topological((W.sub(1), V1), facetdim, bndry_facets)
            bc = dirichletbc(u_bc, bdofsW_V1, W.sub(1))
            A = assemble_matrix(J, bcs=[bc])
            A.assemble()
            b = assemble_vector(F)
            apply_lifting(b, [J], bcs=[[bc]], x0=[U.x.petsc_vec], alpha=-1.0)
            b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            set_bc(b, [bc], x0=U.x.petsc_vec, alpha=-1.0)
            assert A.getType() != "nest"
            Anorm = A.norm()
            bnorm = b.norm()
            A.destroy()
            b.destroy()
            return Anorm, bnorm

        Anorm0, bnorm0 = blocked()
        Anorm1, bnorm1 = nested()
        assert Anorm1 == pytest.approx(Anorm0, 1.0e-6)
        assert bnorm1 == pytest.approx(bnorm0, 1.0e-6)

        Anorm2, bnorm2 = monolithic()
        assert Anorm2 == pytest.approx(Anorm0, 1.0e-5)
        assert bnorm2 == pytest.approx(bnorm0, 1.0e-6)

    def test_assembly_solve_block_nl(self):
        """Solve a two-field nonlinear diffusion like problem with block
        matrix approaches and test that solution is the same."""
        from petsc4py import PETSc

        from dolfinx.fem.petsc import (
            assign,
            create_matrix,
            create_vector,
        )

        mesh = create_unit_square(MPI.COMM_WORLD, 12, 11)
        p = 1
        P = element("Lagrange", mesh.basix_cell(), p, dtype=default_real_type)
        V0 = functionspace(mesh, P)
        V1 = V0.clone()

        def bc_val_0(x):
            return x[0] ** 2 + x[1] ** 2

        def bc_val_1(x):
            return np.sin(x[0]) * np.cos(x[1])

        def initial_guess_u(x):
            return np.sin(x[0]) * np.sin(x[1])

        def initial_guess_p(x):
            return -(x[0] ** 2) - x[1] ** 3

        facetdim = mesh.topology.dim - 1
        bndry_facets = locate_entities_boundary(
            mesh, facetdim, lambda x: np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0)
        )

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

        f, g = 1.0, -3.0
        F = [
            inner((u**2 + 1) * ufl.grad(u), ufl.grad(v)) * dx - inner(f, v) * dx,
            inner((p**2 + 1) * ufl.grad(p), ufl.grad(q)) * dx - inner(g, q) * dx,
        ]
        J = [
            [derivative(F[0], u, du), derivative(F[0], p, dp)],
            [derivative(F[1], u, du), derivative(F[1], p, dp)],
        ]
        F, J = form(F), form(J)

        def blocked_solve():
            """Blocked version"""
            Jmat = create_matrix(J)
            Fvec = create_vector(F, kind="mpi")
            snes = PETSc.SNES().create(MPI.COMM_WORLD)
            snes.setTolerances(rtol=1.0e-15, max_it=10)
            problem = NonlinearPDE_SNESProblem(F, J, [u, p], bcs)
            snes.setFunction(problem.F_block, Fvec)
            snes.setJacobian(problem.J_block, J=Jmat, P=None)

            u.interpolate(initial_guess_u)
            p.interpolate(initial_guess_p)

            x = create_vector(F, kind="mpi")

            assign((u, p), x)

            snes.solve(None, x)
            assert snes.getKSP().getConvergedReason() > 0
            assert snes.getConvergedReason() > 0
            xnorm = x.norm()
            snes.destroy()
            Jmat.destroy()
            Fvec.destroy()
            x.destroy()
            return xnorm

        def nested_solve():
            """Nested version"""
            Jmat = create_matrix(J, kind=[["baij", "aij"], ["aij", "baij"]])
            assert Jmat.getType() == "nest"
            Fvec = create_vector(F, kind="nest")
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
            x = create_vector(F, kind=PETSc.Vec.Type.NEST)
            assert x.getType() == "nest"

            assign((u, p), x)

            snes.solve(None, x)
            assert snes.getKSP().getConvergedReason() > 0
            assert snes.getConvergedReason() > 0
            xnorm = x.norm()
            snes.destroy()
            Jmat.destroy()
            Fvec.destroy()
            x.destroy()
            return xnorm

        def monolithic_solve():
            """Monolithic version"""
            E = mixed_element([P, P])
            W = functionspace(mesh, E)
            U = Function(W)
            dU = ufl.TrialFunction(W)
            u0, u1 = ufl.split(U)
            v0, v1 = ufl.TestFunctions(W)

            F = (
                inner((u0**2 + 1) * ufl.grad(u0), ufl.grad(v0)) * dx
                + inner((u1**2 + 1) * ufl.grad(u1), ufl.grad(v1)) * dx
                - inner(f, v0) * ufl.dx
                - inner(g, v1) * dx
            )
            J = derivative(F, U, dU)
            F, J = form(F), form(J)

            u0_bc = Function(V0)
            u0_bc.interpolate(bc_val_0)
            u1_bc = Function(V1)
            u1_bc.interpolate(bc_val_1)
            bdofsW0_V0 = locate_dofs_topological((W.sub(0), V0), facetdim, bndry_facets)
            bdofsW1_V1 = locate_dofs_topological((W.sub(1), V1), facetdim, bndry_facets)
            bcs = [
                dirichletbc(u0_bc, bdofsW0_V0, W.sub(0)),
                dirichletbc(u1_bc, bdofsW1_V1, W.sub(1)),
            ]

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
            assign(U, x)

            snes.solve(None, x)
            assert snes.getKSP().getConvergedReason() > 0
            assert snes.getConvergedReason() > 0
            xnorm = x.norm()
            snes.destroy()
            Jmat.destroy()
            Fvec.destroy()
            x.destroy()
            return xnorm

        norm0 = blocked_solve()
        norm2 = monolithic_solve()
        # FIXME: PETSc nested solver mis-behaves in parallel an single
        # precision. Investigate further.
        if not (
            (PETSc.ScalarType == np.float32 or PETSc.ScalarType == np.complex64)
            and mesh.comm.size > 1
        ):
            norm1 = nested_solve()
            assert norm1 == pytest.approx(norm0, 1.0e-6)
        assert norm2 == pytest.approx(norm0, 1.0e-6)

    @pytest.mark.parametrize(
        "mesh",
        [
            create_unit_square(MPI.COMM_WORLD, 12, 11, ghost_mode=GhostMode.none),
            create_unit_square(MPI.COMM_WORLD, 12, 11, ghost_mode=GhostMode.shared_facet),
            create_unit_cube(MPI.COMM_WORLD, 3, 5, 4, ghost_mode=GhostMode.none),
            create_unit_cube(MPI.COMM_WORLD, 3, 5, 4, ghost_mode=GhostMode.shared_facet),
        ],
    )
    def test_assembly_solve_taylor_hood_nl(self, mesh):
        """Assemble Stokes problem with Taylor-Hood elements and solve."""
        from petsc4py import PETSc

        from dolfinx.fem.petsc import (
            assign,
            create_matrix,
            create_vector,
        )

        gdim = mesh.geometry.dim
        P2 = functionspace(mesh, ("Lagrange", 2, (gdim,)))
        P1 = functionspace(mesh, ("Lagrange", 1))

        def boundary0(x):
            """Define boundary x = 0"""
            return np.isclose(x[0], 0.0)

        def boundary1(x):
            """Define boundary x = 1"""
            return np.isclose(x[0], 1.0)

        def initial_guess_u(x):
            u_init = np.vstack((np.sin(x[0]) * np.sin(x[1]), np.cos(x[0]) * np.cos(x[1])))
            if gdim == 3:
                u_init = np.vstack((u_init, np.cos(x[2])))
            return u_init

        def initial_guess_p(x):
            return -(x[0] ** 2) - x[1] ** 3

        u_bc_0 = Function(P2)
        u_bc_0.interpolate(lambda x: np.vstack(tuple(x[j] + float(j) for j in range(gdim))))

        u_bc_1 = Function(P2)
        u_bc_1.interpolate(lambda x: np.vstack(tuple(np.sin(x[j]) for j in range(gdim))))

        facetdim = mesh.topology.dim - 1
        bndry_facets0 = locate_entities_boundary(mesh, facetdim, boundary0)
        bndry_facets1 = locate_entities_boundary(mesh, facetdim, boundary1)

        bdofs0 = locate_dofs_topological(P2, facetdim, bndry_facets0)
        bdofs1 = locate_dofs_topological(P2, facetdim, bndry_facets1)

        bcs = [dirichletbc(u_bc_0, bdofs0), dirichletbc(u_bc_1, bdofs1)]

        u, p = Function(P2), Function(P1)
        du, dp = ufl.TrialFunction(P2), ufl.TrialFunction(P1)
        v, q = ufl.TestFunction(P2), ufl.TestFunction(P1)

        F = [
            inner(ufl.grad(u), ufl.grad(v)) * dx + inner(p, ufl.div(v)) * dx,
            inner(ufl.div(u), q) * dx,
        ]
        J = [
            [derivative(F[0], u, du), derivative(F[0], p, dp)],
            [derivative(F[1], u, du), derivative(F[1], p, dp)],
        ]
        P = [[J[0][0], None], [None, inner(dp, q) * dx]]
        F, J, P = form(F), form(J), form(P)

        def blocked():
            """Blocked and monolithic"""
            Jmat = create_matrix(J)
            Pmat = create_matrix(P)
            Fvec = create_vector(F, kind="mpi")

            snes = PETSc.SNES().create(MPI.COMM_WORLD)
            snes.setTolerances(rtol=1.0e-15, max_it=20)
            snes.getKSP().setType("minres")

            problem = NonlinearPDE_SNESProblem(F, J, [u, p], bcs, P=P)
            snes.setFunction(problem.F_block, Fvec)
            snes.setJacobian(problem.J_block, J=Jmat, P=Pmat)

            u.interpolate(initial_guess_u)
            p.interpolate(initial_guess_p)
            x = create_vector(F, kind="mpi")

            assign((u, p), x)

            snes.solve(None, x)
            assert snes.getConvergedReason() > 0
            snes.destroy()
            Jnorm = Jmat.norm()
            Fnorm = Fvec.norm()
            xnorm = x.norm()
            Jmat.destroy()
            Fvec.destroy()
            x.destroy()
            return Jnorm, Fnorm, xnorm

        def nested():
            """Blocked and nested"""
            Jmat = create_matrix(J, kind=PETSc.Mat.Type.NEST)
            Pmat = create_matrix(P, kind=PETSc.Mat.Type.NEST)
            Fvec = create_vector(F, kind=PETSc.Vec.Type.NEST)

            snes = PETSc.SNES().create(MPI.COMM_WORLD)
            snes.setTolerances(rtol=1.0e-15, max_it=20)
            nested_IS = Jmat.getNestISs()
            snes.getKSP().setType("minres")
            snes.getKSP().setTolerances(rtol=1e-8)
            snes.getKSP().getPC().setType("fieldsplit")
            snes.getKSP().getPC().setFieldSplitIS(["u", nested_IS[0][0]], ["p", nested_IS[1][1]])

            problem = NonlinearPDE_SNESProblem(F, J, [u, p], bcs, P=P)
            snes.setFunction(problem.F_nest, Fvec)
            snes.setJacobian(problem.J_nest, J=Jmat, P=Pmat)

            u.interpolate(initial_guess_u)
            p.interpolate(initial_guess_p)
            x = create_vector(F, "nest")

            assign((u, p), x)

            x.set(0.0)
            snes.solve(None, x)
            assert snes.getConvergedReason() > 0
            snes.destroy()
            Jnorm = nest_matrix_norm(Jmat)
            Fnorm = Fvec.norm()
            xnorm = x.norm()
            Jmat.destroy(), Fvec.destroy()
            x.destroy()
            Pmat.destroy()
            return Jnorm, Fnorm, xnorm

        def monolithic():
            """Monolithic"""
            P2_el = element(
                "Lagrange",
                mesh.basix_cell(),
                2,
                shape=(mesh.geometry.dim,),
                dtype=default_real_type,
            )
            P1_el = element("Lagrange", mesh.basix_cell(), 1, dtype=default_real_type)
            TH = mixed_element([P2_el, P1_el])
            W = functionspace(mesh, TH)
            U = Function(W)
            dU = ufl.TrialFunction(W)
            u, p = ufl.split(U)
            du, dp = ufl.split(dU)
            v, q = ufl.TestFunctions(W)

            F = (
                inner(ufl.grad(u), ufl.grad(v)) * dx
                + inner(p, ufl.div(v)) * dx
                + inner(ufl.div(u), q) * dx
            )
            J = derivative(F, U, dU)
            P = inner(ufl.grad(du), ufl.grad(v)) * dx + inner(dp, q) * dx
            F, J, P = form(F), form(J), form(P)

            bdofsW0_P2_0 = locate_dofs_topological((W.sub(0), P2), facetdim, bndry_facets0)
            bdofsW0_P2_1 = locate_dofs_topological((W.sub(0), P2), facetdim, bndry_facets1)
            bcs = [
                dirichletbc(u_bc_0, bdofsW0_P2_0, W.sub(0)),
                dirichletbc(u_bc_1, bdofsW0_P2_1, W.sub(0)),
            ]

            Jmat = create_matrix(J)
            Pmat = create_matrix(P)
            Fvec = create_vector(F)

            snes = PETSc.SNES().create(MPI.COMM_WORLD)
            snes.setTolerances(rtol=1.0e-15, max_it=20)
            snes.getKSP().setType("minres")

            problem = NonlinearPDE_SNESProblem(F, J, U, bcs, P=P)
            snes.setFunction(problem.F_mono, Fvec)
            snes.setJacobian(problem.J_mono, J=Jmat, P=Pmat)

            U.sub(0).interpolate(initial_guess_u)
            U.sub(1).interpolate(initial_guess_p)

            x = create_vector(F)
            assign(U, x)

            snes.solve(None, x)
            assert snes.getConvergedReason() > 0
            snes.destroy()
            Jnorm = Jmat.norm()
            Fnorm = Fvec.norm()
            xnorm = x.norm()
            Jmat.destroy()
            Fvec.destroy()
            x.destroy()
            return Jnorm, Fnorm, xnorm

        Jnorm0, Fnorm0, xnorm0 = blocked()
        Jnorm1, Fnorm1, xnorm1 = nested()
        assert Jnorm1 == pytest.approx(Jnorm0, 1.0e-3, abs=1.0e-6)
        assert Fnorm1 == pytest.approx(Fnorm0, 1.0e-6, abs=1.0e-5)
        assert xnorm1 == pytest.approx(xnorm0, 1.0e-6, abs=1.0e-5)

        Jnorm2, Fnorm2, xnorm2 = monolithic()
        assert Jnorm2 == pytest.approx(Jnorm1, rel=1.0e-3, abs=1.0e-6)
        assert Fnorm2 == pytest.approx(Fnorm0, 1.0e-6, abs=1.0e-5)
        assert xnorm2 == pytest.approx(xnorm0, 1.0e-6, abs=1.0e-6)
