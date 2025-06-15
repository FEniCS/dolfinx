# Copyright (C) 2019-2025 Nathan Sime, Garth N. Wells, Jørgen S. Dokken
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for assembly."""

import math
from functools import partial

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

        import dolfinx.fem.petsc
        import dolfinx.nls.petsc

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
            """Blocked version

            Illustrates how to use high-level class and then drop down to SNES
            for options and solve.
            """
            u.interpolate(initial_guess_u)
            p.interpolate(initial_guess_p)

            problem = dolfinx.fem.petsc.NonlinearProblem(
                F, [u, p], J=J, bcs=bcs, mat_kind="mpi", vec_kind="mpi"
            )

            petsc_options = {"snes_rtol": 1.0e-15, "snes_max_it": 10, "snes_monitor": None}

            snes = problem.solver
            opts = PETSc.Options()
            opts.prefixPush(snes.getOptionsPrefix())
            for k, v in petsc_options.items():
                opts[k] = v
            opts.prefixPop()
            snes.setFromOptions()
            for k in petsc_options.keys():
                del opts[k]

            x = problem.x
            dolfinx.fem.petsc.assign([u, p], x)
            snes.solve(None, x)

            assert snes.getConvergedReason() > 0
            assert snes.getKSP().getConvergedReason() > 0

            # NOTE: snes.solve does not assign x back into [u, p]
            # automatically.
            dolfinx.fem.petsc.assign(x, [u, p])
            xnorm = x.norm()

            return xnorm

        def nested_solve():
            """Nested version

            Illustrates how to work directly with the SNES object (no NonlinearProblem).
            """
            u.interpolate(initial_guess_u)
            p.interpolate(initial_guess_p)
            snes = PETSc.SNES().create(mesh.comm)
            residual = dolfinx.fem.form(F)
            jacobian = dolfinx.fem.form(J)
            A = dolfinx.fem.petsc.create_matrix(jacobian, "nest")
            b = dolfinx.fem.petsc.create_vector(residual, "nest")
            x = dolfinx.fem.petsc.create_vector(residual, "nest")
            snes.setFunction(
                partial(dolfinx.fem.petsc.assemble_residual, [u, p], residual, jacobian, bcs), b
            )
            snes.setJacobian(
                partial(dolfinx.fem.petsc.assemble_jacobian, [u, p], jacobian, None, bcs), A, None
            )

            nested_IS = snes.getJacobian()[0].getNestISs()
            snes.getKSP().setType("gmres")
            snes.getKSP().setTolerances(rtol=1e-12)
            snes.getKSP().getPC().setType("fieldsplit")
            snes.getKSP().getPC().setFieldSplitIS(["u", nested_IS[0][0]], ["p", nested_IS[1][1]])
            dolfinx.fem.petsc.assign([u, p], x)
            snes.solve(None, x)
            dolfinx.fem.petsc.assign(x, [u, p])
            assert snes.getConvergedReason() > 0
            assert snes.getKSP().getConvergedReason() > 0
            assert snes.getConvergedReason() > 0
            xnorm = x.norm()
            snes.destroy()
            return xnorm

        def monolithic_solve():
            """Monolithic version

            Uses high level NonlinearProblem class only.
            """
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

            petsc_options = {"snes_rtol": 1.0e-15, "snes_max_it": 10}
            U.sub(0).interpolate(initial_guess_u)
            U.sub(1).interpolate(initial_guess_p)

            problem = dolfinx.fem.petsc.NonlinearProblem(
                F,
                U,
                J=J,
                bcs=bcs,
                petsc_options=petsc_options,
            )

            x, converged_reason, _ = problem.solve()
            assert converged_reason > 0
            xnorm = x.norm()
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
        import dolfinx.fem.petsc
        import dolfinx.nls.petsc

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

        def blocked():
            """Blocked and monolithic"""
            u.interpolate(initial_guess_u)
            p.interpolate(initial_guess_p)
            petsc_options = {
                "snes_rtol": 1.0e-15,
                "snes_max_it": 10,
                "snes_monitor": None,
                "ksp_type": "minres",
            }
            problem = dolfinx.fem.petsc.NonlinearProblem(
                F,
                [u, p],
                bcs=bcs,
                P=P,
                petsc_options=petsc_options,
                mat_kind="mpi",
                vec_kind="mpi",
            )
            x, converged_reason, _ = problem.solve()
            assert converged_reason > 0
            Jnorm = problem.solver.getJacobian()[0].norm()
            Fnorm = problem.solver.getFunction()[0].norm()
            xnorm = x.norm()
            return Jnorm, Fnorm, xnorm

        def nested():
            """Blocked and nested

            Shows how to setup some SNES options programatically.
            """
            u.interpolate(initial_guess_u)
            p.interpolate(initial_guess_p)

            problem = dolfinx.fem.petsc.NonlinearProblem(
                F, [u, p], J=J, bcs=bcs, mat_kind="nest", vec_kind="nest", P=P
            )
            nested_IS = problem.solver.getJacobian()[0].getNestISs()
            problem.solver.setTolerances(rtol=1.0e-15, max_it=20)
            problem.solver.getKSP().setType("minres")
            problem.solver.getKSP().setTolerances(rtol=1e-8)
            problem.solver.getKSP().getPC().setType("fieldsplit")
            problem.solver.getKSP().getPC().setFieldSplitIS(
                ["u", nested_IS[0][0]], ["p", nested_IS[1][1]]
            )

            x, converged_reason, _ = problem.solve()
            assert converged_reason > 0
            xnorm = x.norm()
            Jnorm = nest_matrix_norm(problem.solver.getJacobian()[0])
            Fnorm = problem.solver.getFunction()[0].norm()
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

            bdofsW0_P2_0 = locate_dofs_topological((W.sub(0), P2), facetdim, bndry_facets0)
            bdofsW0_P2_1 = locate_dofs_topological((W.sub(0), P2), facetdim, bndry_facets1)
            bcs = [
                dirichletbc(u_bc_0, bdofsW0_P2_0, W.sub(0)),
                dirichletbc(u_bc_1, bdofsW0_P2_1, W.sub(0)),
            ]

            U.sub(0).interpolate(initial_guess_u)
            U.sub(1).interpolate(initial_guess_p)

            petsc_options = {
                "snes_rtol": 1.0e-15,
                "snes_max_it": 20,
                "ksp_type": "minres",
                "snes_monitor": None,
            }
            problem = dolfinx.fem.petsc.NonlinearProblem(
                F,
                U,
                J=J,
                bcs=bcs,
                P=P,
                petsc_options=petsc_options,
            )
            x, converged_reason, _ = problem.solve()
            assert converged_reason > 0
            xnorm = x.norm()
            Jnorm = problem.solver.getJacobian()[0].norm()
            Fnorm = problem.solver.getFunction()[0].norm()
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
