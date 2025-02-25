# Copyright (C) 2019-2025 Nathan Sime, Jørgen S. Dokken
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for assembly."""

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

        from dolfinx.cpp.la.petsc import scatter_local_vectors
        from dolfinx.fem.petsc import (
            apply_lifting,
            apply_lifting_nest,
            assemble_matrix,
            assemble_matrix_block,
            assemble_matrix_nest,
            assemble_vector,
            assemble_vector_block,
            assemble_vector_nest,
            create_vector_block,
            create_vector_nest,
            set_bc,
            set_bc_nest,
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
            x = create_vector_block(L_block)
            scatter_local_vectors(
                x,
                [u.x.petsc_vec.array_r, p.x.petsc_vec.array_r],
                [
                    (u.function_space.dofmap.index_map, u.function_space.dofmap.index_map_bs),
                    (p.function_space.dofmap.index_map, p.function_space.dofmap.index_map_bs),
                ],
            )
            x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

            # Ghosts are updated inside assemble_vector_block
            A = assemble_matrix_block(a_block, bcs=[bc])
            b = assemble_vector_block(L_block, a_block, bcs=[bc], x0=x, alpha=-1.0)
            A.assemble()
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
            x = create_vector_nest(L_block)
            for x1_soln_pair in zip(x.getNestSubVecs(), (u, p)):
                x1_sub, soln_sub = x1_soln_pair
                soln_sub.x.petsc_vec.ghostUpdate(
                    addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
                )
                soln_sub.x.petsc_vec.copy(result=x1_sub)
                x1_sub.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

            A = assemble_matrix_nest(a_block, bcs=[bc])
            b = assemble_vector_nest(L_block)
            apply_lifting_nest(b, a_block, bcs=[bc], x0=x, alpha=-1.0)
            for b_sub in b.getNestSubVecs():
                b_sub.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            bcs0 = bcs_by_block([L.function_spaces[0] for L in L_block], [bc])

            set_bc_nest(b, bcs0, x, alpha=-1.0)
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
            """Blocked version"""
            u.interpolate(initial_guess_u)
            p.interpolate(initial_guess_p)

            snes_options = {"snes_rtol": 1.0e-15, "snes_max_it": 10, "snes_monitor": None}
            snes, x = dolfinx.nls.petsc.create_snes_solver(
                F, [u, p], J=J, bcs=bcs, assembly_type=dolfinx.fem.AssemblyType.block
            )
            opts = PETSc.Options()
            for k, v in snes_options.items():
                opts[k] = v
            snes.setFromOptions()
            for k, _ in snes_options.items():
                del opts[k]

            snes.solve(None, x)
            assert snes.getConvergedReason() > 0
            assert snes.getKSP().getConvergedReason() > 0
            dolfinx.nls.petsc.copy_block_vec_to_functions([u, p], x)
            xnorm = x.norm()
            x.destroy()
            return xnorm

        def nested_solve():
            """Nested version"""
            u.interpolate(initial_guess_u)
            p.interpolate(initial_guess_p)
            solver = dolfinx.nls.petsc.SNESSolver(
                F, [u, p], J=J, bcs=bcs, assembly_type=dolfinx.fem.AssemblyType.nest
            )
            nested_IS = solver.snes.getJacobian()[0].getNestISs()
            solver.snes.getKSP().setType("gmres")
            solver.snes.getKSP().setTolerances(rtol=1e-12)
            solver.snes.getKSP().getPC().setType("fieldsplit")
            solver.snes.getKSP().getPC().setFieldSplitIS(
                ["u", nested_IS[0][0]], ["p", nested_IS[1][1]]
            )
            x, converged_reason, _ = solver.solve()
            solver.copy_vec_to_function([u, p], x)
            assert solver.snes.getConvergedReason() > 0
            assert solver.snes.getKSP().getConvergedReason() > 0
            assert converged_reason > 0
            xnorm = x.norm()
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

            snes_options = {"snes_rtol": 1.0e-15, "snes_max_it": 10}
            U.sub(0).interpolate(initial_guess_u)
            U.sub(1).interpolate(initial_guess_p)

            solver = dolfinx.nls.petsc.SNESSolver(
                F,
                U,
                J=J,
                bcs=bcs,
                assembly_type=dolfinx.fem.AssemblyType.default,
                snes_options=snes_options,
            )

            x, converged_reason, _ = solver.solve()
            assert converged_reason > 0
            solver.copy_vec_to_function(U, x)
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

            snes_options = {
                "snes_rtol": 1.0e-15,
                "snes_max_it": 10,
                "snes_monitor": None,
                "ksp_type": "minres",
            }
            solver = dolfinx.nls.petsc.SNESSolver(
                F,
                [u, p],
                bcs=bcs,
                P=P,
                assembly_type=dolfinx.fem.AssemblyType.block,
                snes_options=snes_options,
            )
            x, converged_reason, _ = solver.solve()
            assert converged_reason > 0
            solver.copy_vec_to_function([u, p], x)
            Jnorm = solver.snes.getJacobian()[0].norm()
            Fnorm = solver.snes.getFunction()[0].norm()
            xnorm = x.norm()
            return Jnorm, Fnorm, xnorm

        def nested():
            """Blocked and nested"""
            u.interpolate(initial_guess_u)
            p.interpolate(initial_guess_p)

            solver = dolfinx.nls.petsc.SNESSolver(
                F, [u, p], J=J, bcs=bcs, assembly_type=dolfinx.fem.AssemblyType.nest, P=P
            )
            nested_IS = solver.snes.getJacobian()[0].getNestISs()
            solver.snes.setTolerances(rtol=1.0e-15, max_it=20)
            solver.snes.getKSP().setType("minres")
            solver.snes.getKSP().setTolerances(rtol=1e-8)
            solver.snes.getKSP().getPC().setType("fieldsplit")
            solver.snes.getKSP().getPC().setFieldSplitIS(
                ["u", nested_IS[0][0]], ["p", nested_IS[1][1]]
            )

            x, converged_reason, _ = solver.solve()
            assert converged_reason > 0
            solver.copy_vec_to_function([u, p], x)
            xnorm = x.norm()
            Jnorm = nest_matrix_norm(solver.snes.getJacobian()[0])
            Fnorm = solver.snes.getFunction()[0].norm()
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

            snes_options = {
                "snes_rtol": 1.0e-15,
                "snes_max_it": 20,
                "ksp_type": "minres",
                "snes_monitor": None,
            }
            solver = dolfinx.nls.petsc.SNESSolver(
                F,
                U,
                J=J,
                bcs=bcs,
                P=P,
                assembly_type=dolfinx.fem.AssemblyType.default,
                snes_options=snes_options,
            )
            x, converged_reason, _ = solver.solve()
            assert converged_reason > 0
            solver.copy_vec_to_function(U, x)
            xnorm = x.norm()
            Jnorm = solver.snes.getJacobian()[0].norm()
            Fnorm = solver.snes.getFunction()[0].norm()
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
