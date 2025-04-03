# Copyright (C) 2024-2025 Jørgen S. Dokken
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for high-level wrapper around PETSc for linear and non-linear problems"""

from mpi4py import MPI

import numpy as np
import pytest

import basix.ufl
import dolfinx
import ufl


@pytest.mark.petsc4py
class TestPETScSolverWrappers:
    @pytest.mark.parametrize(
        "mode", [dolfinx.mesh.GhostMode.none, dolfinx.mesh.GhostMode.shared_facet]
    )
    def test_compare_solvers(self, mode):
        """Test that the wrapper for Linear problem and NonlinearProblem give the same result"""
        from petsc4py import PETSc

        import dolfinx.fem.petsc
        import dolfinx.nls.petsc

        msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 12, 12, ghost_mode=mode)
        V = dolfinx.fem.functionspace(msh, ("Lagrange", 1))
        uh = dolfinx.fem.Function(V)
        v = ufl.TestFunction(V)
        x = ufl.SpatialCoordinate(msh)
        f = x[0] * ufl.sin(x[1])
        F = ufl.inner(uh, v) * ufl.dx - ufl.inner(f, v) * ufl.dx
        u = ufl.TrialFunction(V)
        a = ufl.replace(F, {uh: u})

        sys = PETSc.Sys()
        if MPI.COMM_WORLD.size == 1:
            factor_type = "petsc"
        elif sys.hasExternalPackage("mumps"):
            factor_type = "mumps"
        elif sys.hasExternalPackage("superlu_dist"):
            factor_type = "superlu_dist"
        else:
            pytest.skip("No external solvers available in parallel")

        petsc_options = {
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": factor_type,
        }
        linear_problem = dolfinx.fem.petsc.LinearProblem(
            ufl.lhs(a), ufl.rhs(a), petsc_options=petsc_options
        )
        u_lin = linear_problem.solve()

        nonlinear_problem = dolfinx.fem.petsc.NonlinearProblem(F, uh)

        solver = dolfinx.nls.petsc.NewtonSolver(msh.comm, nonlinear_problem)
        ksp = solver.krylov_solver
        ksp.setType("preonly")
        ksp.getPC().setType("lu")
        ksp.getPC().setFactorSolverType(factor_type)

        eps = 100 * np.finfo(dolfinx.default_scalar_type).eps

        solver.atol = eps
        solver.rtol = eps
        solver.solve(uh)
        assert np.allclose(u_lin.x.array, uh.x.array, atol=eps, rtol=eps)

    @pytest.mark.parametrize(
        "mode", [dolfinx.mesh.GhostMode.none, dolfinx.mesh.GhostMode.shared_facet]
    )
    @pytest.mark.parametrize("kind", [None, "mpi", "nest"])
    def test_mixed_system(self, mode, kind):
        from petsc4py import PETSc

        msh = dolfinx.mesh.create_unit_square(
            MPI.COMM_WORLD, 12, 12, ghost_mode=mode, dtype=PETSc.RealType
        )

        def top_bc(x):
            return np.isclose(x[1], 1.0)

        msh.topology.create_connectivity(msh.topology.dim - 1, msh.topology.dim)
        bndry_facets = dolfinx.mesh.locate_entities_boundary(msh, msh.topology.dim - 1, top_bc)

        el_0 = basix.ufl.element("Lagrange", msh.basix_cell(), 1, dtype=PETSc.RealType)
        el_1 = basix.ufl.element("Lagrange", msh.basix_cell(), 2, dtype=PETSc.RealType)

        if kind is None:
            me = basix.ufl.mixed_element([el_0, el_1])
            W = dolfinx.fem.functionspace(msh, me)
            V, _ = W.sub(0).collapse()
            Q, _ = W.sub(1).collapse()
        else:
            V = dolfinx.fem.functionspace(msh, el_0)
            Q = dolfinx.fem.functionspace(msh, el_1)
            W = ufl.MixedFunctionSpace(V, Q)

        u, p = ufl.TrialFunctions(W)
        v, q = ufl.TestFunctions(W)

        a00 = ufl.inner(u, v) * ufl.dx
        a11 = ufl.inner(p, q) * ufl.dx
        x = ufl.SpatialCoordinate(msh)
        f = x[0] + 3 * x[1]
        g = -(x[1] ** 2) + x[0]
        L0 = ufl.inner(f, v) * ufl.dx
        L1 = ufl.inner(g, q) * ufl.dx

        f_expr = dolfinx.fem.Expression(f, V.element.interpolation_points)
        g_expr = dolfinx.fem.Expression(g, Q.element.interpolation_points)
        u_bc = dolfinx.fem.Function(V)
        u_bc.interpolate(f_expr)
        p_bc = dolfinx.fem.Function(Q)
        p_bc.interpolate(g_expr)

        if kind is None:
            a = a00 + a11
            L = L0 + L1
            dofs_V = dolfinx.fem.locate_dofs_topological(
                (W.sub(0), V), msh.topology.dim - 1, bndry_facets
            )
            dofs_Q = dolfinx.fem.locate_dofs_topological(
                (W.sub(1), Q), msh.topology.dim - 1, bndry_facets
            )
            bcs = [
                dolfinx.fem.dirichletbc(u_bc, dofs_V, W.sub(0)),
                dolfinx.fem.dirichletbc(p_bc, dofs_Q, W.sub(1)),
            ]
        else:
            a = [[a00, None], [None, a11]]
            L = [L0, L1]
            dofs_V = dolfinx.fem.locate_dofs_topological(V, msh.topology.dim - 1, bndry_facets)
            dofs_Q = dolfinx.fem.locate_dofs_topological(Q, msh.topology.dim - 1, bndry_facets)
            bcs = [
                dolfinx.fem.dirichletbc(u_bc, dofs_V),
                dolfinx.fem.dirichletbc(p_bc, dofs_Q),
            ]

        petsc_options = {
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "ksp_error_if_not_converged": True,
        }

        problem = dolfinx.fem.petsc.LinearProblem(
            a, L, bcs=bcs, petsc_options=petsc_options, kind=kind
        )
        wh = problem.solve()
        if kind is None:
            uh, ph = wh.split()
        else:
            uh, ph = wh
        error_uh = dolfinx.fem.form(ufl.inner(uh - f, uh - f) * ufl.dx)
        error_ph = dolfinx.fem.form(ufl.inner(ph - g, ph - g) * ufl.dx)
        local_uh_L2 = dolfinx.fem.assemble_scalar(error_uh)
        local_ph_L2 = dolfinx.fem.assemble_scalar(error_ph)
        global_uh_L2 = np.sqrt(msh.comm.allreduce(local_uh_L2, op=MPI.SUM))
        global_ph_L2 = np.sqrt(msh.comm.allreduce(local_ph_L2, op=MPI.SUM))
        tol = 500 * np.finfo(dolfinx.default_scalar_type).eps
        assert global_uh_L2 < tol and global_ph_L2 < tol
