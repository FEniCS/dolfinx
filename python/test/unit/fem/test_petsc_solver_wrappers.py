# Copyright (C) 2024 Jørgen S. Dokken
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for high-level wrapper around PETSc for linear and non-linear problems"""

from mpi4py import MPI

import numpy as np
import pytest

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
        if sys.hasExternalPackage("mumps"):
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
