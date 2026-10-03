# Copyright (C) 2024-2026 Jørgen S. Dokken and Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for high-level wrapper around PETSc for linear and non-linear problems."""

from mpi4py import MPI

import numpy as np
import pytest

import basix.ufl
import dolfinx
import ufl


@pytest.mark.petsc4py
class TestPETScSolverWrappers:
    """Test PETSc solver wrappers for linear and nonlinear problems."""

    @pytest.mark.parametrize(
        "mode",
        [dolfinx.mesh.GhostMode.none, dolfinx.mesh.GhostMode.shared_facet],
    )
    def test_compare_solution_linear_vs_nonlinear_problem(self, mode):
        """Test that the wrapper for Linear problem and NonlinearProblem give the same result."""
        from petsc4py import PETSc

        import dolfinx.fem.petsc

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

        petsc_options_linear = {
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": factor_type,
        }
        petsc_options_prefix_linear = (
            f"test_compare_solution_linear_vs_nonlinear_problem__{mode}_linear_"
        )
        linear_problem = dolfinx.fem.petsc.LinearProblem(
            ufl.lhs(a),
            ufl.rhs(a),
            petsc_options_prefix=petsc_options_prefix_linear,
            petsc_options=petsc_options_linear,
        )
        u_lin = linear_problem.solve()
        assert linear_problem.solver.getConvergedReason() > 0

        eps = 100 * np.finfo(dolfinx.default_scalar_type).eps

        # Compare LinearProblem solution against the one obtained by NonlinearProblem
        petsc_options_nonlinear = {
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": factor_type,
            "snes_atol": eps,
            "snes_rtol": eps,
        }
        petsc_options_prefix_nonlinear = (
            f"test_compare_solution_linear_vs_nonlinear_problem__{mode}__nonlinear_"
        )
        u_nonlin = dolfinx.fem.Function(V)
        nonlinear_problem = dolfinx.fem.petsc.NonlinearProblem(
            ufl.replace(F, {uh: u_nonlin}),
            u_nonlin,
            petsc_options_prefix=petsc_options_prefix_nonlinear,
            petsc_options=petsc_options_nonlinear,
        )
        nonlinear_problem.solve()
        assert nonlinear_problem.solver.getConvergedReason() > 0

        assert np.allclose(u_lin.x.array, u_nonlin.x.array, atol=eps, rtol=eps)

        with u_lin.x.petsc_vec.localForm() as _u_lin, u_nonlin.x.petsc_vec.localForm() as _u_nonlin:
            assert np.allclose(_u_lin.array_r, _u_nonlin.array_r, atol=eps, rtol=eps)

    @pytest.mark.parametrize(
        "mode", [dolfinx.mesh.GhostMode.none, dolfinx.mesh.GhostMode.shared_facet]
    )
    @pytest.mark.parametrize("kind", [None, "mpi", "nest", [["aij", None], [None, "baij"]]])
    def test_mixed_system(self, mode, kind):
        """Test solving a mixed system using different PETSc matrix layouts."""
        from petsc4py import PETSc

        import dolfinx.fem.petsc

        if not PETSc.Sys().hasExternalPackage("mumps"):
            pytest.skip("MUMPS is required to factor this system")

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

        petsc_options_prefix = (
            f"test_mixed_system_{kind if isinstance(kind, str) else 'nest_2d_list'}_"
        )
        petsc_options = {
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "ksp_error_if_not_converged": True,
        }
        problem = dolfinx.fem.petsc.LinearProblem(
            a,
            L,
            bcs=bcs,
            kind=kind,
            petsc_options_prefix=petsc_options_prefix,
            petsc_options=petsc_options,
        )
        wh = problem.solve()
        assert problem.solver.getConvergedReason() > 0
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

    @pytest.mark.parametrize(
        "mode", [dolfinx.mesh.GhostMode.none, dolfinx.mesh.GhostMode.shared_facet]
    )
    def test_overlapping_dirichlet_bcs(self, mode):
        """Test a LinearProblem with two non-zero bcs sharing degrees-of-freedom.

        A dof constrained by more than one bc is set by the last bc in
        the sequence that constrains it, in both ``apply_lifting`` and
        ``set_bc``. The solve must therefore agree with the solve for a
        single bc carrying the resolved values, and swapping the bc
        order must change the solution accordingly.
        """
        from petsc4py import PETSc

        import dolfinx.fem.petsc

        sys = PETSc.Sys()
        if MPI.COMM_WORLD.size == 1:
            factor_type = "petsc"
        elif sys.hasExternalPackage("mumps"):
            factor_type = "mumps"
        elif sys.hasExternalPackage("superlu_dist"):
            factor_type = "superlu_dist"
        else:
            pytest.skip("No external solvers available in parallel")

        msh = dolfinx.mesh.create_unit_square(
            MPI.COMM_WORLD, 12, 12, ghost_mode=mode, dtype=PETSc.RealType
        )
        V = dolfinx.fem.functionspace(msh, ("Lagrange", 2))

        u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
        x = ufl.SpatialCoordinate(msh)
        a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        L = ufl.inner(2 + x[0], v) * ufl.dx

        tdim = msh.topology.dim
        msh.topology.create_connectivity(tdim - 1, tdim)
        bndry_facets = dolfinx.mesh.exterior_facet_indices(msh.topology)
        left_facets = dolfinx.mesh.locate_entities_boundary(
            msh, tdim - 1, lambda x: np.isclose(x[0], 0.0)
        )
        dofs_all = dolfinx.fem.locate_dofs_topological(V, tdim - 1, bndry_facets)
        dofs_left = dolfinx.fem.locate_dofs_topological(V, tdim - 1, left_facets)
        assert np.isin(dofs_left, dofs_all).all()

        # Two non-zero, spatially varying boundary values. The left-edge
        # dofs are constrained by both bcs.
        g_all = dolfinx.fem.Function(V)
        g_all.interpolate(lambda x: 1.0 + x[0] + 2.0 * x[1])
        g_left = dolfinx.fem.Function(V)
        g_left.interpolate(lambda x: 3.0 - x[1])
        bc_all = dolfinx.fem.dirichletbc(g_all, dofs_all)
        bc_left = dolfinx.fem.dirichletbc(g_left, dofs_left)

        # Single bc holding the values that [bc_all, bc_left] resolves to
        g_ref = dolfinx.fem.Function(V)
        g_ref.x.array[:] = g_all.x.array
        g_ref.x.array[dofs_left] = g_left.x.array[dofs_left]
        bc_ref = dolfinx.fem.dirichletbc(g_ref, dofs_all)

        def solve(bcs, label):
            problem = dolfinx.fem.petsc.LinearProblem(
                a,
                L,
                bcs=bcs,
                petsc_options_prefix=f"test_overlapping_dirichlet_bcs_{mode}_{label}_",
                petsc_options={
                    "ksp_type": "preonly",
                    "pc_type": "lu",
                    "pc_factor_mat_solver_type": factor_type,
                    "ksp_error_if_not_converged": True,
                },
            )
            uh = problem.solve()
            assert problem.solver.getConvergedReason() > 0
            return uh

        eps = 1000 * np.finfo(dolfinx.default_scalar_type).eps

        # bc_left is applied last, so it wins on the shared dofs
        uh = solve([bc_all, bc_left], "overlap")
        assert np.allclose(uh.x.array[dofs_left], g_left.x.array[dofs_left], atol=eps, rtol=eps)
        only_all = np.setdiff1d(dofs_all, dofs_left)
        assert np.allclose(uh.x.array[only_all], g_all.x.array[only_all], atol=eps, rtol=eps)
        uh_ref = solve([bc_ref], "resolved")
        assert np.allclose(uh.x.array, uh_ref.x.array, atol=eps, rtol=eps)

        # Reversing the order makes bc_all win everywhere, which is the
        # same system as applying bc_all alone
        uh_rev = solve([bc_left, bc_all], "reversed")
        assert np.allclose(uh_rev.x.array[dofs_all], g_all.x.array[dofs_all], atol=eps, rtol=eps)
        uh_all = solve([bc_all], "all")
        assert np.allclose(uh_rev.x.array, uh_all.x.array, atol=eps, rtol=eps)

        # The two orderings really do give different solutions
        assert not np.allclose(uh.x.array, uh_rev.x.array, atol=eps, rtol=eps)

    @pytest.mark.parametrize("kind", [None, "mpi", "nest"])
    def test_nonlinear_problem_bc_updates(self, kind):
        """BC reassignment updates both callbacks; input lists are snapshots."""
        from petsc4py import PETSc

        from dolfinx.fem.petsc import NonlinearProblem

        if MPI.COMM_WORLD.size == 1:
            factor_type = "petsc"
        elif PETSc.Sys().hasExternalPackage("mumps"):
            factor_type = "mumps"
        elif PETSc.Sys().hasExternalPackage("superlu_dist"):
            factor_type = "superlu_dist"
        else:
            pytest.skip("No external solvers available in parallel")

        msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 3, 3, dtype=PETSc.RealType)
        V = dolfinx.fem.functionspace(msh, ("Lagrange", 1))
        spaces = [V] if kind is None else [V, V.clone()]
        eps = 1000 * np.finfo(PETSc.RealType).eps
        options = {"snes_atol": eps, "snes_rtol": eps, "snes_error_if_not_converged": True}
        if kind == "nest":
            options["pc_type"] = "fieldsplit"
            options["pc_fieldsplit_type"] = "additive"
            for i in range(len(spaces)):
                options[f"fieldsplit_{i}_ksp_type"] = "preonly"
                options[f"fieldsplit_{i}_pc_type"] = "lu"
                options[f"fieldsplit_{i}_pc_factor_mat_solver_type"] = factor_type
        else:
            options.update(ksp_type="preonly", pc_type="lu", pc_factor_mat_solver_type=factor_type)

        def make_problem(bcs, label):
            functions = [dolfinx.fem.Function(space) for space in spaces]
            residuals = [
                ufl.inner(u - (i + 2), ufl.TestFunction(space)) * ufl.dx
                for i, (u, space) in enumerate(zip(functions, spaces, strict=True))
            ]
            return NonlinearProblem(
                residuals[0] if kind is None else residuals,
                functions[0] if kind is None else functions,
                bcs=bcs,
                kind=kind,
                petsc_options_prefix=f"test_nonlinear_bc_updates_{kind}_{label}_",
                petsc_options=options,
            )

        left = dolfinx.fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 0.0))
        right = dolfinx.fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 1.0))
        g = dolfinx.fem.Constant(msh, PETSc.ScalarType(5))
        bc_left = dolfinx.fem.dirichletbc(g, left, V)
        bc_right = dolfinx.fem.dirichletbc(PETSc.ScalarType(7), right, V)
        bcs = [bc_left]
        problem = make_problem(bcs, "updated")
        bcs.clear()
        assert problem.bcs == (bc_left,)

        for step, conditions in enumerate([(bc_left,), (bc_right,), ()]):
            if step > 0:
                problem.bcs = conditions if conditions else None
            for value in (5, 6):
                g.value = PETSc.ScalarType(value)
                problem.solve()
                reference = make_problem(conditions, f"reference_{step}_{value}")
                reference.solve()
                actual = [problem.u] if kind is None else problem.u
                expected = [reference.u] if kind is None else reference.u
                for u, u_ref in zip(actual, expected, strict=True):
                    assert np.allclose(u.x.array, u_ref.x.array, atol=eps, rtol=eps)

    @pytest.mark.parametrize("blocked", [False, True])
    def test_nonlinear_preconditioner_spaces(self, blocked):
        """Reject incompatible preconditioner spaces in constructors and callbacks."""
        from petsc4py import PETSc

        from dolfinx.fem.petsc import NonlinearProblem, assemble_jacobian, create_matrix

        msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 3, 3, dtype=PETSc.RealType)
        V = dolfinx.fem.functionspace(msh, ("Lagrange", 1))
        W = V.clone()
        u, w = dolfinx.fem.Function(V), dolfinx.fem.Function(W)
        v, z = ufl.TestFunction(V), ufl.TestFunction(W)
        aV = ufl.inner(ufl.TrialFunction(V), v) * ufl.dx
        aW = ufl.inner(ufl.TrialFunction(W), z) * ufl.dx
        residual = (
            [ufl.inner(u, v) * ufl.dx, ufl.inner(w, z) * ufl.dx]
            if blocked
            else (ufl.inner(u, v) * ufl.dx)
        )
        unknown = [u, w] if blocked else u
        invalid = [[aW, None], [None, aV]] if blocked else aW
        message = "Preconditioner form must have the same function spaces"
        with pytest.raises(ValueError, match=message):
            NonlinearProblem(
                residual,
                unknown,
                P=invalid,
                petsc_options_prefix=f"test_invalid_preconditioner_{blocked}_",
            )

        problem = NonlinearProblem(
            residual,
            unknown,
            P=[[aV, None], [None, aW]] if blocked else aV,
            petsc_options_prefix=f"test_valid_preconditioner_{blocked}_",
        )
        incompatible = dolfinx.fem.form(invalid)
        P_mat = create_matrix(incompatible)
        try:
            with pytest.raises(ValueError, match=message):
                assemble_jacobian(
                    problem.solver,
                    problem.x,
                    problem.A,
                    P_mat,
                    problem.u,
                    problem.J,
                    incompatible,
                    ([], []),
                    ([], 1.0),
                )
        finally:
            P_mat.destroy()
