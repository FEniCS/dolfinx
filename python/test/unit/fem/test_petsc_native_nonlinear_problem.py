# Copyright (C) 2026 Jørgen S. Dokken
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Native NonlinearProblem against PETSc SNES with matching options."""

import numpy as np
import pytest

import dolfinx

# Native options and the SNES options they correspond to
_SNES_OPTIONS = {
    "atol": "snes_atol",
    "rtol": "snes_rtol",
    "stol": "snes_stol",
    "max_it": "snes_max_it",
    "divergence_tolerance": "snes_divergence_tolerance",
    "linesearch_damping": "snes_linesearch_damping",
}


@pytest.mark.petsc4py
@pytest.mark.skipif(not dolfinx.has_superlu_dist, reason="No SuperLU_DIST")
@pytest.mark.parametrize(
    "options",
    [{}, {"linesearch_damping": 0.7}, {"divergence_tolerance": 10.0}, {"max_it": 3}],
    ids=["defaults", "damping", "divergence", "max_it"],
)
def test_native_nonlinear_problem_matches_snes(options, request, nonlinear_system) -> None:
    """Same iterations, residual norms and reason as SNES ``newtonls``.

    SNES uses the ``basic`` line search, the native class's only one,
    and a direct linear solver, so the Newton steps agree up to the
    round-off of the factorisations. Each case has its own options
    prefix, as options set in the PETSc database persist between
    cases.
    """
    from petsc4py import PETSc

    from dolfinx.fem.petsc import NonlinearProblem as SNESProblem
    from dolfinx.fem.problems import NonlinearProblem

    dtype = PETSc.ScalarType
    F, u, bcs, _, _ = nonlinear_system(dtype, coupled=True)
    native = NonlinearProblem(F, u, bcs=bcs, newton_options=options)
    native.solve()

    linesearch = "none" if PETSc.Sys.getVersion() >= (3, 26, 0) else "basic"
    petsc_options = {
        "snes_linesearch_type": linesearch,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }
    petsc_options.update({_SNES_OPTIONS[key]: value for key, value in options.items()})
    F, u, bcs, _, _ = nonlinear_system(dtype, coupled=True)
    problem = SNESProblem(
        F,
        u,
        bcs=bcs,
        kind="mpi",
        petsc_options_prefix=f"test_native_nonlinear_{request.node.callspec.id}_",
        petsc_options=petsc_options,
    )
    snes = problem.solver
    snes.setConvergenceHistory()
    problem.solve()

    assert int(native.converged_reason) == snes.getConvergedReason()
    assert native.num_iterations == snes.getIterationNumber()
    fnorms, _ = snes.getConvergenceHistory()
    norms = np.array(native.residual_norms)
    assert norms.shape == fnorms.shape
    assert np.allclose(norms, fnorms, rtol=1e-8, atol=1e-10 * norms[0])
