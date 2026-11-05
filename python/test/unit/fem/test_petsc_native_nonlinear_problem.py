# Copyright (C) 2026 Jørgen S. Dokken
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Native NonlinearProblem against PETSc SNES with matching options."""

from mpi4py import MPI

import numpy as np
import pytest

import dolfinx
from dolfinx.fem import (
    Expression,
    Function,
    dirichletbc,
    functionspace,
    locate_dofs_topological,
)
from dolfinx.mesh import create_unit_square, exterior_facet_indices
from ufl import SpatialCoordinate, TestFunction, div, dx, grad, inner

# Native options and the SNES options they correspond to
_SNES_OPTIONS = {
    "atol": "snes_atol",
    "rtol": "snes_rtol",
    "stol": "snes_stol",
    "max_it": "snes_max_it",
    "divergence_tolerance": "snes_divergence_tolerance",
    "linesearch_damping": "snes_linesearch_damping",
}


def _coupled_system(dtype):
    """Coupled nonlinear system on P2 and P1, with zero initial guesses.

    ``-div((1 + u0**2) grad(u0)) + u1 = f0`` and
    ``-div(grad(u1)) + u0 * u1 = f1``, with Dirichlet conditions on the
    whole boundary. From the zero guess the residual norm first grows by
    more than 10, and Newton's method takes more than 3 iterations.
    """
    mesh_dtype = dtype().real.dtype
    mesh = create_unit_square(MPI.COMM_WORLD, 4, 5, dtype=mesh_dtype)
    V0 = functionspace(mesh, ("Lagrange", 2))
    V1 = functionspace(mesh, ("Lagrange", 1))
    u0, u1 = Function(V0, dtype=dtype), Function(V1, dtype=dtype)
    v0, v1 = TestFunction(V0), TestFunction(V1)
    x = SpatialCoordinate(mesh)
    g0 = 1 + x[0] ** 2 + 2 * x[1]
    g1 = 1 + x[0] + x[1]
    f0 = -div((1 + g0**2) * grad(g0)) + g1
    f1 = -div(grad(g1)) + g0 * g1
    F = [
        inner((1 + u0**2) * grad(u0), grad(v0)) * dx + inner(u1 - f0, v0) * dx,
        inner(grad(u1), grad(v1)) * dx + inner(u0 * u1 - f1, v1) * dx,
    ]

    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim - 1, tdim)
    facets = exterior_facet_indices(mesh.topology)
    bcs = []
    for V, g in ((V0, g0), (V1, g1)):
        gh = Function(V, dtype=dtype)
        gh.interpolate(Expression(g, V.element.interpolation_points, dtype=dtype))
        bcs.append(dirichletbc(gh, locate_dofs_topological(V, tdim - 1, facets)))
    return F, [u0, u1], bcs


@pytest.mark.petsc4py
@pytest.mark.skipif(not dolfinx.has_superlu_dist, reason="No SuperLU_DIST")
@pytest.mark.parametrize(
    "options",
    [{}, {"linesearch_damping": 0.7}, {"divergence_tolerance": 10.0}, {"max_it": 3}],
    ids=["defaults", "damping", "divergence", "max_it"],
)
def test_native_nonlinear_problem_matches_snes(options, request) -> None:
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
    if np.dtype(dtype) == np.complex64:
        pytest.skip("SuperLU_DIST does not support complex64.")
    F, u, bcs = _coupled_system(dtype)
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
    F, u, bcs = _coupled_system(dtype)
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
