# Copyright (C) 2026 Jack S. Hale, Chris Richardson
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for native variational problem classes."""

from mpi4py import MPI

import numpy as np
import pytest

import dolfinx
from dolfinx import default_scalar_type
from dolfinx.fem import (
    Constant,
    Function,
    assemble_scalar,
    dirichletbc,
    form,
    functionspace,
    locate_dofs_topological,
)
from dolfinx.mesh import create_unit_square, exterior_facet_indices
from ufl import (
    SpatialCoordinate,
    TestFunction,
    TrialFunction,
    as_vector,
    div,
    dot,
    dx,
    grad,
    inner,
)


@pytest.mark.parametrize(
    "dtype",
    [
        np.float32,
        np.float64,
        pytest.param(np.complex128, marks=pytest.mark.xfail_win32_complex),
    ],
)
@pytest.mark.skipif(not dolfinx.has_superlu_dist, reason="No SuperLU_DIST")
def test_superlu_problem(dtype) -> None:
    """Manufactured Poisson and screened problem with exact solution u = x[1]**3."""
    from dolfinx.fem.problems import LinearProblem

    mesh_dtype = dtype().real.dtype
    mesh = create_unit_square(MPI.COMM_WORLD, 5, 5, dtype=mesh_dtype)
    V = functionspace(mesh, ("Lagrange", 4))
    u, v = TrialFunction(V), TestFunction(V)

    a = inner(grad(u), grad(v)) * dx

    # Exact solution
    def u_ex(x):
        return x[1] ** 3

    x = SpatialCoordinate(mesh)
    f = -div(grad(u_ex(x)))
    L = inner(f, v) * dx

    u_bc = Function(V, dtype=dtype)
    u_bc.interpolate(u_ex)

    # Create Dirichlet boundary condition
    facetdim = mesh.topology.dim - 1
    mesh.topology.create_connectivity(facetdim, mesh.topology.dim)
    bndry_facets = exterior_facet_indices(mesh.topology)
    bdofs = locate_dofs_topological(V, facetdim, bndry_facets)
    bc = dirichletbc(u_bc, bdofs)

    def check_error(u_ex, uh):
        M = (u_ex(x) - uh) ** 2 * dx
        M = form(M, dtype=dtype)
        error = mesh.comm.allreduce(assemble_scalar(M), op=MPI.SUM)
        eps = np.sqrt(np.finfo(dtype).eps)
        assert np.isclose(error, 0.0, atol=eps)

    problem = LinearProblem(a, L, [bc], dtype=dtype, superlu_dist_options={"SymmetricMode": "YES"})
    uh = problem.solve()

    check_error(u_ex, uh)

    # Second solve
    uh = problem.solve()
    check_error(u_ex, uh)


@pytest.mark.skipif(not dolfinx.has_superlu_dist, reason="No SuperLU_DIST")
def test_superlu_problem_default_args() -> None:
    from dolfinx.fem.problems import LinearProblem

    mesh = create_unit_square(MPI.COMM_WORLD, 5, 5)
    V = functionspace(mesh, ("Lagrange", 4))
    u, v = TrialFunction(V), TestFunction(V)

    a = inner(grad(u), grad(v)) * dx + inner(u, v) * dx
    L = inner(1.0, v) * dx

    # Check that no optional arguments leads to successful solve
    problem = LinearProblem(a, L)
    _ = problem.solve()


@pytest.mark.parametrize(
    "dtype",
    [
        np.float32,
        np.float64,
        pytest.param(np.complex128, marks=pytest.mark.xfail_win32_complex),
    ],
)
@pytest.mark.parametrize("upper_coupling", [True, False])
@pytest.mark.skipif(not dolfinx.has_superlu_dist, reason="No SuperLU_DIST")
def test_superlu_problem_blocked(dtype, upper_coupling) -> None:
    """Coupled reaction-diffusion system on a P2 and a vector P1 space.

    The exact solution lies in the discrete spaces, so it is recovered
    up to round-off. Without ``upper_coupling`` the block ``a[0][1]`` is
    ``None``.
    """
    from dolfinx.fem.problems import LinearProblem

    mesh_dtype = dtype().real.dtype
    mesh = create_unit_square(MPI.COMM_WORLD, 4, 5, dtype=mesh_dtype)
    V0 = functionspace(mesh, ("Lagrange", 2))
    V1 = functionspace(mesh, ("Lagrange", 1, (2,)))
    u0, v0 = TrialFunction(V0), TestFunction(V0)
    u1, v1 = TrialFunction(V1), TestFunction(V1)
    x = SpatialCoordinate(mesh)

    def u0_ex(x):
        return x[0] ** 2 + x[0] * x[1]

    def u1_ex(x):
        return as_vector([x[0] + 2 * x[1], 1 - x[1]])

    e = as_vector([1, 1])
    a00 = inner(grad(u0), grad(v0)) * dx + inner(u0, v0) * dx
    a01 = inner(dot(u1, e), v0) * dx if upper_coupling else None
    a10 = inner(u0 * e, v1) * dx
    a11 = inner(grad(u1), grad(v1)) * dx + 2 * inner(u1, v1) * dx
    f0 = -div(grad(u0_ex(x))) + u0_ex(x)
    if upper_coupling:
        f0 += dot(u1_ex(x), e)
    f1 = -div(grad(u1_ex(x))) + 2 * u1_ex(x) + u0_ex(x) * e
    L = [inner(f0, v0) * dx, inner(f1, v1) * dx]

    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim - 1, tdim)
    facets = exterior_facet_indices(mesh.topology)
    g0 = Function(V0, dtype=dtype)
    g0.interpolate(lambda x: x[0] ** 2 + x[0] * x[1])
    g1 = Function(V1, dtype=dtype)
    g1.interpolate(lambda x: np.vstack([x[0] + 2 * x[1], 1 - x[1]]))
    bcs = [
        dirichletbc(g0, locate_dofs_topological(V0, tdim - 1, facets)),
        dirichletbc(g1, locate_dofs_topological(V1, tdim - 1, facets)),
    ]

    u = [Function(V0, dtype=dtype), Function(V1, dtype=dtype)]
    problem = LinearProblem([[a00, a01], [a10, a11]], L, bcs=bcs, u=u, dtype=dtype)
    for _ in range(2):
        uh0, uh1 = problem.solve()
        assert uh0 is u[0] and uh1 is u[1]
        for uh, u_ex in ((uh0, u0_ex(x)), (uh1, u1_ex(x))):
            M = form(inner(uh - u_ex, uh - u_ex) * dx, dtype=dtype)
            error = np.sqrt(mesh.comm.allreduce(assemble_scalar(M), op=MPI.SUM).real)
            assert error < 100 * np.sqrt(np.finfo(dtype).eps)
    assert isinstance(problem.A, tuple) and isinstance(problem.b, tuple) and len(problem.b) == 2
    assert isinstance(problem.u, tuple)


@pytest.mark.skipif(not dolfinx.has_superlu_dist, reason="No SuperLU_DIST")
def test_superlu_problem_compiled_and_zero_rhs() -> None:
    """Compiled forms are accepted, and a None right-hand side is zero."""
    from dolfinx.fem.problems import LinearProblem

    mesh = create_unit_square(MPI.COMM_WORLD, 4, 4)
    V0 = functionspace(mesh, ("Lagrange", 2))
    V1 = functionspace(mesh, ("Lagrange", 1))
    u0, v0 = TrialFunction(V0), TestFunction(V0)
    u1, v1 = TrialFunction(V1), TestFunction(V1)

    # A single compiled form, as accepted before blocked problems
    a00 = inner(grad(u0), grad(v0)) * dx + inner(u0, v0) * dx
    uh = LinearProblem(form(a00), form(inner(1.0, v0) * dx)).solve()
    assert isinstance(uh, Function)

    a = [[a00, inner(u1, v0) * dx], [inner(u0, v1) * dx, inner(u1, v1) * dx]]
    L0 = inner(1.0, v0) * dx
    uh_none = LinearProblem(a, [L0, None]).solve()
    uh_zero = LinearProblem(a, [L0, inner(Constant(mesh, 0.0), v1) * dx]).solve()
    for x_none, x_zero in zip(uh_none, uh_zero, strict=True):
        assert np.allclose(x_none.x.array, x_zero.x.array)


@pytest.mark.skipif(not dolfinx.has_superlu_dist, reason="No SuperLU_DIST")
def test_superlu_problem_blocked_invalid() -> None:
    """Invalid blocked input raises before any solve."""
    from dolfinx.fem.problems import LinearProblem

    mesh = create_unit_square(MPI.COMM_WORLD, 3, 3)
    V0 = functionspace(mesh, ("Lagrange", 2))
    V1 = functionspace(mesh, ("Lagrange", 1))
    u0, v0 = TrialFunction(V0), TestFunction(V0)
    u1, v1 = TrialFunction(V1), TestFunction(V1)
    a = [[inner(grad(u0), grad(v0)) * dx, inner(u1, v0) * dx], [inner(u0, v1) * dx, None]]
    L = [inner(1.0, v0) * dx, inner(1.0, v1) * dx]

    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim - 1, tdim)
    facets = exterior_facet_indices(mesh.topology)
    dofs1 = locate_dofs_topological(V1, tdim - 1, facets)
    bc1 = dirichletbc(np.array(0.0, dtype=default_scalar_type), dofs1, V1)

    # A None diagonal block cannot hold the constrained rows' diagonal
    with pytest.raises(ValueError, match=r"a\[1\]\[1\] is None"):
        LinearProblem(a, L, bcs=[bc1])

    # Nor can a diagonal block with different test and trial spaces
    u1c = TrialFunction(V1.clone())
    a_c = [[a[0][0], inner(u1c, v0) * dx], [inner(u0, v1) * dx, inner(u1c, v1) * dx]]
    with pytest.raises(ValueError, match="different test and trial spaces"):
        LinearProblem(a_c, L, bcs=[bc1])

    # A rejected assignment keeps the previous conditions
    problem = LinearProblem(a, L)
    with pytest.raises(ValueError, match="is None"):
        problem.bcs = [bc1]
    assert problem.bcs == ()

    # A space may appear in only one block row and one block column
    a_dup = [[inner(u0, v0) * dx, inner(u0, v0) * dx], [inner(u0, v0) * dx, inner(u0, v0) * dx]]
    with pytest.raises(ValueError, match="appears in another"):
        LinearProblem(a_dup, [inner(1.0, v0) * dx, inner(1.0, v0) * dx])

    # Square block array, one linear form per block row, and a and L
    # both blocked or both single forms
    with pytest.raises(ValueError, match="square"):
        LinearProblem([a[0]], L[:1])
    with pytest.raises(ValueError, match="block rows"):
        LinearProblem(a, L[:1])
    with pytest.raises(ValueError, match="both be single forms"):
        LinearProblem(a, L[0])
    with pytest.raises(ValueError, match="test space of L"):
        LinearProblem(a, L[::-1])

    # One solution function per block column, on its trial space and of
    # the given dtype
    with pytest.raises(ValueError, match="functions"):
        LinearProblem(a, L, u=[Function(V0)])
    with pytest.raises(ValueError, match="trial space"):
        LinearProblem(a, L, u=[Function(V1), Function(V0)])
    with pytest.raises(ValueError, match=r"u\[1\]\.dtype"):
        LinearProblem(a, L, u=[Function(V0), Function(V1, dtype=np.complex128)])
