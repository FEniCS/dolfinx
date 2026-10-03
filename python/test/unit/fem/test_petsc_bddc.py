# Copyright (C) 2025-2026 Stefano Zampini and Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Tests for BDDC solves of MATIS operators.

The cases here all have Dirichlet degrees of freedom that are *shared*
between processes. That is what exercises the MATIS boundary condition
handling: each process holding such a degree of freedom must place a
share of the diagonal value on its local matrix, since MATIS sums the
local contributions and a process with an empty local row makes the
subdomain solves singular.
"""

from mpi4py import MPI

import numpy as np
import pytest

import ufl
from dolfinx import default_scalar_type, la
from dolfinx.fem import (
    Function,
    assemble_scalar,
    dirichletbc,
    form,
    functionspace,
    locate_dofs_topological,
)
from dolfinx.mesh import (
    CellType,
    GhostMode,
    create_unit_square,
    exterior_facet_indices,
)


def _tols():
    """Tolerance and comparison bounds for the scalar type in use.

    In order: the relative residual asked of CG, a *relative* bound on
    the difference between the BDDC and the direct solution, and an
    absolute bound for nodal comparisons. CG stops on the residual, so
    the solution it returns is only as accurate as that tolerance,
    amplified by the conditioning of the operator. The second bound
    allows two orders of magnitude for that amplification, against a
    measured factor of under two.
    """
    single = np.finfo(default_scalar_type).bits == 32
    return (1.0e-5, 1.0e-3, 1.0e-4) if single else (1.0e-10, 1.0e-8, 1.0e-8)


def _norm(comm, u):
    """Euclidean norm of a distributed array, given its owned part."""
    return np.sqrt(comm.allreduce(float(np.sum(np.abs(u) ** 2)), MPI.SUM))


def _mesh(n=16):
    """Unit square without ghost cells, as MATIS requires."""
    msh = create_unit_square(MPI.COMM_WORLD, n, n, CellType.triangle, ghost_mode=GhostMode.none)
    msh.topology.create_connectivity(msh.topology.dim - 1, msh.topology.dim)
    return msh


def _num_shared(V, dofs):
    """Count the dofs in ``dofs`` that are held by more than one process."""
    imap, bs = V.dofmap.index_map, V.dofmap.index_map_bs
    count = la.vector(imap, bs, dtype=np.int32)
    count.array[:] = 1
    count.scatter_reverse(la.InsertMode.add)
    count.scatter_forward()
    local = int(np.count_nonzero(count.array[dofs] > 1))
    return V.mesh.comm.allreduce(local, MPI.SUM)


@pytest.mark.petsc4py
def test_bddc_poisson_shared_bc_dofs():
    """BDDC must solve a scalar problem whose bc dofs are shared.

    P1 puts degrees of freedom at vertices, so wherever a subdomain
    interface meets the boundary the Dirichlet dof is shared.
    """
    rtol, stol, atol = _tols()
    from dolfinx.fem.petsc import LinearProblem

    msh = _mesh()
    V = functionspace(msh, ("Lagrange", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(msh)
    # u = 1 + x^2 + 2y^2 solves -div(grad(u)) = -6, and is non-zero on
    # the boundary, so the boundary values are exercised rather than
    # being satisfied by any solution that merely zeroes the bc rows
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(default_scalar_type(-6), v) * ufl.dx

    g = Function(V)
    g.interpolate(lambda x: 1 + x[0] ** 2 + 2 * x[1] ** 2)
    dofs = locate_dofs_topological(V, msh.topology.dim - 1, exterior_facet_indices(msh.topology))
    bcs = [dirichletbc(g, dofs)]
    assert np.abs(g.x.array[dofs]).max() > 0.0
    if msh.comm.size > 1:
        assert _num_shared(V, dofs) > 0

    uh, u_ref = Function(V), Function(V)
    LinearProblem(
        a,
        L,
        u=uh,
        bcs=bcs,
        kind="is",
        petsc_options_prefix="test_bddc_poisson_",
        petsc_options={
            "ksp_type": "cg",
            "pc_type": "bddc",
            "ksp_rtol": rtol,
            "ksp_error_if_not_converged": True,
        },
    ).solve()
    LinearProblem(
        a,
        L,
        u=u_ref,
        bcs=bcs,
        petsc_options_prefix="test_bddc_poisson_ref_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    ).solve()

    # The BDDC solve must reproduce a direct solve of the same system
    n = V.dofmap.index_map.size_local
    diff = _norm(msh.comm, uh.x.array[:n] - u_ref.x.array[:n])
    assert diff < stol * _norm(msh.comm, u_ref.x.array[:n])

    # The solution must take the prescribed values on the constrained
    # dofs, including those shared between processes
    assert np.allclose(uh.x.array[dofs], g.x.array[dofs], rtol=rtol, atol=atol)

    # ... and the system must be a sane discretisation of the problem
    u_exact = 1 + x[0] ** 2 + 2 * x[1] ** 2
    error = form(ufl.inner(uh - u_exact, uh - u_exact) * ufl.dx)
    l2 = np.sqrt(msh.comm.allreduce(assemble_scalar(error), MPI.SUM).real)
    assert l2 < 1.0e-2


@pytest.mark.petsc4py
def test_bddc_component_wise_bc():
    """BDDC must solve a vector problem constrained in one component only.

    A roller-type condition constrains one component of a shared vertex
    and leaves the other free, so the boundary condition cannot be
    handled by dropping the whole node from the local space.
    """
    rtol, stol, atol = _tols()
    from dolfinx.fem.petsc import LinearProblem

    msh = _mesh()
    V = functionspace(msh, ("Lagrange", 1, (2,)))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(msh)
    f = ufl.as_vector((ufl.sin(ufl.pi * x[0]), 1.0))
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(u, v) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx

    Vx, _ = V.sub(0).collapse()
    dofs = locate_dofs_topological(
        (V.sub(0), Vx), msh.topology.dim - 1, exterior_facet_indices(msh.topology)
    )
    gx = Function(Vx)
    gx.interpolate(lambda x: 1 + x[1])
    bcs = [dirichletbc(gx, dofs, V.sub(0))]
    assert np.abs(gx.x.array[dofs[1]]).max() > 0.0
    if msh.comm.size > 1:
        assert _num_shared(V, dofs[0]) > 0

    uh, u_ref = Function(V), Function(V)
    LinearProblem(
        a,
        L,
        u=uh,
        bcs=bcs,
        kind="is",
        petsc_options_prefix="test_bddc_vector_",
        petsc_options={
            "ksp_type": "cg",
            "pc_type": "bddc",
            "ksp_rtol": rtol,
            "ksp_error_if_not_converged": True,
        },
    ).solve()
    LinearProblem(
        a,
        L,
        u=u_ref,
        bcs=bcs,
        petsc_options_prefix="test_bddc_vector_ref_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    ).solve()

    n = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
    diff = _norm(msh.comm, uh.x.array[:n] - u_ref.x.array[:n])
    assert diff < stol * _norm(msh.comm, u_ref.x.array[:n])

    # The constrained component must take the prescribed values, and the
    # free component must not have been constrained with it
    assert np.allclose(uh.x.array[dofs[0]], gx.x.array[dofs[1]], rtol=rtol, atol=atol)
    free = dofs[0] + 1
    assert np.abs(uh.x.array[free] - gx.x.array[dofs[1]]).max() > atol


@pytest.mark.petsc4py
def test_matis_bc_diagonal_matches_aij():
    """Applying bcs to a MATIS matrix must give the same operator as AIJ.

    The diagonal value each process contributes is scaled so that the
    contributions sum to the requested value, independently of how many
    processes share the degree of freedom.
    """
    from petsc4py import PETSc

    from dolfinx.fem.petsc import assemble_matrix

    msh = _mesh(8)
    V = functionspace(msh, ("Lagrange", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = form(ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(u, v) * ufl.dx)
    dofs = locate_dofs_topological(V, msh.topology.dim - 1, exterior_facet_indices(msh.topology))
    bcs = [dirichletbc(default_scalar_type(2.5), dofs, V)]

    A_is = assemble_matrix(a, bcs=bcs, kind="is")
    A_is.assemble()
    A_aij = assemble_matrix(a, bcs=bcs)
    A_aij.assemble()

    # The bc rows must carry exactly 1 on the assembled operator, not a
    # multiple of the number of sharing processes. Convert into a new
    # matrix: passing no output converts A_is in place.
    A_cmp = A_is.convert(PETSc.Mat.Type.AIJ, PETSc.Mat())
    owned = np.asarray(dofs[dofs < V.dofmap.index_map.size_local], dtype=np.int32)
    assert np.allclose(A_cmp.getDiagonal().getArray()[owned], 1.0)
    assert A_is.getType() == PETSc.Mat.Type.IS

    A_cmp.axpy(-1.0, A_aij, PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
    eps = np.finfo(default_scalar_type).eps
    assert A_cmp.norm() == pytest.approx(0.0, abs=100 * eps * A_aij.norm())

    A_is.destroy(), A_aij.destroy(), A_cmp.destroy()
