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
    """Krylov tolerance and comparison bound for the scalar type in use."""
    single = np.finfo(default_scalar_type).bits == 32
    return (1.0e-5, 1.0e-4) if single else (1.0e-10, 1.0e-8)


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
    rtol, atol = _tols()
    from dolfinx.fem.petsc import LinearProblem

    msh = _mesh()
    V = functionspace(msh, ("Lagrange", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(msh)
    f = 2 * ufl.pi**2 * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx

    dofs = locate_dofs_topological(V, msh.topology.dim - 1, exterior_facet_indices(msh.topology))
    bcs = [dirichletbc(default_scalar_type(0), dofs, V)]
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
    diff = np.sqrt(
        msh.comm.allreduce(float(np.sum(np.abs(uh.x.array[:n] - u_ref.x.array[:n]) ** 2)), MPI.SUM)
    )
    assert diff < atol

    # ... and the system must be a sane discretisation of the problem
    u_exact = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
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
    rtol, atol = _tols()
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
    bcs = [dirichletbc(Function(Vx), dofs, V.sub(0))]
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
    diff = np.sqrt(
        msh.comm.allreduce(float(np.sum(np.abs(uh.x.array[:n] - u_ref.x.array[:n]) ** 2)), MPI.SUM)
    )
    assert diff < atol


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
    bcs = [dirichletbc(default_scalar_type(0), dofs, V)]

    A_is = assemble_matrix(a, bcs=bcs, kind="is")
    A_is.assemble()
    A_aij = assemble_matrix(a, bcs=bcs)
    A_aij.assemble()

    # The bc rows must carry exactly 1 on the assembled operator, not a
    # multiple of the number of sharing processes
    d = A_is.convert(PETSc.Mat.Type.AIJ).getDiagonal()
    owned = np.asarray(dofs[dofs < V.dofmap.index_map.size_local], dtype=np.int32)
    assert np.allclose(d.getArray()[owned], 1.0)

    A_cmp = A_is.convert(PETSc.Mat.Type.AIJ)
    A_cmp.axpy(-1.0, A_aij, PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
    eps = np.finfo(default_scalar_type).eps
    assert A_cmp.norm() == pytest.approx(0.0, abs=100 * eps * A_aij.norm())

    A_is.destroy(), A_aij.destroy(), A_cmp.destroy()
