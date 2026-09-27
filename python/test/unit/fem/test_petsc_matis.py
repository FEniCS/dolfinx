# Copyright (C) 2025-2026 Stefano Zampini and Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Tests for assembly into PETSc's unassembled MATIS format."""

from mpi4py import MPI
from petsc4py import PETSc

import numpy as np
import pytest

import ufl
from dolfinx.fem import Constant, form, functionspace
from dolfinx.fem.petsc import assemble_matrix
from dolfinx.mesh import CellType, GhostMode, create_unit_cube, create_unit_square


def _unit_mesh(cell_type, n):
    """Create a mesh without ghost cells, as MATIS requires."""
    if cell_type in (CellType.triangle, CellType.quadrilateral):
        return create_unit_square(MPI.COMM_WORLD, n, n, cell_type, ghost_mode=GhostMode.none)
    return create_unit_cube(MPI.COMM_WORLD, n, n, n, cell_type, ghost_mode=GhostMode.none)


@pytest.mark.petsc4py
@pytest.mark.parametrize("cell_type", [CellType.triangle, CellType.quadrilateral])
@pytest.mark.parametrize("degree", [1, 2])
@pytest.mark.parametrize("shape", [None, (2,)])
def test_matis_matches_aij(cell_type, degree, shape):
    """A MATIS matrix must assemble to the same operator as an AIJ one.

    MATIS holds one unassembled matrix per process, with the global
    operator being the sum of the local contributions. Converting to AIJ
    performs that sum, which must reproduce a directly assembled matrix.
    """
    msh = _unit_mesh(cell_type, 6)
    V = functionspace(msh, ("Lagrange", degree, shape) if shape else ("Lagrange", degree))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    k = Constant(msh, PETSc.ScalarType(2.5))
    a = form(k * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(u, v) * ufl.dx)

    A_is = assemble_matrix(a, kind="is")
    A_is.assemble()
    assert A_is.getType() == PETSc.Mat.Type.IS

    A_aij = assemble_matrix(a)
    A_aij.assemble()

    A_converted = A_is.convert(PETSc.Mat.Type.AIJ)
    A_converted.axpy(-1.0, A_aij, PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
    assert A_converted.norm() == pytest.approx(0.0, abs=1e-10 * A_aij.norm())

    A_is.destroy(), A_aij.destroy(), A_converted.destroy()


@pytest.mark.petsc4py
def test_matis_local_to_global_map():
    """MATIS requires the local-to-global maps on the matrix communicator.

    ``MatSetLocalToGlobalMapping_IS`` calls ``PetscCheckSameComm``, so a
    map built on ``MPI_COMM_SELF`` is rejected in a PETSc debug build.
    """
    msh = _unit_mesh(CellType.triangle, 4)
    V = functionspace(msh, ("Lagrange", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = form(ufl.inner(u, v) * ufl.dx)

    A = assemble_matrix(a, kind="is")
    A.assemble()
    rmap, cmap = A.getLGMap()
    assert rmap.getSize() == V.dofmap.index_map.size_local + V.dofmap.index_map.num_ghosts
    assert rmap.getComm().getSize() == msh.comm.size
    assert cmap.getComm().getSize() == msh.comm.size
    A.destroy()


@pytest.mark.petsc4py
@pytest.mark.parametrize("kind", [None, "is"])
def test_blocked_matis(kind):
    """A blocked matrix of kind ``kind`` must match the AIJ equivalent.

    ``create_matrix_block`` builds field-concatenated local-to-global
    maps and passes them to the matrix constructor, which for MATIS must
    happen before preallocation.
    """
    msh = _unit_mesh(CellType.triangle, 6)
    P1 = functionspace(msh, ("Lagrange", 1))
    P2 = functionspace(msh, ("Lagrange", 2))
    Q = ufl.MixedFunctionSpace(P1, P2)
    p, q = ufl.TrialFunctions(Q)
    r, s = ufl.TestFunctions(Q)
    a = form(
        ufl.extract_blocks(
            ufl.inner(p, r) * ufl.dx + ufl.inner(q, r) * ufl.dx + ufl.inner(q, s) * ufl.dx
        )
    )

    A = assemble_matrix(a, kind=kind)
    A.assemble()
    A_ref = assemble_matrix(a)
    A_ref.assemble()

    A_cmp = A.convert(PETSc.Mat.Type.AIJ) if kind == "is" else A
    assert np.isclose(A_cmp.norm(), A_ref.norm(), rtol=1e-10)
    A.destroy(), A_ref.destroy()
