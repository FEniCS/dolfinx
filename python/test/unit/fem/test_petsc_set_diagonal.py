# Copyright (C) 2026 Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for setting diagonal values of PETSc matrices."""

from mpi4py import MPI

import numpy as np
import pytest

import ufl
from dolfinx.fem import form, functionspace
from dolfinx.mesh import create_unit_square


@pytest.mark.petsc4py
class TestPETScSetDiagonal:
    """Test setting diagonal values of PETSc matrices."""

    def test_set_diagonal_per_row(self) -> None:
        """Test setting a different diagonal value for each row."""
        from petsc4py import PETSc

        from dolfinx.fem.petsc import create_matrix, set_diagonal

        mesh = create_unit_square(MPI.COMM_WORLD, 6, 5)
        V = functionspace(mesh, ("Lagrange", 1))
        u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
        a = form(ufl.inner(u, v) * ufl.dx, dtype=PETSc.ScalarType)

        # Every other owned row, with value rows[i] + 1 on row rows[i]
        rows = np.arange(0, V.dofmap.index_map.size_local, 2, dtype=np.int32)
        diagonals = (rows + 1).astype(PETSc.ScalarType)

        def diagonal(A):
            """Owned diagonal entries of A, indexed by local row."""
            d = A.getDiagonal()
            values = d.array.copy()
            d.destroy()
            return values

        A = create_matrix(a)
        set_diagonal(A, rows, diagonals)
        A.assemble()
        diag = diagonal(A)
        assert np.allclose(diag[rows], diagonals)
        mask = np.ones(diag.shape[0], dtype=bool)
        mask[rows] = False
        assert np.allclose(diag[mask], 0.0)

        # Adding the same values again doubles the diagonal
        set_diagonal(A, rows, diagonals, PETSc.InsertMode.ADD)
        A.assemble()
        assert np.allclose(diagonal(A)[rows], 2 * diagonals)

        # Number of values must match number of rows
        with pytest.raises(ValueError):
            set_diagonal(A, rows, diagonals[:-1])

        # Only INSERT_VALUES and ADD_VALUES are supported
        with pytest.raises(ValueError):
            set_diagonal(A, rows, diagonals, PETSc.InsertMode.MAX)

        A.destroy()
