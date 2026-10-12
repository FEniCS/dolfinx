# Copyright (C) 2026 Jørgen S. Dokken
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Tests for the PETSc interpolation matrix between spaces on different meshes."""

from mpi4py import MPI

import numpy as np
import pytest

from dolfinx import default_real_type, fem, mesh


@pytest.mark.petsc4py
@pytest.mark.parametrize(
    "element0, element1",
    [
        (("Lagrange", 2), ("Lagrange", 1)),
        (("Lagrange", 2, (2,)), ("RT", 1)),
        (("N1curl", 1), ("N1curl", 2)),
    ],
)
def test_petsc_matches_interpolate_nonmatching(element0, element1):
    from petsc4py import PETSc

    import dolfinx.fem.petsc

    dtype = default_real_type
    mesh0 = mesh.create_unit_square(
        MPI.COMM_WORLD, 5, 7, ghost_mode=mesh.GhostMode.shared_facet, dtype=dtype
    )
    mesh1 = mesh.create_rectangle(
        MPI.COMM_WORLD,
        [np.array([0.1, 0.05]), np.array([0.9, 0.8])],
        (4, 3),
        cell_type=mesh.CellType.quadrilateral,
        dtype=dtype,
    )
    V0 = fem.functionspace(mesh0, element0)
    V1 = fem.functionspace(mesh1, element1)
    u0 = fem.Function(V0, dtype=PETSc.ScalarType)
    u0.x.array[:] = np.random.default_rng(5).random(u0.x.array.size)
    u0.x.scatter_forward()

    eps = np.finfo(dtype).eps
    A = dolfinx.fem.petsc.nonmatching_interpolation_matrix(V0, V1, padding=1e2 * eps)
    A.assemble()
    u1 = fem.Function(V1, dtype=PETSc.ScalarType)
    A.mult(u0.x.petsc_vec, u1.x.petsc_vec)
    u1.x.scatter_forward()

    num_cells = mesh1.topology.index_map(mesh1.topology.dim).size_local
    cells = np.arange(num_cells, dtype=np.int32)
    data = fem.create_interpolation_data(V1, V0, cells, padding=1e2 * eps)
    u1_ref = fem.Function(V1, dtype=PETSc.ScalarType)
    u1_ref.interpolate_nonmatching(u0, cells, data)

    num_owned = V1.dofmap.index_map.size_local * V1.dofmap.index_map_bs
    scale = max(np.abs(u1_ref.x.array).max(initial=0), 1)
    np.testing.assert_allclose(
        u1.x.array[:num_owned], u1_ref.x.array[:num_owned], rtol=0, atol=1e3 * eps * scale
    )
    A.destroy()
