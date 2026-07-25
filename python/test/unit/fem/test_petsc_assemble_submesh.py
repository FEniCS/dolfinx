# Copyright (C) 2022-2024 Joseph P. Dean, Jørgen S. Dokken
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for submesh assembly requiring PETSc."""

from mpi4py import MPI

import numpy as np
import pytest

import ufl
from dolfinx import fem
from dolfinx.mesh import GhostMode, create_submesh, create_unit_square, locate_entities


@pytest.mark.petsc4py
def test_mixed_measures():
    """Test block assembly of forms where the integration measure in each
    block may be different.
    """
    from petsc4py import PETSc

    from dolfinx.fem.petsc import assemble_vector

    comm = MPI.COMM_WORLD
    msh = create_unit_square(comm, 16, 21, ghost_mode=GhostMode.none)

    # Create a submesh of some cells
    tdim = msh.topology.dim
    smsh_cells = locate_entities(msh, tdim, lambda x: x[0] <= 0.5)
    smsh, entity_map = create_submesh(msh, tdim, smsh_cells)[:2]

    # Create function spaces over each mesh
    V = fem.functionspace(msh, ("Lagrange", 1))
    Q = fem.functionspace(smsh, ("Lagrange", 1))

    # Define two integration measures, one over the mesh, the other over the submesh
    dx_msh = ufl.Measure("dx", msh, subdomain_data=[(1, smsh_cells)])
    dx_smsh = ufl.Measure("dx", smsh)

    # Trial and test functions
    v = ufl.TestFunction(V)
    q = ufl.TestFunction(Q)

    entity_maps = [entity_map]
    # First, assemble a block vector using both dx_msh and dx_smsh
    L = [fem.form(ufl.inner(2.3, v) * dx_msh), fem.form(ufl.inner(1.3, q) * dx_smsh)]
    b0 = assemble_vector(L, kind=PETSc.Vec.Type.MPI)
    b0.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    # Now, assemble the same vector using only dx_msh
    L = [
        fem.form(ufl.inner(2.3, v) * dx_msh),
        fem.form(ufl.inner(1.3, q) * dx_msh(1), entity_maps=entity_maps),
    ]
    b1 = assemble_vector(L, kind=PETSc.Vec.Type.MPI)
    b1.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    # Check the results are the same
    assert np.allclose(b0.norm(), b1.norm())
