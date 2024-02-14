# Copyright (C) 2022 Jørgen S. Dokken
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for sparsity pattern creation"""

from mpi4py import MPI

from dolfinx.cpp.la import SparsityPattern
from dolfinx.fem import functionspace, locate_dofs_topological
from dolfinx.mesh import create_unit_square, exterior_facet_indices


def test_add_diagonal():
    """Test adding entries to diagonal of sparsity pattern"""
    mesh = create_unit_square(MPI.COMM_WORLD, 10, 10)
    gdim = mesh.geometry.dim
    V = functionspace(mesh, ("Lagrange", 1, (gdim,)))
    pattern = SparsityPattern(
        mesh.comm,
        [V.dofmap.index_map, V.dofmap.index_map],
        [V.dofmap.index_map_bs, V.dofmap.index_map_bs],
    )
    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    facets = exterior_facet_indices(mesh.topology)
    blocks = locate_dofs_topological(V, mesh.topology.dim - 1, facets)
    pattern.insert_diagonal(blocks)
    pattern.finalize()
    assert len(blocks) == pattern.num_nonzeros
