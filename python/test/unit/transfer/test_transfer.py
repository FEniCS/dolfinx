# Copyright (C) 2024 Paul T. Kühner
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from mpi4py import MPI

import pytest

from dolfinx.fem import functionspace
from dolfinx.mesh import GhostMode, create_interval, refine
from dolfinx.transfer import create_sparsity_pattern, inclusion_mapping


@pytest.mark.parametrize(
    "ghost_mode", [GhostMode.none, GhostMode.shared_vertex, GhostMode.shared_facet]
)
def test_1d(ghost_mode):
    mesh = create_interval(MPI.COMM_WORLD, 10, (0, 1), ghost_mode=ghost_mode)
    mesh_fine, _, _ = refine(mesh)
    inclusion_map = inclusion_mapping(mesh, mesh_fine)
    V = functionspace(mesh, ("Lagrange", 1, (1,)))
    V_fine = functionspace(mesh_fine, ("Lagrange", 1, (1,)))
    V_fine.mesh.topology.create_connectivity(1, 0)
    V_fine.mesh.topology.create_connectivity(0, 1)
    create_sparsity_pattern(V, V_fine, inclusion_map)
    # continue with assembly of matrix
