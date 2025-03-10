# Copyright (C) 2024 Paul T. Kühner
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from mpi4py import MPI

import pytest

from dolfinx.fem import functionspace
from dolfinx.mesh import GhostMode, create_interval, refine
from dolfinx.multigrid import create_sparsity_pattern, inclusion_mapping


@pytest.mark.parametrize(
    "ghost_mode", [GhostMode.none, GhostMode.shared_vertex, GhostMode.shared_facet]
)
def test_1d(ghost_mode):
    mesh = create_interval(MPI.COMM_WORLD, 10, (0, 1), ghost_mode=ghost_mode)
    mesh_fine, _, _ = refine(mesh)
    inclusion_map = inclusion_mapping(mesh, mesh_fine)
    V = functionspace(mesh, ("Lagrange", 1, (1,)))
    V_fine = functionspace(mesh_fine, ("Lagrange", 1, (1,)))
    assert V.element == V_fine.element
    V_fine.mesh.topology.create_connectivity(1, 0)
    V_fine.mesh.topology.create_connectivity(0, 1)
    create_sparsity_pattern(V, V_fine, inclusion_map)
    # T = matrix_csr(sp)
    # assemble_transfer_matrix(
    #     T._cpp_object,
    #     V._cpp_object,
    #     V_fine._cpp_object,
    #     inclusion_map,
    #     lambda i: 1.0 if i == 0 else 0.5,
    # )
    # continue with assembly of matrix


@pytest.mark.parametrize("degree", [2, 3, 4])
@pytest.mark.parametrize(
    "ghost_mode", [GhostMode.none, GhostMode.shared_vertex, GhostMode.shared_facet]
)
def test_1d_higher_order(degree, ghost_mode):
    mesh = create_interval(MPI.COMM_WORLD, 10, (0, 1), ghost_mode=ghost_mode)
    mesh_fine, _, _ = refine(mesh)

    # this is a strictly geometric operation, and thus should pass
    inclusion_map = inclusion_mapping(mesh, mesh_fine)

    V = functionspace(mesh, ("Lagrange", degree, (1,)))
    V_fine = functionspace(mesh_fine, ("Lagrange", degree, (1,)))

    V_fine.mesh.topology.create_connectivity(1, 0)
    V_fine.mesh.topology.create_connectivity(0, 1)

    # Check not supported throws
    with pytest.raises(Exception):
        create_sparsity_pattern(V, V_fine, inclusion_map)


if __name__ == "__main__":
    pytest.main([__file__])
