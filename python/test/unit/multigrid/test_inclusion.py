# Copyright (C) 2024 Paul T. Kühner
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from mpi4py import MPI

import pytest

from dolfinx.mesh import GhostMode, create_interval, create_unit_cube, create_unit_square, refine
from dolfinx.multigrid import inclusion_mapping


@pytest.mark.parametrize(
    "ghost_mode", [GhostMode.none, GhostMode.shared_vertex, GhostMode.shared_facet]
)
def test_1d(ghost_mode):
    mesh = create_interval(MPI.COMM_WORLD, 10, (0, 1), ghost_mode=ghost_mode)
    mesh_fine, _, _ = refine(mesh)
    inclusion_mapping(mesh, mesh_fine, True)
    # TODO: extend with future operations on inclusion mappings


@pytest.mark.parametrize(
    "ghost_mode", [GhostMode.none, GhostMode.shared_vertex, GhostMode.shared_facet]
)
def test_2d(ghost_mode):
    mesh = create_unit_square(MPI.COMM_WORLD, 5, 5, ghost_mode=ghost_mode)
    mesh.topology.create_entities(1)
    mesh_fine, _, _ = refine(mesh)
    inclusion_mapping(mesh, mesh_fine, True)
    # TODO: extend with future operations on inclusion mappings


@pytest.mark.parametrize(
    "ghost_mode", [GhostMode.none, GhostMode.shared_vertex, GhostMode.shared_facet]
)
def test_3d(ghost_mode):
    mesh = create_unit_cube(MPI.COMM_WORLD, 5, 5, 5, ghost_mode=ghost_mode)
    mesh.topology.create_entities(1)
    mesh_fine, _, _ = refine(mesh)
    inclusion_mapping(mesh, mesh_fine, True)
    # TODO: extend with future operations on inclusion mappings


if __name__ == "__main__":
    pytest.main([__file__])
