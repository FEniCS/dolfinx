# Copyright (C) 2024 Paul T. Kühner
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from mpi4py import MPI

import numpy as np
import pytest
from numpy import typing as npt

from dolfinx.mesh import (
    GhostMode,
    IdentityPartitionerPlaceholder,
    Mesh,
    create_cell_partitioner,
    create_unit_cube,
    create_unit_interval,
    create_unit_square,
    refine,
)
from dolfinx.multigrid import inclusion_mapping


def check_inclusion_map(
    mesh_from: Mesh, mesh_to: Mesh, map: npt.NDArray[np.int32], expect_all: bool = False
):
    if expect_all:
        assert np.all(map[: mesh_from.topology.index_map(0).size_local] >= 0)

    for i in range(len(map)):
        if map[i] == -1:
            continue
        assert np.allclose(mesh_from.geometry.x[i], mesh_to.geometry.x[map[i]])


@pytest.mark.parametrize("gdim", [1, 2, 3])
@pytest.mark.parametrize("ghost_mode_coarse", [GhostMode.none, GhostMode.shared_facet])
@pytest.mark.parametrize(
    "partitioner_fine",
    [
        create_cell_partitioner(GhostMode.none),
        create_cell_partitioner(GhostMode.shared_facet),
        IdentityPartitionerPlaceholder(),
    ],
)
def test_inclusion_map(gdim, ghost_mode_coarse, partitioner_fine):
    comm = MPI.COMM_WORLD
    if gdim == 1:
        mesh = create_unit_interval(comm, 10, ghost_mode=ghost_mode_coarse)
    elif gdim == 2:
        mesh = create_unit_square(comm, 5, 5, ghost_mode=ghost_mode_coarse)
    else:
        mesh = create_unit_cube(comm, 5, 5, 5, ghost_mode=ghost_mode_coarse)

    mesh.topology.create_entities(1)
    mesh_fine, _, _ = refine(mesh, partitioner=partitioner_fine)
    map = inclusion_mapping(mesh, mesh_fine)
    check_inclusion_map(
        mesh, mesh_fine, map, type(partitioner_fine) is IdentityPartitionerPlaceholder
    )

    map = inclusion_mapping(mesh_fine, mesh)
    check_inclusion_map(mesh_fine, mesh, map)
