# Copyright (C) 2021 Jørgen S. Dokken
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import dolfinx
from mpi4py import MPI
import numpy as np


def xtest_index_map_compression():

    mesh = dolfinx.UnitSquareMesh(MPI.COMM_WORLD, 8, 8)
    vertex_map = mesh.topology.index_map(0)
    sl = vertex_map.size_local
    org_ghosts = vertex_map.ghosts
    org_ghost_owners = vertex_map.ghost_owner_rank()

    # Create an index map where only every third local index is saved

    owned_entities = np.arange(0, sl, 4, dtype=np.int32)
    org_global_entities = vertex_map.local_to_global(owned_entities)

    # Add every fourth ghost
    ghost_pos = np.arange(0, len(org_ghosts), 4, dtype=np.int32)
    sub_ghosts = org_ghosts[ghost_pos]
    entities = np.hstack([owned_entities, ghost_pos + sl])
    sub_ghosts = np.array(sub_ghosts, dtype=np.int64)

    # Create compressed index map
    entities = np.array(entities, dtype=np.int32)
    new_map, org_glob = dolfinx.cpp.common.compress_index_map(vertex_map, entities)

    # Check that the new map has at least as many indices as the input
    # Might have more due to owned indices on other processes
    new_sl = new_map.size_local
    assert len(owned_entities) <= new_sl

    # Check that output of compression is sensible
    assert len(org_glob) == new_map.size_local + new_map.num_ghosts

    # Check that all original entities are contained in new index map (might be more local entries due to owned
    # entries being used on ghost processes
    assert np.isin(org_global_entities, org_glob[:new_sl]).all()
    assert len(org_global_entities) <= new_sl

    # Check that all original ghosts are in the new index map
    # Not necessarily in the same order, as the initial index map does not
    # sort ghosts per process
    assert len(sub_ghosts) == new_map.num_ghosts
    new_ghosts = org_glob[new_sl:]
    assert np.isin(new_ghosts, sub_ghosts).all()

    # Check that ghost owner is the same for matching global index
    new_ghost_owners = new_map.ghost_owner_rank()
    for (i, ghost) in enumerate(new_ghosts):
        index = np.flatnonzero(org_ghosts == ghost)[0]
        assert(org_ghost_owners[index] == new_ghost_owners[i])
