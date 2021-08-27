# Copyright (C) 2021 Jørgen S. Dokken and Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import dolfinx
import numpy as np
from mpi4py import MPI


def test_sub_index_map():

    comm = MPI.COMM_WORLD
    myrank = comm.rank
    if myrank == 0:
        ghosts = []
        map = dolfinx.cpp.common.IndexMap(MPI.COMM_WORLD, 50, np.arange(
            1, comm.size, dtype=np.int32), ghosts, len(ghosts) * [0])
    else:
        ghosts = [7, 5, 2, 0, 16]
        map = dolfinx.cpp.common.IndexMap(MPI.COMM_WORLD, 25, [], ghosts, len(ghosts) * [0])

    assert map.size_global == 50 + 25 * (comm.size - 1)

    # Create an index map where only every second (even) local index is
    # extracted
    entities = np.arange(0, map.size_local, 2, dtype=np.int32)
    submap, ghosts_pos = map.create_submap(entities)

    for pos, global_idx in zip(ghosts_pos, ghosts[::2]):
        assert ghosts[pos] % 2 == 0

    # assert submap.size_local == len(entities)
    # assert submap.size_global == mesh.mpi_comm().allreduce(len(entities), op=MPI.SUM)

    # org_global_entities = vertex_map.local_to_global(owned_entities)

    # # Add every fourth ghost
    # ghost_pos = np.arange(0, len(org_ghosts), 4, dtype=np.int32)
    # sub_ghosts = org_ghosts[ghost_pos]
    # entities = np.hstack([owned_entities, ghost_pos + sl])
    # sub_ghosts = np.array(sub_ghosts, dtype=np.int64)

    # Create sub  index map

    # # Check that the new map has at least as many indices as the input
    # # Might have more due to owned indices on other processes
    # new_sl = new_map.size_local
    # assert len(owned_entities) <= new_sl

    # # Check that output of compression is sensible
    # assert len(org_glob) == new_map.size_local + new_map.num_ghosts

    # # Check that all original entities are contained in new index map (might be more local entries due to owned
    # # entries being used on ghost processes
    # assert np.isin(org_global_entities, org_glob[:new_sl]).all()
    # assert len(org_global_entities) <= new_sl

    # # Check that all original ghosts are in the new index map
    # # Not necessarily in the same order, as the initial index map does not
    # # sort ghosts per process
    # assert len(sub_ghosts) == new_map.num_ghosts
    # new_ghosts = org_glob[new_sl:]
    # assert np.isin(new_ghosts, sub_ghosts).all()

    # # Check that ghost owner is the same for matching global index
    # new_ghost_owners = new_map.ghost_owner_rank()
    # for (i, ghost) in enumerate(new_ghosts):
    #     index = np.flatnonzero(org_ghosts == ghost)[0]
    #     assert(org_ghost_owners[index] == new_ghost_owners[i])
