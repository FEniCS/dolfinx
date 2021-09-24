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

    # Create index map with one ghost from each other process
    dest_ranks = np.delete(np.arange(0, comm.size, dtype=np.int32), myrank)
    size_local = np.math.factorial(7)
    assert comm.size < 8
    local_range = (size_local * myrank, size_local * (myrank + 1))
    ghosts = np.array([i % size_local + size_local * dest_ranks[i] for i in range(len(dest_ranks))])
    src_ranks = dest_ranks
    map = dolfinx.cpp.common.IndexMap(MPI.COMM_WORLD, size_local, dest_ranks, ghosts, src_ranks)
    #assert map.size_global == size_local * comm.size
    # Create sub-map with (myrank + myrank % 2) elements
    new_local_size = int((myrank + myrank % 2))
    local_indices = np.arange(new_local_size, dtype=np.int32)

    submap, ghosts_pos = map.create_submap(local_indices)
    assert submap.size_global == sum([rank + rank % 2 for rank in range(comm.size)])
    assert submap.size_local == new_local_size
    sub_owners = submap.ghost_owner_rank()
    owners = map.ghost_owner_rank()
    # Running this code on 2 procs gives you that the ghost on process 0 is owned by process 0
    print(comm.rank, "Parent range", map.local_range, "Parent local", local_indices, "Child range", submap.local_range,
          "Child ghosts", submap.ghosts, "Parent ghosts", map.ghosts, "Sub owners", sub_owners)
    assert np.allclose(dest_ranks, owners)

    # First rank has no elements
    if comm.rank == 0:
        assert(submap.size_local == 0)
    # Check that rank on sub-process ghosts is the same as the parent map
    for (owner, pos) in zip(sub_owners, ghosts_pos):
        assert(owners[pos] == owner)
    # print(
    #     f"{myrank}:, {submap.ghosts}, {sub_owners}, {map.ghosts[ghosts_pos]} {np.array(owners, dtype=np.int32)[ghosts_pos]}")
