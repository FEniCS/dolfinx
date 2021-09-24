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

    # Create a list of ranks, including my rank

    n = 7
    size_local = np.math.factorial(n)
    assert comm.size < n + 1
    local_range = (size_local * myrank, size_local * (myrank + 1))

    # Pick on ghost from every other rank
    dest_ranks = np.delete(np.arange(0, comm.size, dtype=np.int32), myrank)
    ghosts = np.array([i % size_local + size_local * dest_ranks[i] for i in range(len(dest_ranks))])
    # print(ghosts, myrank)
    src_ranks = dest_ranks
    # print(dest_ranks, myrank)

    map = dolfinx.cpp.common.IndexMap(MPI.COMM_WORLD, size_local, dest_ranks, ghosts, src_ranks)
    assert map.size_global == size_local * comm.size

    print("\n")
    print("Ghosts", ghosts, myrank)

    # Create sub-map with (myrank + myrank % 2) elements
    new_local_size = int((myrank + myrank % 2))
    local_indices = np.arange(new_local_size, dtype=np.int32)
    print(local_indices, local_range, myrank)

    submap, ghosts_pos = map.create_submap(local_indices)

    # Check local and global size
    assert submap.size_local == new_local_size
    assert submap.size_global == sum([rank + rank % 2 for rank in range(comm.size)])

    owners_sub = submap.ghost_owner_rank()
    print(owners_sub)
    owners = map.ghost_owner_rank()

    # Running this code on 2 procs gives you that the ghost on process 0
    # is owned by process 0
    print(comm.rank, "Range", map.local_range, "Local", local_indices,
          "Sub range", submap.local_range, "Sub ghosts", submap.ghosts, "Ghosts", map.ghosts, "Sub owners", owners_sub)
    assert np.allclose(dest_ranks, owners)

    # First rank has no elements
    if comm.rank == 0:
        assert(submap.size_local == 0)
    # Check that rank on sub-process ghosts is the same as the parent map
    for (owner, pos) in zip(owners_sub, ghosts_pos):
        assert(owners[pos] == owner)
    print(f"{myrank}: , {submap.ghosts}, {owners_sub},\
          {map.ghosts[ghosts_pos]} {np.array(owners, dtype=np.int32)[ghosts_pos]}")
