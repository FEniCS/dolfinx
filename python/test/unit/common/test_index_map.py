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
    assert comm.size < n + 1
    size_local = np.math.factorial(n)
    local_range = (size_local * myrank, size_local * (myrank + 1))

    # FIXME: document which indices are ghosted
    # Ghost one index from from every other rank
    dest_ranks = np.delete(np.arange(0, comm.size, dtype=np.int32), myrank)
    ghosts = np.array([size_local * dest_ranks[r] + r % size_local for r in range(len(dest_ranks))])
    src_ranks = dest_ranks

    # print(ghosts, myrank)
    # print(dest_ranks, myrank)

    # Create index map
    map = dolfinx.cpp.common.IndexMap(comm, size_local, dest_ranks, ghosts, src_ranks)
    assert map.size_global == size_local * comm.size

    owners = map.ghost_owner_rank()

    print("\n")
    print("Ghosts", ghosts, myrank)

    # Build list of the first (myrank + myrank % 2) local indices
    local_size_sub = int((myrank + myrank % 2))
    local_indices = np.arange(local_size_sub, dtype=np.int32)
    print(local_indices, local_range, myrank)

    # Create sub-index map, and map from ghost posittion in new map to
    # position in old map
    submap, ghosts_pos_sub = map.create_submap(local_indices)

    # Check local and global sizes
    assert submap.size_local == local_size_sub
    assert submap.size_global == sum([rank + rank % 2 for rank in range(comm.size)])

    owners_sub = submap.ghost_owner_rank()
    # print("owners_sub", owners_sub)

    # Running this code on 2 procs gives you that the ghost on process 0
    # is owned by process 0
    # print(comm.rank, "Range", map.local_range, "Local", local_indices,
    #       "Sub range", submap.local_range, "Sub ghosts", submap.ghosts, "Ghosts", map.ghosts, "Sub owners", owners_sub)
    assert np.allclose(dest_ranks, owners)

    # Check that first rank has no elements
    if comm.rank == 0:
        assert submap.size_local == 0

    # Check that rank on sub-process ghosts is the same as the parent map
    assert (owners[ghosts_pos_sub] == owners_sub).all()

    if myrank == 1:
        # print(type(owners))
        # print(ghosts_pos_sub.shape)
        # print(owners[ghosts_pos_sub])
        print(owners[ghosts_pos_sub], owners_sub)

    # print(f"{myrank}: , {submap.ghosts}, {owners_sub},\
    #       {map.ghosts[ghosts_pos_sub]} {np.array(owners, dtype=np.int32)[ghosts_pos_sub]}")
