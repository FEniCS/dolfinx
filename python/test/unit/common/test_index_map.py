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

    n = 7
    assert comm.size < n + 1
    size_local = np.math.factorial(n)

    # FIXME: document which indices are ghosted
    # Ghost one index from from every other rank
    dest_ranks = np.delete(np.arange(0, comm.size, dtype=np.int32), myrank)
    ghosts = np.array([size_local * dest_ranks[r] + r % size_local for r in range(len(dest_ranks))])
    print(ghosts)
    src_ranks = dest_ranks

    # Create index map
    map = dolfinx.cpp.common.IndexMap(comm, size_local, dest_ranks, ghosts, src_ranks)
    assert map.size_global == size_local * comm.size

    owners = map.ghost_owner_rank()

    # Build list of the first (myrank + myrank % 2) local indices
    local_size_sub = int((myrank + myrank % 2))
    local_indices = np.arange(local_size_sub, dtype=np.int32)
    # print(f"myrank = {myrank}, local_indices = {local_indices}, test = {map.local_to_global(local_indices)}")

    # Create sub-index map, and map from ghost posittion in new map to
    # position in old map
    submap, ghosts_pos_sub = map.create_submap(local_indices)

    # Check local and global sizes
    assert submap.size_local == local_size_sub
    assert submap.size_global == sum([rank + rank % 2 for rank in range(comm.size)])

    # Running this code on 2 procs gives you that the ghost on process 0
    # is owned by process 0
    # print(comm.rank, "Range", map.local_range, "Local", local_indices,
    #       "Sub range", submap.local_range, "Sub ghosts", submap.ghosts,
    #       "Ghosts", map.ghosts, "Sub owners", submap.ghost_owner_rank())
    # print(f"myrank = {myrank}, map.local_range = {map.local_range}, submap.local_range = {submap.local_range}, submap.ghosts={submap.ghosts}, map.ghosts = {map.ghosts}, submap.ghost_owner_rank() = {submap.ghost_owner_rank()}")
    assert (dest_ranks == owners).all()

    # Check that first rank has no elements
    if comm.rank == 0:
        assert submap.size_local == 0

    # Check that rank on sub-process ghosts is the same as the parent
    # map
    assert (owners[ghosts_pos_sub] == submap.ghost_owner_rank()).all()

    # print(f"{myrank}: , {submap.ghosts}, {owners_sub},\
    #       {map.ghosts[ghosts_pos_sub]} {np.array(owners, dtype=np.int32)[ghosts_pos_sub]}")


def my_test_sub_index_map():
    comm = MPI.COMM_WORLD
    myrank = comm.rank

    # Create index map with one ghost from each other process
    n = 7
    assert comm.size < n + 1
    size_local = np.math.factorial(n)

    # FIXME: document which indices are ghosted
    # Ghost one index from from every other rank
    dest_ranks = np.delete(np.arange(0, comm.size, dtype=np.int32), myrank)
    ghosts = np.array([size_local * dest_ranks[r] + r % size_local for r in range(len(dest_ranks))])
    src_ranks = dest_ranks

    # Create index map
    map = dolfinx.cpp.common.IndexMap(comm, size_local, dest_ranks, ghosts, src_ranks)
    assert map.size_global == size_local * comm.size

    # Build list of the first (myrank + myrank % 2) local indices
    local_size_sub = [int((rank + rank % 2)) for rank in range(comm.size)]
    local_indices = [np.arange(local_size_sub[rank], dtype=np.int32) for rank in range(comm.size)]

    # Create sub-index map, and map from ghost posittion in new map to
    # position in old map
    submap, ghosts_pos_sub = map.create_submap(local_indices[myrank])
    # print(f"{myrank} {submap.global_indices()}")

    # Check local and global sizes
    assert submap.size_local == local_size_sub[myrank]
    assert submap.size_global == sum([rank + rank % 2 for rank in range(comm.size)])

    owners = map.ghost_owner_rank()
    assert (dest_ranks == owners).all()

    # Check that first rank has no elements
    if comm.rank == 0:
        assert submap.size_local == 0

    # Check that rank on sub-process ghosts is the same as the parent
    # map
    assert (owners[ghosts_pos_sub] == submap.ghost_owner_rank()).all()
    
    # print(f"myrank = {myrank}, map.local_range = {map.local_range}, submap.local_range = {submap.local_range}, map.ghosts = {map.ghosts}, submap.ghosts={submap.ghosts}, map.ghost_owner_rank() = {map.ghost_owner_rank()}, submap.ghost_owner_rank() = {submap.ghost_owner_rank()}, submap.global_indices() = {submap.global_indices()}")

    # Check that ghost indices are correct in submap
    # NOTE This assumes size_local is the same for all ranks
    submap_global_to_map_global_map = np.concatenate([local_indices[rank] + size_local * rank
                                                      for rank in range(comm.size)])
    submap_ghosts = []
    for map_ghost in map.ghosts:
        submap_ghost = np.where(submap_global_to_map_global_map == map_ghost)[0]
        if submap_ghost.size != 0:
            submap_ghosts.append(submap_ghost[0])
    assert np.allclose(submap.ghosts, submap_ghosts)


my_test_sub_index_map()
