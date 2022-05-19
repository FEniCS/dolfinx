# Copyright (C) 2021 Jørgen S. Dokken and Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import numpy as np

import dolfinx
from dolfinx.mesh import GhostMode, create_unit_square

from mpi4py import MPI


def test_sub_index_map():
    comm = MPI.COMM_WORLD
    my_rank = comm.rank

    # Create index map with one ghost from each other process
    n = 7
    assert comm.size < n + 1
    map_local_size = np.math.factorial(n)

    # The ghosts added are the ith ghost from the ith process relative
    # to the current rank, i.e. rank 0 contains the first index of rank
    # 2, second of rank 3 etc. rank 1 contains the first index of rank
    # 0, the second of rank 2 etc.
    # Ghost one index from from every other rank
    dest_ranks = np.delete(np.arange(0, comm.size, dtype=np.int32), my_rank)
    map_ghosts = np.array([map_local_size * dest_ranks[r] + r %
                          map_local_size for r in range(len(dest_ranks))], dtype=np.int64)
    src_ranks = dest_ranks

    # Create index map
    map = dolfinx.common.IndexMap(comm, map_local_size, [dest_ranks, src_ranks], map_ghosts, src_ranks)
    assert map.size_global == map_local_size * comm.size

    # Build list for each rank of the first (myrank + myrank % 2) local
    # indices
    submap_local_size = [int((rank + rank % 2)) for rank in range(comm.size)]
    local_indices = [np.arange(submap_local_size[rank], dtype=np.int32) for rank in range(comm.size)]

    # Create sub index map and a map from the ghost position in new map
    # to the position in old map
    submap, ghosts_pos_sub = map.create_submap(local_indices[my_rank])

    # Check local and global sizes
    assert submap.size_local == submap_local_size[my_rank]
    assert submap.size_global == sum([rank + rank % 2 for rank in range(comm.size)])

    # Check that first rank has no elements
    if comm.rank == 0:
        assert submap.size_local == 0

    # Check that rank on sub-process ghosts is the same as the parent
    # map
    owners = map.owners
    assert (dest_ranks == owners).all()

    subowners = submap.owners
    assert (owners[ghosts_pos_sub] == subowners).all()

    # Check that ghost indices are correct in submap
    # NOTE This assumes size_local is the same for all ranks
    # TODO Consider renaming to something shorter
    submap_global_to_map_global_map = np.concatenate([local_indices[rank] + map_local_size * rank
                                                      for rank in range(comm.size)])
    # FIXME Do this more elegantly
    submap_ghosts = []
    for map_ghost in map.ghosts:
        submap_ghost = np.where(submap_global_to_map_global_map == map_ghost)[0]
        if submap_ghost.size != 0:
            submap_ghosts.append(submap_ghost[0])
    assert np.allclose(submap.ghosts, submap_ghosts)


def test_sub_index_map_ghost_mode_none():
    n = 2
    mesh = create_unit_square(MPI.COMM_WORLD, n, n, ghost_mode=GhostMode.none)
    tdim = mesh.topology.dim
    map = mesh.topology.index_map(tdim)
    submap_indices = range(0, min(2, map.size_local))
    map.create_submap(submap_indices)
