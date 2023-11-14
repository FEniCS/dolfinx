# Copyright (C) 2021 Jørgen S. Dokken and Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import math

from mpi4py import MPI

import numpy as np

import dolfinx
from dolfinx.mesh import GhostMode, create_unit_square


def test_sub_index_map():
    comm = MPI.COMM_WORLD
    my_rank = comm.rank

    # Create index map with one ghost from each other process
    n = 7
    assert comm.size < n + 1
    map_local_size = math.factorial(n)

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
    n = 3
    mesh = create_unit_square(MPI.COMM_WORLD, n, n, ghost_mode=GhostMode.none)
    tdim = mesh.topology.dim
    map = mesh.topology.index_map(tdim)
    submap_indices = np.arange(0, min(2, map.size_local), dtype=np.int32)
    map.create_submap(submap_indices)


def test_index_map_ghost_lifetime():
    """Test lifetime management of arrays."""
    # Create index map with one ghost from each other process. The
    # ghosts added are the ith ghost from the ith process relative to
    # the current rank, i.e. rank 0 contains the first index of rank 2,
    # second of rank 3 etc. rank 1 contains the first index of rank 0,
    # the second of rank 2 etc. Ghost one index from from every other
    # rank
    comm = MPI.COMM_WORLD
    n = 7
    assert comm.size < n + 1
    local_size = math.factorial(n)
    dest = np.delete(np.arange(0, comm.size, dtype=np.int32), comm.rank)
    map_ghosts = np.array([local_size * dest[r] + r % local_size for r in range(len(dest))], dtype=np.int64)
    src = dest
    map = dolfinx.common.IndexMap(comm, local_size, [dest, src], map_ghosts, src)
    assert map.size_global == local_size * comm.size

    # Test lifetime management
    ghosts = map.ghosts
    assert np.array_equal(ghosts, map_ghosts)
    del map
    assert np.array_equal(ghosts, map_ghosts)


# TODO: Add test for case where more than one two process shares an index
# whose owner changes in the submap
def test_create_submap_connected():
    """
    Test create_submap_conn. The diagram illustrates the case with four
    processes. Original map numbering and connectivity (G indicates a ghost
    index):
    Global    Rank 0    Rank 1    Rank 2    Rank 3
    1 - 0     1 - 0
    | / |     | / |
    3 - 2    3G - 2     0 - 2G
    | / |               | / |
    5 - 4              3G - 1     0 - 2G
    | / |                         | / |
    7 - 6                        3G - 1     0 - 3G
    | / |                                   | / |
    9 - 8                                   2 - 1
    We now create a submap of the "upper triangular" parts to
    get the following:
    Global    Rank 0    Rank 1    Rank 2    Rank 3
    1 - 0     1 - 0
    | /       | /
    2 - 3     2G        0 - 1
    | /                 | /
    4 - 5               2G        0 - 1
    | /                           | /
    6 - 8                         2G        0 - 2
    | /                                     | /
    7                                       1
    """
    comm = MPI.COMM_WORLD

    if comm.size == 1:
        return

    if comm.rank == 0:
        local_size = 3
        ghosts = np.array([local_size], dtype=np.int64)
        owners = np.array([1], dtype=np.int32)
        submap_indices = np.array([0, 1, 3], dtype=np.int32)
    elif comm.rank == comm.size - 1:
        local_size = 3
        ghosts = np.array([2 * comm.rank], dtype=np.int64)
        owners = np.array([comm.rank - 1], dtype=np.int32)
        submap_indices = np.array([0, 3, 2], dtype=np.int32)
    else:
        local_size = 2
        ghosts = np.array([2 * comm.rank, 2 * comm.rank + 3], dtype=np.int64)
        owners = np.array([comm.rank - 1, comm.rank + 1], dtype=np.int32)
        submap_indices = np.array([0, 2, 3], dtype=np.int32)

    imap = dolfinx.common.IndexMap(comm, local_size, ghosts, owners)
    sub_imap, sub_imap_to_imap = imap.create_submap_conn(submap_indices)

    if comm.rank == 0:
        assert sub_imap.size_local == 2
        assert np.array_equal(sub_imap.ghosts, [2])
        assert np.array_equal(sub_imap.owners, [comm.rank + 1])
        assert np.array_equal(sub_imap_to_imap, [0, 1, 3])
    elif comm.rank == comm.size - 1:
        assert sub_imap.size_local == 3
        assert np.array_equal(sub_imap.ghosts, [])
        assert np.array_equal(sub_imap.owners, [])
        assert np.array_equal(sub_imap_to_imap, [0, 2, 3])
    else:
        assert sub_imap.size_local == 2
        assert np.array_equal(sub_imap.ghosts, [2 * (comm.rank + 1)])
        assert np.array_equal(sub_imap.owners, [comm.rank + 1])
        assert np.array_equal(sub_imap_to_imap, [0, 2, 3])

    global_indices = sub_imap.local_to_global(
        np.arange(sub_imap.size_local + sub_imap.num_ghosts,
                  dtype=np.int32))
    assert np.array_equal(
        global_indices, np.arange(comm.rank * 2, comm.rank * 2 + 3))
