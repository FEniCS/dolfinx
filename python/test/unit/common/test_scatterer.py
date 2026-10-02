# Copyright (C) 2022-2026 Igor Baratta, Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for the Scatterer interface."""

from mpi4py import MPI

import numpy as np
import pytest

from dolfinx.common import index_map, scatterer


@pytest.mark.parametrize("dtype", [np.int64, np.float32, np.float64, np.complex64, np.complex128])
def test_scatter_forward(dtype):
    """Test forward scatter."""
    comm = MPI.COMM_WORLD

    # Create an index map with shared entries across all processes
    local_size = 50
    dest = np.delete(np.arange(0, comm.size, dtype=np.int32), comm.rank)
    map_ghosts = np.array(
        [local_size * dest[r] + r % local_size for r in range(len(dest))], dtype=np.int64
    )
    src = dest
    map = index_map(comm, local_size, (map_ghosts, src), dest_src=[dest, src])
    assert map.size_global == local_size * comm.size

    sc = scatterer(map)
    v = np.zeros((map.size_local + map.num_ghosts), dtype=dtype)

    # Fill local part with rank of this process and scatter
    v[: map.size_local] = comm.rank
    assert np.all(v[map.size_local :] == 0)

    local_idx = sc.local_indices_block
    remote_idx = sc.remote_indices_block
    send_buffer = v[local_idx]
    recv_buffer = np.empty(remote_idx.size, dtype=dtype)
    request = sc.scatter_fwd_begin(send_buffer, recv_buffer, 1)
    sc.scatter_fwd_end(request)
    v[map.size_local :][remote_idx] = recv_buffer

    # Received values should match the owners in the index map
    assert np.all(v[map.size_local :] == map.owners)


@pytest.mark.parametrize("dtype", [np.int64, np.float32, np.float64, np.complex64, np.complex128])
def test_scatter_reverse(dtype):
    """Test reverse scatter."""
    comm = MPI.COMM_WORLD

    # Create an index map sharing first entry with other processes
    local_size = 50
    dest = np.delete(np.arange(0, comm.size, dtype=np.int32), comm.rank)
    map_ghosts = np.array([local_size * dest[r] for r in range(len(dest))], dtype=np.int64)
    src = dest
    map = index_map(comm, local_size, (map_ghosts, src), dest_src=[dest, src])
    assert map.size_global == local_size * comm.size

    # Fill ghost part with ones and reverse scatter
    sc = scatterer(map)
    v = np.zeros((local_size + map.num_ghosts), dtype=dtype)
    v[local_size:] = 1

    local_idx = sc.local_indices_block
    remote_idx = sc.remote_indices_block
    send_buffer = v[local_size:][remote_idx]
    recv_buffer = np.empty(local_idx.size, dtype=dtype)
    request = sc.scatter_rev_begin(send_buffer, recv_buffer, 1)
    sc.scatter_rev_end(request)
    np.add.at(v, local_idx, recv_buffer)

    assert sum(v[:local_size]) == comm.size - 1
