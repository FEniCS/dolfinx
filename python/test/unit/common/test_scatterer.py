from mpi4py import MPI

import numpy as np
import pytest

import dolfinx
from dolfinx import cpp as _cpp


@pytest.mark.parametrize("dtype", [np.int64, np.float64, np.float32])
def test_scatter_forward(dtype):
    """Test forward and reverse scatter."""
    comm = MPI.COMM_WORLD

    # Create an index map with shared entries across all processes
    local_size = 50
    dest = np.delete(np.arange(0, comm.size, dtype=np.int32), comm.rank)
    map_ghosts = np.array(
        [local_size * dest[r] + r % local_size for r in range(len(dest))], dtype=np.int64
    )
    src = dest
    map = dolfinx.common.IndexMap(comm, local_size, [dest, src], map_ghosts, src)
    assert map.size_global == local_size * comm.size

    sc = _cpp.common.Scatterer(map, 1)
    v = np.zeros((map.size_local + map.num_ghosts), dtype=dtype)

    # Fill local part with rank of this process and scatter
    v[: map.size_local] = comm.rank
    assert np.all(v[map.size_local :] == 0)
    sc.scatter_fwd(v, v[map.size_local :])
    # Received values should match the owners in the index map
    assert np.all(v[map.size_local :] == map.owners)


@pytest.mark.parametrize("dtype", [np.int64, np.float32, np.float64])
def test_scatter_reverse(dtype):
    """Test forward and reverse scatter."""
    comm = MPI.COMM_WORLD

    # Create an index map sharing first entry with other processes
    local_size = 50
    dest = np.delete(np.arange(0, comm.size, dtype=np.int32), comm.rank)
    map_ghosts = np.array([local_size * dest[r] for r in range(len(dest))], dtype=np.int64)
    src = dest
    map = dolfinx.common.IndexMap(comm, local_size, [dest, src], map_ghosts, src)
    assert map.size_global == local_size * comm.size

    # Fill ghost part with ones and reverse scatter
    sc = _cpp.common.Scatterer(map, 1)
    v = np.zeros((local_size + map.num_ghosts), dtype=dtype)
    v[local_size:] = 1

    sc.scatter_rev(v, v[local_size:])
    assert sum(v[:local_size]) == comm.size - 1
