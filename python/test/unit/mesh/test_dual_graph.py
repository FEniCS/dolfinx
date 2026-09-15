from mpi4py import MPI

import numpy as np
import pytest

from dolfinx import graph, mesh


def to_adj(cells, dtype):
    cflat = []
    coff = [0]
    for c in cells:
        cflat += c
        cc = coff[-1] + len(c)
        coff += [cc]
    adj = graph.adjacencylist(np.array(cflat, dtype=dtype), np.array(coff, dtype=dtype))
    return adj


def test_dgrsph_1d():
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()
    n0 = rank * 3
    x = n0 + 3
    if rank == size - 1:
        x = 0
    # Circular chain of interval cells
    cells = [[n0, n0 + 1], [n0 + 1, n0 + 2], [n0 + 2, x]]
    w = mesh.build_dual_graph(MPI.COMM_WORLD, mesh.CellType.interval, to_adj(cells, np.int64), 2, 1)
    assert w.num_nodes == 3
    for i in range(w.num_nodes):
        assert len(w.links(i)) == 2


def test_build_dual_graph_mismatched_cells_and_celltypes_raises_on_every_rank():
    """build_dual_graph must reject a cells/celltypes length mismatch on every rank.

    Regression test: the check that len(cells) == len(celltypes) used to run
    after an empty-mesh early return, so a rank with zero cells skipped the
    check and entered the collective dual-graph computation while a rank
    with cells raised, causing a hang instead of every rank reporting the
    error uniformly.
    """
    comm = MPI.COMM_WORLD
    # Every rank passes 2 cell types but only 1 cell array -- the mismatch
    # itself, not the cell content, is what must be rejected uniformly.
    if comm.rank == 0:
        cells = [np.array([0, 1, 2], dtype=np.int64)]
    else:
        cells = [np.array([], dtype=np.int64)]
    with pytest.raises(RuntimeError):
        mesh.build_dual_graph(
            comm, [mesh.CellType.triangle, mesh.CellType.quadrilateral], cells, None, 1
        )
