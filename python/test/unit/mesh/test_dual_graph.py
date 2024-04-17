from mpi4py import MPI

import numpy as np

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
    w = mesh.build_dual_graph(MPI.COMM_WORLD, mesh.CellType.interval, to_adj(cells, np.int64))
    assert w.num_nodes == 3
    for i in range(w.num_nodes):
        assert len(w.links(i)) == 2
