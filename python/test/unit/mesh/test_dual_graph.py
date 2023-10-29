
from mpi4py import MPI

from dolfinx import cpp as _cpp
from dolfinx import mesh


def to_adj(cells):
    cflat = []
    coff = [0]
    for c in cells:
        cflat += c
        cc = coff[-1] + len(c)
        coff += [cc]
    adj = _cpp.graph.AdjacencyList_int64(cflat, coff)
    return adj


def test_dgrsph_1d():
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()
    n0 = rank * 3
    x = n0 + 3
    if (rank == size - 1):
        x = 0
    # Circular chain of interval cells
    cells = [[n0, n0 + 1], [n0 + 1, n0 + 2], [n0 + 2, x]]
    w = mesh.build_dual_graph(MPI.COMM_WORLD, to_adj(cells), 1)
    assert w.num_nodes == 3
    for i in range(w.num_nodes):
        assert len(w.links(i)) == 2
