
from mpi4py import MPI

import numpy as np

from dolfinx import cpp as _cpp
from dolfinx import mesh


def to_adj(cells):
    cflat = []
    coff = [0]
    for c in cells:
        cflat += c
        cc = coff[-1] + len(c)
        coff += [cc]
    adj = _cpp.graph.AdjacencyList_int64(np.array(cflat), np.array(coff))
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


def test_dgrsph_2d():
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()
    n0 = rank * 6
    x = n0 + 6
    if (rank == size - 1):
        x = 0
    # Circular chain of quadrilateral cells
    #    (3)---(4)---(5)--- ... (9)---(10)---(11)--- (*)
    #    /     /     /          /     /     /      /
    #   /  0  /  1  /  2       /  3  /  4  /  5   /
    #  /     /     /          /     /     /      /
    # (0)---(1)---(2)--- ... (6)---(7)---(8)--- (*)
    cells = [[n0, n0 + 1, n0 + 3, n0 + 4],
             [n0 + 1, n0 + 2, n0 + 4, n0 + 5],
             [n0 + 2, x, n0 + 5, x + 1]]
    w = mesh.build_dual_graph(MPI.COMM_WORLD, to_adj(cells), 1)
    assert w.num_nodes == 3
    for i in range(w.num_nodes):
        assert len(w.links(i)) == 2


def test_dgrsph_1d_spikes():
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()
    n0 = rank * 6
    x = n0 + 6
    if (rank == size - 1):
        x = 0
    # Circular chain of interval cells, with interval "spikes"
    # (3)   (4)   (5)        (9)    (10)  (11)
    #  |     |     |          |     |     |
    #  |3    |4    |5         |9    |10   |11
    #  |     |     |          |     |     |
    # (0)---(1)---(2)--- ... (6)---(7)---(8)--- (*)
    #     0     1     2      6     7     8
    cells = [[n0, n0 + 1], [n0 + 1, n0 + 2], [n0 + 2, x],
             [n0, n0 + 3], [n0 + 1, n0 + 4], [n0 + 2, n0 + 5]]
    w = mesh.build_dual_graph(MPI.COMM_WORLD, to_adj(cells), 1)
    assert w.num_nodes == 6
    for i in range(0, 3):
        assert len(w.links(i)) == 4  # chain cells connect to 4 neigbour cells
    for i in range(3, 6):
        assert len(w.links(i)) == 2  # spike cells connect to 2 neigbour cells


def test_dgrsph_2d_spikes():
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()
    n0 = rank * 12
    x = n0 + 12
    if (rank == size - 1):
        x = 0
    # Circular chain of quadrilateral cells, with quadrilateral "spikes"
    #     (9)   (10)  (11)
    #    / |    /|    /|
    #   / 3|   /4|   /5|
    #  /   |  /  |  /  |
    # (6)  |(7)  |(8)  |
    #  |  (3)+--(4)+--(5)--- ... (*)
    #  |  /  |  /  |  /          /
    #  | / 0 | / 1 | /  2       /  (2nd rank not shown)
    #  |/    |/    |/          /
    # (0)---(1)---(2)--- ... (*)
    cells = [[n0, n0 + 1, n0 + 3, n0 + 4],
             [n0 + 1, n0 + 2, n0 + 4, n0 + 5],
             [n0 + 2, x, n0 + 5, x + 1],
             [n0, n0 + 3, n0 + 6, n0 + 9],
             [n0 + 1, n0 + 4, n0 + 7, n0 + 10],
             [n0 + 2, n0 + 5, n0 + 8, n0 + 11]]
    w = mesh.build_dual_graph(MPI.COMM_WORLD, to_adj(cells), 1)
    assert w.num_nodes == 6
    for i in range(0, 3):
        assert len(w.links(i)) == 4  # chain cells connect to 4 neigbour cells
    for i in range(3, 6):
        assert len(w.links(i)) == 2  # spike cells connect to 2 neigbour cells
