
from dolfinx import cpp as _cpp
from dolfinx import mesh
from mpi4py import MPI


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
    w = mesh.build_dual_graph(MPI.COMM_WORLD, to_adj(cells), 1)[0]
    assert w.num_nodes == 3
    for i in range(w.num_nodes):
        assert len(w.links(i)) == 2


def test_dgrsph_2d():
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()
    n0 = rank * 6
    x, y = n0 + 6, n0 + 7
    if (rank == size - 1):
        x, y = 0, 1
    # Chain of triangles and quads
    cells = [[n0, n0 + 1, n0 + 2],
             [n0 + 1, n0 + 2, n0 + 3, n0 + 4],
             [n0 + 2, n0 + 4, n0 + 5],
             [n0 + 4, n0 + 5, x, y]]
    w = mesh.build_dual_graph(MPI.COMM_WORLD, to_adj(cells), 2)[0]
    assert w.num_nodes == 4
    for i in range(w.num_nodes):
        assert len(w.links(i)) == 2


def test_dgrsph_3d():
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()
    n0 = rank * 9
    X, Y, Z = n0 + 9, n0 + 10, n0 + 11
    if (rank == size - 1):
        X, Y, Z = 0, 1, 2
    # Chain of tet-prism-hex-pyramid
    cells = [[n0, n0 + 1, n0 + 2, n0 + 3],
             [n0 + 1, n0 + 2, n0 + 3, X, Y, n0 + 4],
             [n0 + 1, n0 + 2, X, Y, n0 + 5, n0 + 6, n0 + 7, n0 + 8],
             [X, Y, n0 + 7, n0 + 8, Z]]
    w = mesh.build_dual_graph(MPI.COMM_WORLD, to_adj(cells), 3)[0]
    assert w.num_nodes == 4
    for i in range(w.num_nodes):
        assert len(w.links(i)) == 2
