from mpi4py import MPI

import numpy as np

import dolfinx.cpp as _cpp
from dolfinx.cpp.mesh import (
    GhostMode,
    create_mesh,
)
from dolfinx.fem import coordinate_element
from dolfinx.mesh import CellType


def test_create_mixed_mesh():
    nx = 7
    ny = 11
    nz = 8
    n_cells = nx * ny * nz
    cells: list = [[], [], []]
    orig_idx: list = [[], [], []]
    idx = 0
    for i in range(n_cells):
        iz = i // (nx * ny)
        j = i % (nx * ny)
        iy = j // nx
        ix = j % nx

        v0 = (iz * (ny + 1) + iy) * (nx + 1) + ix
        v1 = v0 + 1
        v2 = v0 + (nx + 1)
        v3 = v1 + (nx + 1)
        v4 = v0 + (nx + 1) * (ny + 1)
        v5 = v1 + (nx + 1) * (ny + 1)
        v6 = v2 + (nx + 1) * (ny + 1)
        v7 = v3 + (nx + 1) * (ny + 1)
        if iz < nz // 2:
            cells[0] += [v0, v1, v2, v3, v4, v5, v6, v7]
            orig_idx[0] += [idx]
            idx += 1
        elif iz == nz // 2:
            # pyramid
            cells[1] += [v0, v1, v2, v3, v6]
            orig_idx[1] += [idx]
            idx += 1
            # tet
            cells[2] += [v0, v1, v4, v6]
            cells[2] += [v4, v6, v5, v1]
            cells[2] += [v5, v6, v7, v1]
            cells[2] += [v6, v7, v3, v1]
            orig_idx[2] += [idx, idx + 1, idx + 2, idx + 3]
            idx += 4
        else:
            # tet
            cells[2] += [v0, v1, v2, v6]
            cells[2] += [v1, v2, v3, v6]
            cells[2] += [v0, v1, v4, v6]
            cells[2] += [v4, v6, v5, v1]
            cells[2] += [v5, v6, v7, v1]
            cells[2] += [v6, v7, v3, v1]
            orig_idx[2] += [idx, idx + 1, idx + 2, idx + 3, idx + 4, idx + 5]
            idx += 6

    n_points = (nx + 1) * (ny + 1) * (nz + 1)
    sqxy = (nx + 1) * (ny + 1)
    geom = []
    for v in range(n_points):
        iz = v // sqxy
        p = v % sqxy
        iy = p // (nx + 1)
        ix = p % (nx + 1)
        geom += [[ix / nx, iy / ny, iz / nz]]

    if MPI.COMM_WORLD.rank == 0:
        cells_np = [np.array(c) for c in cells]
        geomx = np.array(geom, dtype=np.float64)
    else:
        cells_np = [np.zeros(0) for c in cells]
        geomx = np.zeros((0, 3), dtype=np.float64)

    hexahedron = coordinate_element(CellType.hexahedron, 1)
    pyramid = coordinate_element(CellType.pyramid, 1, variant=1)
    tetrahedron = coordinate_element(CellType.tetrahedron, 1)

    part = _cpp.mesh.create_cell_partitioner(GhostMode.none)
    mesh = create_mesh(
        MPI.COMM_WORLD,
        cells_np,
        [hexahedron._cpp_object, pyramid._cpp_object, tetrahedron._cpp_object],
        geomx,
        3,
        part,
    )

    entity_types = mesh.topology.entity_types[3]
    assert entity_types[0] == CellType.hexahedron
    assert entity_types[1] == CellType.pyramid
    assert entity_types[2] == CellType.tetrahedron

    for i in range(3):
        assert (
            mesh.topology.connectivity((3, i), (0, 0)).num_nodes
            == mesh.topology.index_maps(3)[i].size_local
        )

    num_cells = sum(mesh.topology.index_maps(3)[i].size_local for i in range(3))
    all_cells = mesh.comm.allreduce(num_cells)

    assert all_cells == 2079
