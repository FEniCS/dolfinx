# Copyright (C) 2024 Chris Richardson
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

# Fixtures specific to dolfinx unit tests

from mpi4py import MPI

import numpy as np
import pytest

from dolfinx.cpp.mesh import create_mesh
from dolfinx.fem import coordinate_element
from dolfinx.mesh import CellType, GhostMode, create_cell_partitioner


@pytest.fixture
def mixed_topology_mesh():
    # Create a mesh
    nx = 8
    ny = 8
    nz = 8
    n_cells = nx * ny * nz

    cells: list = [[], [], [], []]
    orig_idx: list = [[], [], [], []]
    geom = []

    if MPI.COMM_WORLD.rank == 0:
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

            if iz < nz / 2:
                if (ix < nx / 2 and iy < ny / 2) or (ix >= nx / 2 and iy >= ny / 2):
                    cells[0] += [v0, v1, v2, v3, v4, v5, v6, v7]
                    orig_idx[0] += [idx]
                    idx += 1
                else:
                    cells[1] += [v0, v1, v3, v4, v5, v7]
                    orig_idx[1] += [idx]
                    idx += 1
                    cells[1] += [v0, v2, v3, v4, v6, v7]
                    orig_idx[1] += [idx]
                    idx += 1
            else:
                if (iy < ny / 2 and ix >= nx / 2) or (iy >= ny / 2 and ix < nx / 2):
                    cells[2] += [v0, v1, v3, v7]
                    orig_idx[2] += [idx]
                    idx += 1
                    cells[2] += [v0, v1, v7, v5]
                    orig_idx[2] += [idx]
                    idx += 1
                    cells[2] += [v0, v5, v7, v4]
                    orig_idx[2] += [idx]
                    idx += 1
                    cells[2] += [v0, v3, v2, v7]
                    orig_idx[2] += [idx]
                    idx += 1
                    cells[2] += [v0, v6, v4, v7]
                    orig_idx[2] += [idx]
                    idx += 1
                    cells[2] += [v0, v2, v6, v7]
                    orig_idx[2] += [idx]
                    idx += 1
                else:
                    cells[3] += [v0, v1, v2, v3, v7]
                    orig_idx[3] += [idx]
                    idx += 1
                    cells[3] += [v0, v1, v4, v5, v7]
                    orig_idx[3] += [idx]
                    idx += 1
                    cells[3] += [v0, v2, v4, v6, v7]
                    orig_idx[3] += [idx]
                    idx += 1

        n_points = (nx + 1) * (ny + 1) * (nz + 1)
        sqxy = (nx + 1) * (ny + 1)
        for v in range(n_points):
            iz = v // sqxy
            p = v % sqxy
            iy = p // (nx + 1)
            ix = p % (nx + 1)
            geom += [[ix / nx, iy / ny, iz / nz]]

    cells_np = [np.array(c) for c in cells]
    geomx = np.array(geom, dtype=np.float64)
    if len(geom) == 0:
        geomx = np.empty((0, 3), dtype=np.float64)
    else:
        geomx = np.array(geom, dtype=np.float64)

    cell_types = [CellType.hexahedron, CellType.prism, CellType.tetrahedron, CellType.pyramid]
    coordinate_elements = [coordinate_element(cell, 1) for cell in cell_types]
    part = create_cell_partitioner(GhostMode.none)
    return create_mesh(
        MPI.COMM_WORLD, cells_np, [e._cpp_object for e in coordinate_elements], geomx, part
    )
