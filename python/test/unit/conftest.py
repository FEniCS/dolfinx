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

# Values a test argument is parametrized over when it does not say otherwise,
# so that the common case needs no decorator. Use the marks in unit.marks for
# anything else; an explicit parametrize also wins.
_defaults = {
    "dtype": [np.float32, np.float64],
    "cell_type": [
        CellType.triangle,
        CellType.quadrilateral,
        CellType.tetrahedron,
        CellType.hexahedron,
    ],
    "ghost_mode": [GhostMode.none, GhostMode.shared_facet],
}


def _parametrized(metafunc, name):
    """Check whether a test parametrizes ``name`` itself."""
    for marker in metafunc.definition.iter_markers("parametrize"):
        if marker.args and name in [a.strip() for a in str(marker.args[0]).split(",")]:
            return True
    return False


def pytest_generate_tests(metafunc):
    for arg, values in _defaults.items():
        if arg in metafunc.fixturenames and not _parametrized(metafunc, arg):
            metafunc.parametrize(arg, values)


@pytest.fixture
def xtype(dtype):
    """Real type underlying ``dtype``, e.g. the mesh geometry type."""
    return dtype(0).real.dtype


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
                    # Pyramids x 2 + tets x 2 (quads on top/bottom of cube,
                    # triangles on the 4 side faces).
                    cells[3] += [v0, v1, v2, v3, v7]  # base z=0, apex V7
                    orig_idx[3] += [idx]
                    idx += 1
                    cells[3] += [v4, v5, v6, v7, v0]  # base z=1, apex V0
                    orig_idx[3] += [idx]
                    idx += 1
                    cells[2] += [v0, v2, v6, v7]
                    orig_idx[2] += [idx]
                    idx += 1
                    cells[2] += [v0, v1, v5, v7]
                    orig_idx[2] += [idx]
                    idx += 1

        n_points = (nx + 1) * (ny + 1) * (nz + 1)
        sqxy = (nx + 1) * (ny + 1)
        for v in range(n_points):
            iz = v // sqxy
            p = v % sqxy
            iy = p // (nx + 1)
            ix = p % (nx + 1)
            geom += [[ix / nx, iy / ny, iz / nz]]

    cells_np = [np.array(c, dtype=np.int64) for c in cells]
    geomx = np.array(geom, dtype=np.float64)
    if len(geom) == 0:
        geomx = np.empty((0, 3), dtype=np.float64)
    else:
        geomx = np.array(geom, dtype=np.float64)

    cell_types = [CellType.hexahedron, CellType.prism, CellType.tetrahedron, CellType.pyramid]
    coordinate_elements = [coordinate_element(cell, 1) for cell in cell_types]
    part = create_cell_partitioner(GhostMode.none, 2)
    max_cells_per_facet = 2
    return create_mesh(
        MPI.COMM_WORLD,
        cells_np,
        [e._cpp_object for e in coordinate_elements],
        geomx,
        part,
        max_cells_per_facet,
    )
