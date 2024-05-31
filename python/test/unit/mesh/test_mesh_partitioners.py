# Copyright (C) 2020 Igor A. Baratta
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from pathlib import Path

from mpi4py import MPI

import numpy as np
import pytest

import dolfinx
import dolfinx.graph
import ufl
from basix.ufl import element
from dolfinx import default_real_type
from dolfinx.cpp.mesh import cell_num_vertices
from dolfinx.io import XDMFFile
from dolfinx.mesh import (
    CellType,
    GhostMode,
    compute_midpoints,
    create_box,
    create_cell_partitioner,
    create_mesh,
)

partitioners = [dolfinx.graph.partitioner()]
try:
    from dolfinx.graph import partitioner_scotch

    partitioners.append(partitioner_scotch())
except ImportError:
    partitioners.append(
        pytest.param(None, marks=pytest.mark.skip(reason="DOLFINx build without SCOTCH"))
    )
try:
    from dolfinx.graph import partitioner_parmetis

    partitioners.append(partitioner_parmetis())
except ImportError:
    partitioners.append(
        pytest.param(None, marks=pytest.mark.skip(reason="DOLFINx built without Parmetis"))
    )
try:
    from dolfinx.graph import partitioner_kahip

    partitioners.append(partitioner_kahip())
except ImportError:
    partitioners.append(
        pytest.param(None, marks=pytest.mark.skip(reason="DOLFINx built without KaHiP"))
    )


@pytest.mark.parametrize("gpart", partitioners)
@pytest.mark.parametrize("Nx", [5, 10])
@pytest.mark.parametrize("cell_type", [CellType.tetrahedron, CellType.hexahedron, CellType.prism])
def test_partition_box_mesh(gpart, Nx, cell_type):
    part = create_cell_partitioner(gpart)
    mesh = create_box(
        MPI.COMM_WORLD,
        [np.array([0, 0, 0]), np.array([1, 1, 1])],
        [Nx, Nx, Nx],
        cell_type,
        ghost_mode=GhostMode.shared_facet,
        partitioner=part,
    )
    tdim = mesh.topology.dim
    c = {CellType.tetrahedron: 6, CellType.prism: 2, CellType.hexahedron: 1}[cell_type]
    assert mesh.topology.index_map(tdim).size_global == Nx**3 * c
    assert mesh.topology.index_map(tdim).size_local != 0
    assert mesh.topology.index_map(0).size_global == (Nx + 1) ** 3


@pytest.mark.skipif(default_real_type != np.float64, reason="float32 not supported yet")
@pytest.mark.parametrize("Nx", [3, 10, 13])
@pytest.mark.parametrize("cell_type", [CellType.tetrahedron, CellType.hexahedron])
def test_custom_partitioner(tempdir, Nx, cell_type):
    mpi_comm = MPI.COMM_WORLD
    Lx = mpi_comm.size
    points = [np.array([0, 0, 0]), np.array([Lx, Lx, Lx])]
    mesh = create_box(mpi_comm, points, [Nx, Nx, Nx], cell_type, ghost_mode=GhostMode.shared_facet)

    filename = Path(tempdir, "u1_.xdmf")
    with XDMFFile(mpi_comm, filename, "w") as file:
        file.write_mesh(mesh)

    # Read all geometry data on all processes
    with XDMFFile(MPI.COMM_SELF, filename, "r") as file:
        x_global = file.read_geometry_data()

    # Read topology data
    with XDMFFile(MPI.COMM_WORLD, filename, "r") as file:
        cell_shape, cell_degree = file.read_cell_type()
        x = file.read_geometry_data()
        topo = file.read_topology_data()

    num_local_coor = x.shape[0]
    all_sizes = mpi_comm.allgather(num_local_coor)
    all_sizes.insert(0, 0)
    all_ranges = np.cumsum(all_sizes)

    # Testing the premise: coordinates are read contiguously in chunks
    rank = mpi_comm.rank
    assert np.all(x_global[all_ranges[rank] : all_ranges[rank + 1]] == x)

    domain = ufl.Mesh(element("Lagrange", cell_shape.name, cell_degree, shape=(3,)))

    # Partition mesh in layers, capture geometrical data and topological
    # data from outer scope
    def partitioner(*args):
        midpoints = np.mean(x_global[topo], axis=1)
        dest = np.floor(midpoints[:, 0] % mpi_comm.size).astype(np.int32)
        return dolfinx.cpp.graph.AdjacencyList_int32(dest)

    new_mesh = create_mesh(mpi_comm, topo, x, domain, partitioner)

    tdim = new_mesh.topology.dim
    assert (
        mesh.topology.index_map(tdim).size_global == new_mesh.topology.index_map(tdim).size_global
    )
    num_cells = new_mesh.topology.index_map(tdim).size_local
    new_mesh.topology.create_connectivity(tdim, tdim)
    cell_midpoints = compute_midpoints(new_mesh, tdim, np.arange(num_cells))
    assert num_cells > 0
    assert np.all(cell_midpoints[:, 0] >= mpi_comm.rank)
    assert np.all(cell_midpoints[:, 0] <= mpi_comm.rank + 1)


def test_asymmetric_partitioner():
    mpi_comm = MPI.COMM_WORLD
    n = mpi_comm.Get_size()
    r = mpi_comm.Get_rank()
    domain = ufl.Mesh(element("Lagrange", "triangle", 1, shape=(2,)))

    # Create a simple triangle mesh with a strip on each process
    topo = []
    for i in range(10):
        j = i * (n + 1) + r
        k = (i + 1) * (n + 1) + r
        topo += [[j, j + 1, k]]
        topo += [[j + 1, k, k + 1]]
    topo = np.array(topo, dtype=int)

    # Dummy geometry
    if r == 0:
        x = np.zeros((11 * (n + 1), 2), dtype=np.float64)
    else:
        x = np.zeros((0, 2), dtype=np.float64)

    # Send cells to self, and if on process 1, also send to process 0.
    def partitioner(comm, n, cell_types, topo):
        r = comm.Get_rank()
        dests = []
        offsets = [0]
        num_cells = len(topo[0]) // cell_num_vertices(cell_types[0])
        for i in range(num_cells):
            dests.append(r)
            if r == 1:
                dests.append(0)
            offsets.append(len(dests))

        dests = np.array(dests, dtype=np.int32)
        offsets = np.array(offsets, dtype=np.int32)
        return dolfinx.cpp.graph.AdjacencyList_int32(dests, offsets)

    new_mesh = create_mesh(mpi_comm, topo, x, domain, partitioner)
    if r == 0 and n > 1:
        assert new_mesh.topology.index_map(2).num_ghosts == 20
    else:
        assert new_mesh.topology.index_map(2).num_ghosts == 0


def test_mixed_topology_partitioning():
    nx = 16
    ny = 16
    nz = 16
    n_cells = nx * ny * nz
    cells = [[], [], []]
    orig_idx = [[], [], []]
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
        if ix < nx // 2:
            cells[0] += [v0, v1, v2, v3, v4, v5, v6, v7]
            orig_idx[0] += [idx]
            idx += 1
        elif ix == nx // 2:
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

    if MPI.COMM_WORLD.rank == 0:
        cells_np = [np.array(c) for c in cells]
    else:
        cells_np = [np.zeros(0) for c in cells]

    nparts = 4
    part = create_cell_partitioner(GhostMode.none)
    p = part(
        MPI.COMM_WORLD,
        nparts,
        [CellType.hexahedron, CellType.pyramid, CellType.tetrahedron],
        cells_np,
    )

    counts = np.array([sum(p.array == i) for i in range(nparts)])
    count_mpi = MPI.COMM_WORLD.allreduce(counts)

    assert count_mpi.sum() == 14080
