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
@pytest.mark.parametrize("cell_type", [CellType.tetrahedron, CellType.hexahedron])
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
    c = 6 if cell_type == CellType.tetrahedron else 1
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
    def partitioner(comm, n, m, topo):
        r = comm.Get_rank()
        dests = []
        offsets = [0]
        for i in range(topo.num_nodes):
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
