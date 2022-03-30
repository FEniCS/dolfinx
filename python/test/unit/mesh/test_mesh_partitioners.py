# Copyright (C) 2020 Igor A. Baratta
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from pathlib import Path

import numpy as np
import pytest

import dolfinx
import dolfinx.graph
import ufl
from dolfinx.io import XDMFFile
from dolfinx.mesh import (CellType, GhostMode, compute_midpoints, create_box,
                          create_cell_partitioner, create_mesh)

from mpi4py import MPI

partitioners = [dolfinx.graph.partitioner()]
try:
    from dolfinx.graph import partitioner_scotch
    partitioners.append(partitioner_scotch())
except ImportError:
    partitioners.append(pytest.param(None, marks=pytest.mark.skip(reason="DOLFINx build without SCOTCH")))
try:
    from dolfinx.graph import partitioner_parmetis
    partitioners.append(partitioner_parmetis())
except ImportError:
    partitioners.append(pytest.param(None, marks=pytest.mark.skip(reason="DOLFINx built without Parmetis")))
try:
    from dolfinx.graph import partitioner_kahip
    partitioners.append(partitioner_kahip())
except ImportError:
    partitioners.append(pytest.param(None, marks=pytest.mark.skip(reason="DOLFINx built without KaHiP")))


@pytest.mark.parametrize("gpart", partitioners)
@pytest.mark.parametrize("Nx", [5, 10])
@pytest.mark.parametrize("cell_type", [CellType.tetrahedron, CellType.hexahedron])
def test_partition_box_mesh(gpart, Nx, cell_type):
    part = create_cell_partitioner(gpart)
    mesh = create_box(MPI.COMM_WORLD, [np.array([0, 0, 0]), np.array([1, 1, 1])], [Nx, Nx, Nx],
                      cell_type, GhostMode.shared_facet, part)
    tdim = mesh.topology.dim
    c = 6 if cell_type == CellType.tetrahedron else 1
    assert mesh.topology.index_map(tdim).size_global == Nx**3 * c
    assert mesh.topology.index_map(tdim).size_local != 0
    assert mesh.topology.index_map(0).size_global == (Nx + 1)**3


@pytest.mark.parametrize("Nx", [3, 10, 13])
@pytest.mark.parametrize("cell_type", [CellType.tetrahedron, CellType.hexahedron])
def test_custom_partitioner(tempdir, Nx, cell_type):
    mpi_comm = MPI.COMM_WORLD
    Lx = mpi_comm.size
    points = [np.array([0, 0, 0]), np.array([Lx, Lx, Lx])]
    mesh = create_box(mpi_comm, points, [Nx, Nx, Nx], cell_type, GhostMode.shared_facet)

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
    assert (np.all(x_global[all_ranges[rank]:all_ranges[rank + 1]] == x))

    cell = ufl.Cell(cell_shape.name)
    domain = ufl.Mesh(ufl.VectorElement("Lagrange", cell, cell_degree))

    # Partition mesh in layers, capture geometrical data and topological
    # data from outer scope
    def partitioner(*args):
        midpoints = np.mean(x_global[topo], axis=1)
        dest = np.floor(midpoints[:, 0] % mpi_comm.size).astype(np.int32)
        return dolfinx.cpp.graph.AdjacencyList_int32(dest)

    ghost_mode = GhostMode.none
    new_mesh = create_mesh(mpi_comm, topo, x, domain, ghost_mode, partitioner)

    tdim = new_mesh.topology.dim
    assert mesh.topology.index_map(tdim).size_global == new_mesh.topology.index_map(tdim).size_global
    num_cells = new_mesh.topology.index_map(tdim).size_local
    cell_midpoints = compute_midpoints(new_mesh, tdim, range(num_cells))
    assert num_cells > 0
    assert np.all(cell_midpoints[:, 0] >= mpi_comm.rank)
    assert np.all(cell_midpoints[:, 0] <= mpi_comm.rank + 1)
