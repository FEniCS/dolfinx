# Copyright (C) 2024 Chris Richardson
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from mpi4py import MPI

import numpy as np

from dolfinx.io.vtkhdf import read_mesh, write_mesh
from dolfinx.mesh import CellType, Mesh, create_unit_cube, create_unit_square


def test_read_write_vtkhdf_mesh2d():
    mesh = create_unit_square(MPI.COMM_WORLD, 5, 5, dtype=np.float32)
    write_mesh("example2d.vtkhdf", mesh)
    mesh2 = read_mesh(MPI.COMM_WORLD, "example2d.vtkhdf", np.float32)
    assert mesh2.geometry.x.dtype == np.float32
    mesh2 = read_mesh(MPI.COMM_WORLD, "example2d.vtkhdf", np.float64)
    assert mesh2.geometry.x.dtype == np.float64
    assert mesh.topology.index_map(2).size_global == mesh2.topology.index_map(2).size_global


def test_read_write_vtkhdf_mesh3d():
    mesh = create_unit_cube(MPI.COMM_WORLD, 5, 5, 5, cell_type=CellType.prism)
    write_mesh("example3d.vtkhdf", mesh)
    mesh2 = read_mesh(MPI.COMM_WORLD, "example3d.vtkhdf")

    assert mesh.topology.index_map(3).size_global == mesh2.topology.index_map(3).size_global


def test_read_write_mixed_topology(mixed_topology_mesh):
    mesh = Mesh(mixed_topology_mesh, None)
    write_mesh("mixed_mesh.vtkhdf", mesh)

    mesh2 = read_mesh(MPI.COMM_WORLD, "mixed_mesh.vtkhdf", np.float64)
    for t in mesh2.topology.entity_types[-1]:
        assert t in mesh.topology.entity_types[-1]
