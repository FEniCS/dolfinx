# Copyright (C) 2024 Chris Richardson
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from mpi4py import MPI

import numpy as np

from dolfinx.cpp.io import read_vtkhdf_mesh, write_vtkhdf_mesh
from dolfinx.mesh import CellType, create_unit_cube, create_unit_square


def test_read_write_vtkhdf_mesh2d():
    mesh = create_unit_square(MPI.COMM_WORLD, 5, 5, dtype=np.float64)
    write_vtkhdf_mesh("example2d.vtkhdf", mesh._cpp_object)
    mesh2 = read_vtkhdf_mesh(MPI.COMM_WORLD, "example2d.vtkhdf")

    assert mesh.topology.index_map(2).size_global == mesh2.topology.index_map(2).size_global


def test_read_write_vtkhdf_mesh3d():
    mesh = create_unit_cube(MPI.COMM_WORLD, 5, 5, 5, dtype=np.float64, cell_type=CellType.prism)
    write_vtkhdf_mesh("example3d.vtkhdf", mesh._cpp_object)
    mesh2 = read_vtkhdf_mesh(MPI.COMM_WORLD, "example3d.vtkhdf")

    assert mesh.topology.index_map(3).size_global == mesh2.topology.index_map(3).size_global
