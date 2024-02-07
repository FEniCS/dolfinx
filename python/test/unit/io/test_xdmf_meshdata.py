# Copyright (C) 2012-2019 Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from pathlib import Path

from mpi4py import MPI

import numpy as np
import pytest

from dolfinx import default_real_type
from dolfinx.io import XDMFFile
from dolfinx.mesh import CellType, create_unit_cube, create_unit_interval, create_unit_square

# Supported XDMF file encoding
if MPI.COMM_WORLD.size > 1:
    encodings = [XDMFFile.Encoding.HDF5]
else:
    encodings = [XDMFFile.Encoding.HDF5, XDMFFile.Encoding.ASCII]

celltypes_2D = [CellType.triangle, CellType.quadrilateral]
celltypes_3D = [CellType.tetrahedron, CellType.hexahedron]


def mesh_factory(tdim, n):
    if tdim == 1:
        return create_unit_interval(MPI.COMM_WORLD, n)
    elif tdim == 2:
        return create_unit_square(MPI.COMM_WORLD, n, n)
    elif tdim == 3:
        return create_unit_cube(MPI.COMM_WORLD, n, n, n)


@pytest.mark.skipif(default_real_type != np.float64, reason="float32 not supported yet")
@pytest.mark.parametrize("tdim", [2, 3])
@pytest.mark.parametrize("n", [6])
def test_read_mesh_data(tempdir, tdim, n):
    filename = Path(tempdir, "mesh.xdmf")
    mesh = mesh_factory(tdim, n)
    encoding = XDMFFile.Encoding.HDF5
    with XDMFFile(mesh.comm, filename, "w", encoding) as file:
        file.write_mesh(mesh)

    with XDMFFile(MPI.COMM_WORLD, filename, "r") as file:
        cell_shape, cell_degree = file.read_cell_type()
        cells = file.read_topology_data()
        x = file.read_geometry_data()

    assert cell_shape == mesh.topology.cell_type
    assert cell_degree == 1
    assert mesh.topology.index_map(tdim).size_global == mesh.comm.allreduce(
        cells.shape[0], op=MPI.SUM
    )
    assert mesh.geometry.index_map().size_global == mesh.comm.allreduce(x.shape[0], op=MPI.SUM)
