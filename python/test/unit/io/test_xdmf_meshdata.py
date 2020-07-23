# Copyright (C) 2012-2019 Garth N. Wells
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import os

import pytest
from dolfinx import UnitCubeMesh, UnitIntervalMesh, UnitSquareMesh
from dolfinx.cpp.mesh import CellType
from dolfinx.io import XDMFFile
from dolfinx_utils.test.fixtures import tempdir
from mpi4py import MPI

assert (tempdir)

# Supported XDMF file encoding
if MPI.COMM_WORLD.size > 1:
    encodings = (XDMFFile.Encoding.HDF5, )
else:
    encodings = (XDMFFile.Encoding.HDF5, XDMFFile.Encoding.ASCII)
    encodings = (XDMFFile.Encoding.HDF5, )

celltypes_2D = [CellType.triangle, CellType.quadrilateral]
celltypes_3D = [CellType.tetrahedron, CellType.hexahedron]


def mesh_factory(tdim, n):
    if tdim == 1:
        return UnitIntervalMesh(MPI.COMM_WORLD, n)
    elif tdim == 2:
        return UnitSquareMesh(MPI.COMM_WORLD, n, n)
    elif tdim == 3:
        return UnitCubeMesh(MPI.COMM_WORLD, n, n, n)


@pytest.fixture
def worker_id(request):
    """Return worker ID when using pytest-xdist to run tests in parallel"""
    if hasattr(request.config, 'slaveinput'):
        return request.config.slaveinput['slaveid']
    else:
        return 'master'


@pytest.mark.parametrize("tdim", [2, 3])
@pytest.mark.parametrize("n", [6])
def test_read_mesh_data(tempdir, tdim, n):
    filename = os.path.join(tempdir, "mesh.xdmf")
    mesh = mesh_factory(tdim, n)
    encoding = XDMFFile.Encoding.HDF5
    with XDMFFile(mesh.mpi_comm(), filename, "w", encoding) as file:
        file.write_mesh(mesh)

    with XDMFFile(MPI.COMM_WORLD, filename, "r") as file:
        cell_type = file.read_cell_type()
        cells = file.read_topology_data()
        x = file.read_geometry_data()

    assert cell_type[0] == mesh.topology.cell_type
    assert cell_type[1] == 1
    assert mesh.topology.index_map(tdim).size_global == mesh.mpi_comm().allreduce(cells.shape[0], op=MPI.SUM)
    assert mesh.geometry.index_map().size_global == mesh.mpi_comm().allreduce(x.shape[0], op=MPI.SUM)
