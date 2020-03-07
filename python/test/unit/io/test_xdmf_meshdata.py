# Copyright (C) 2012-2019 Garth N. Wells
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import os

import pytest

from dolfinx import MPI, UnitCubeMesh, UnitIntervalMesh, UnitSquareMesh
from dolfinx.cpp.mesh import CellType
from dolfinx.io import XDMFFileNew
from dolfinx_utils.test.fixtures import tempdir

assert (tempdir)

# Supported XDMF file encoding
if MPI.size(MPI.comm_world) > 1:
    encodings = (XDMFFileNew.Encoding.HDF5, )
else:
    encodings = (XDMFFileNew.Encoding.HDF5, XDMFFileNew.Encoding.ASCII)
    encodings = (XDMFFileNew.Encoding.HDF5, )

celltypes_2D = [CellType.triangle, CellType.quadrilateral]
celltypes_3D = [CellType.tetrahedron, CellType.hexahedron]


def mesh_factory(tdim, n):
    if tdim == 1:
        return UnitIntervalMesh(MPI.comm_world, n)
    elif tdim == 2:
        return UnitSquareMesh(MPI.comm_world, n, n)
    elif tdim == 3:
        return UnitCubeMesh(MPI.comm_world, n, n, n)


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
    encoding = XDMFFileNew.Encoding.HDF5
    with XDMFFileNew(mesh.mpi_comm(), filename, encoding) as file:
        file.write(mesh)

    with XDMFFileNew(MPI.comm_world, filename) as file:
        cell_type, x, cells = file.read_mesh_data()

    assert cell_type == mesh.topology.cell_type
    assert mesh.topology.index_map(tdim).size_global == MPI.sum(mesh.mpi_comm(), cells.shape[0])
    assert mesh.geometry.index_map().size_global == MPI.sum(mesh.mpi_comm(), x.shape[0])
