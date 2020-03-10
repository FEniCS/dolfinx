# Copyright (C) 2012-2019 Garth N. Wells
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import os

import pytest

from dolfinx import MPI, UnitCubeMesh, UnitIntervalMesh, UnitSquareMesh, cpp
from dolfinx.cpp.mesh import CellType
from dolfinx.io import XDMFFile
from dolfinx_utils.test.fixtures import tempdir

assert (tempdir)

# Supported XDMF file encoding
if MPI.size(MPI.comm_world) > 1:
    encodings = (XDMFFile.Encoding.HDF5, )
else:
    encodings = (XDMFFile.Encoding.HDF5, XDMFFile.Encoding.ASCII)
    encodings = (XDMFFile.Encoding.HDF5, )

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


# @pytest.mark.parametrize("encoding", encodings)
# def test_save_and_load_1d_mesh(tempdir, encoding):
#     filename = os.path.join(tempdir, "mesh.xdmf")
#     mesh = UnitIntervalMesh(MPI.comm_world, 32)
#     with XDMFFile(mesh.mpi_comm(), filename, encoding=encoding) as file:
#         file.write(mesh)
#     with XDMFFile(MPI.comm_world, filename) as file:
#         mesh2 = file.read_mesh()
#     assert mesh.topology.index_map(0).size_global == mesh2.topology.index_map(0).size_global
#     dim = mesh.topology.dim
#     assert mesh.topology.index_map(dim).size_global == mesh2.topology.index_map(dim).size_global


@pytest.mark.parametrize("cell_type", celltypes_2D)
@pytest.mark.parametrize("encoding", encodings)
def test_save_and_load_mesh2D(tempdir, encoding, cell_type):
    filename = os.path.join(tempdir, "mesh.xdmf")
    mesh = UnitSquareMesh(MPI.comm_world, 12, 12, cell_type)
    with XDMFFile(mesh.mpi_comm(), filename, encoding=encoding) as file:
        file.write(mesh)

    with XDMFFile(MPI.comm_world, filename) as file:
        mesh2 = file.read_mesh()
    assert mesh.topology.index_map(0).size_global == mesh2.topology.index_map(0).size_global
    dim = mesh.topology.dim
    assert mesh.topology.index_map(dim).size_global == mesh2.topology.index_map(dim).size_global


@pytest.mark.parametrize("cell_type", celltypes_3D)
@pytest.mark.parametrize("encoding", encodings)
def test_save_and_load_mesh3D(tempdir, encoding, cell_type):
    filename = os.path.join(tempdir, "mesh.xdmf")
    mesh = UnitCubeMesh(MPI.comm_world, 12, 12, 8, cell_type)
    with XDMFFile(mesh.mpi_comm(), filename, encoding=encoding) as file:
        file.write(mesh)

    with XDMFFile(MPI.comm_world, filename) as file:
        mesh2 = file.read_mesh()
    assert mesh.topology.index_map(0).size_global == mesh2.topology.index_map(
        0).size_global
    dim = mesh.topology.dim
    assert mesh.topology.index_map(
        dim).size_global == mesh2.topology.index_map(dim).size_global


@pytest.mark.parametrize("encoding", encodings)
def test_read_write_p2_mesh(tempdir, encoding):
    mesh = cpp.generation.UnitDiscMesh.create(MPI.comm_world,
                                              3,
                                              cpp.mesh.GhostMode.none)

    filename = os.path.join(tempdir, "tri6_mesh.xdmf")
    with XDMFFile(mesh.mpi_comm(), filename, encoding=encoding) as xdmf:
        xdmf.write(mesh)

    with XDMFFile(mesh.mpi_comm(), filename) as xdmf:
        mesh2 = xdmf.read_mesh()

    assert mesh.topology.index_map(0).size_global == mesh2.topology.index_map(
        0).size_global
    dim = mesh.topology.dim
    assert mesh.topology.index_map(
        dim).size_global == mesh2.topology.index_map(dim).size_global
