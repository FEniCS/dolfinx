# Copyright (C) 2012-2019 Garth N. Wells
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import os

import pytest

from dolfinx import (MPI, Function, FunctionSpace, UnitCubeMesh,
                     UnitSquareMesh, cpp)
from dolfinx.cpp.mesh import CellType
from dolfinx.io import XDMFFileNew
from dolfinx_utils.test.fixtures import tempdir

assert (tempdir)

# Supported XDMF file encoding
if MPI.size(MPI.comm_world) > 1:
    encodings = (XDMFFileNew.Encoding.HDF5, )
else:
    encodings = (XDMFFileNew.Encoding.HDF5, XDMFFileNew.Encoding.ASCII)
    encodings = (XDMFFileNew.Encoding.HDF5,)

celltypes_2D = [CellType.triangle, CellType.quadrilateral]
celltypes_3D = [CellType.tetrahedron, CellType.hexahedron]


@pytest.fixture
def worker_id(request):
    """Return worker ID when using pytest-xdist to run tests in parallel"""
    if hasattr(request.config, 'slaveinput'):
        return request.config.slaveinput['slaveid']
    else:
        return 'master'


# celltypes_2D = [CellType.triangle]
# encodings = (XDMFFileNew.Encoding.HDF5, )
@pytest.mark.parametrize("cell_type", celltypes_2D)
@pytest.mark.parametrize("encoding", encodings)
def test_save_and_load_mesh2D(tempdir, encoding, cell_type):
    filename = os.path.join(tempdir, "mesh.xdmf")
    mesh = UnitSquareMesh(MPI.comm_world, 12, 12, cell_type, new_style=True)
    with XDMFFileNew(mesh.mpi_comm(), filename, encoding=encoding) as file:
        file.write(mesh)

    with XDMFFileNew(MPI.comm_world, filename) as file:
        mesh2 = file.read_mesh()
    assert mesh.topology.index_map(0).size_global == mesh2.topology.index_map(
        0).size_global
    dim = mesh.topology.dim
    assert mesh.topology.index_map(
        dim).size_global == mesh2.topology.index_map(dim).size_global


@pytest.mark.parametrize("cell_type", celltypes_3D)
@pytest.mark.parametrize("encoding", encodings)
def test_save_and_load_mesh3D(tempdir, encoding, cell_type):
    filename = os.path.join(tempdir, "mesh.xdmf")
    mesh = UnitCubeMesh(MPI.comm_world, 12, 12, 8, cell_type, new_style=True)
    with XDMFFileNew(mesh.mpi_comm(), filename, encoding=encoding) as file:
        file.write(mesh)

    with XDMFFileNew(MPI.comm_world, filename) as file:
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
                                              cpp.mesh.GhostMode.none,
                                              new_style=True)

    filename = os.path.join(tempdir, "tri6_mesh.xdmf")
    with XDMFFileNew(mesh.mpi_comm(), filename, encoding=encoding) as xdmf:
        xdmf.write(mesh)

    with XDMFFileNew(mesh.mpi_comm(), filename) as xdmf:
        mesh2 = xdmf.read_mesh()

    assert mesh.topology.index_map(0).size_global == mesh2.topology.index_map(
        0).size_global
    dim = mesh.topology.dim
    assert mesh.topology.index_map(
        dim).size_global == mesh2.topology.index_map(dim).size_global


@pytest.mark.parametrize("cell_type", celltypes_2D)
@pytest.mark.parametrize("encoding", encodings)
def test_save_2d_scalar(tempdir, encoding, cell_type):
    filename = os.path.join(tempdir, "u2.xdmf")
    mesh = UnitSquareMesh(MPI.comm_world, 12, 12, cell_type, new_style=True)
    V = FunctionSpace(mesh, ("Lagrange", 2))
    u = Function(V)
    u.vector.set(1.0)
    with XDMFFileNew(mesh.mpi_comm(), filename, encoding=encoding) as file:
        file.write(u)
