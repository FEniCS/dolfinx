# Copyright (C) 2012-2019 Garth N. Wells
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import os

import numpy as np
import pytest

from dolfinx import (MPI, Function, FunctionSpace, MeshFunction, UnitCubeMesh,
                     UnitIntervalMesh, UnitSquareMesh, cpp)
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
        return UnitIntervalMesh(MPI.comm_world, n, new_style=True)
    elif tdim == 2:
        return UnitSquareMesh(MPI.comm_world, n, n, new_style=True)
    elif tdim == 3:
        return UnitCubeMesh(MPI.comm_world, n, n, n, new_style=True)


@pytest.fixture
def worker_id(request):
    """Return worker ID when using pytest-xdist to run tests in parallel"""
    if hasattr(request.config, 'slaveinput'):
        return request.config.slaveinput['slaveid']
    else:
        return 'master'


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


@pytest.mark.parametrize("tdim", [2, 3])
@pytest.mark.parametrize("n", [6, 10])
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


# encodings = (XDMFFileNew.Encoding.ASCII,)
# celltypes_2D = [CellType.triangle, CellType.quadrilateral]
# celltypes_2D = [CellType.quadrilateral]
# celltypes_2D = [CellType.quadrilateral]
data_types = (('int', int), )


@pytest.mark.parametrize("cell_type", celltypes_2D)
@pytest.mark.parametrize("encoding", encodings)
@pytest.mark.parametrize("data_type", data_types)
def test_save_2D_cell_function(tempdir, encoding, data_type, cell_type):
    dtype_str, dtype = data_type
    filename = os.path.join(tempdir, "mf_2D_{}.xdmf".format(dtype_str))
    filename_msh = os.path.join(tempdir, "mf_2D_{}-mesh.xdmf".format(dtype_str))
    mesh = UnitSquareMesh(MPI.comm_world, 22, 11, cell_type, new_style=True)

    # print(mesh.topology.connectivity(2,0))
    # print(mesh.geometry.dofmap())
    # print(mesh.geometry.x)
    # return

    mf = MeshFunction(dtype_str, mesh, mesh.topology.dim, 0)
    mf.name = "cells"

    tdim = mesh.topology.dim
    map = mesh.topology.index_map(tdim)
    num_cells = map.size_local + map.num_ghosts
    mf.values[:] = np.arange(num_cells, dtype=dtype) + map.local_range[0]
    x = mesh.geometry.x
    x_dofmap = mesh.geometry.dofmap()
    for c in range(num_cells):
        dofs = x_dofmap.links(c)
        x_mid = (x[dofs[0], 0] + x[dofs[1], 0] + x[dofs[2], 0]) / 3.0
        if (x_mid < 0.49):
            mf.values[c] = -1
        else:
            mf.values[c] = 1

    # NOTE: We need to write the mesh and mesh function to handle
    # re-odering of indices
    # Write mesh and mesh function
    with XDMFFileNew(mesh.mpi_comm(), filename_msh, encoding=encoding) as file:
        file.write(mesh)
    with XDMFFileNew(mesh.mpi_comm(), filename, encoding=encoding) as file:
        file.write(mf)

    with XDMFFileNew(mesh.mpi_comm(), filename_msh) as xdmf:
        mesh2 = xdmf.read_mesh()
    with XDMFFileNew(mesh.mpi_comm(), filename) as xdmf:
        read_function = getattr(xdmf, "read_mf_" + dtype_str)
        mf_in = read_function(mesh2, "cells")

    map = mesh2.topology.index_map(tdim)
    num_cells = map.size_local + map.num_ghosts
    x = mesh2.geometry.x
    x_dofmap = mesh2.geometry.dofmap()
    for c in range(num_cells):
        dofs = x_dofmap.links(c)
        x_mid = (x[dofs[0], 0] + x[dofs[1], 0] + x[dofs[2], 0]) / 3.0
        if (x_mid < 0.49):
            assert mf_in.values[c] == -1
        else:
            assert mf_in.values[c] == 1
