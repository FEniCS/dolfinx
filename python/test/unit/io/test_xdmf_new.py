# Copyright (C) 2012-2019 Garth N. Wells
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import os

import numpy as np
import pytest
from dolfinx_utils.test.fixtures import tempdir

from dolfinx import (MPI, Function, FunctionSpace, MeshFunction,
                     TensorFunctionSpace, UnitCubeMesh, UnitIntervalMesh,
                     UnitSquareMesh, VectorFunctionSpace, cpp,
                     has_petsc_complex)
from dolfinx.cpp.mesh import CellType
from dolfinx.io import XDMFFileNew

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


# @pytest.mark.parametrize("encoding", encodings)
# def test_save_and_load_1d_mesh(tempdir, encoding):
#     filename = os.path.join(tempdir, "mesh.xdmf")
#     mesh = UnitIntervalMesh(MPI.comm_world, 32, new_style=True)
#     with XDMFFileNew(mesh.mpi_comm(), filename, encoding=encoding) as file:
#         file.write(mesh)
#     with XDMFFileNew(MPI.comm_world, filename) as file:
#         mesh2 = file.read_mesh()
#     assert mesh.topology.index_map(0).size_global == mesh2.topology.index_map(0).size_global
#     dim = mesh.topology.dim
#     assert mesh.topology.index_map(dim).size_global == mesh2.topology.index_map(dim).size_global


@pytest.mark.parametrize("cell_type", celltypes_2D)
@pytest.mark.parametrize("encoding", encodings)
def test_save_and_load_mesh2D(tempdir, encoding, cell_type):
    filename = os.path.join(tempdir, "mesh.xdmf")
    mesh = UnitSquareMesh(MPI.comm_world, 12, 12, cell_type, new_style=True)
    with XDMFFileNew(mesh.mpi_comm(), filename, encoding=encoding) as file:
        file.write(mesh)

    with XDMFFileNew(MPI.comm_world, filename) as file:
        mesh2 = file.read_mesh()
    assert mesh.topology.index_map(0).size_global == mesh2.topology.index_map(0).size_global
    dim = mesh.topology.dim
    assert mesh.topology.index_map(dim).size_global == mesh2.topology.index_map(dim).size_global


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


@pytest.mark.parametrize("encoding", encodings)
def test_save_1d_scalar(tempdir, encoding):
    filename2 = os.path.join(tempdir, "u1_.xdmf")
    mesh = UnitIntervalMesh(MPI.comm_world, 32, new_style=True)
    V = FunctionSpace(mesh, ("Lagrange", 2))
    u = Function(V)
    u.vector.set(1.0 + (1j if has_petsc_complex else 0))
    with XDMFFileNew(mesh.mpi_comm(), filename2, encoding=encoding) as file:
        file.write(u)


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


@pytest.mark.parametrize("cell_type", celltypes_3D)
@pytest.mark.parametrize("encoding", encodings)
def test_save_3d_scalar(tempdir, encoding, cell_type):
    filename = os.path.join(tempdir, "u3.xdmf")
    mesh = UnitCubeMesh(MPI.comm_world, 4, 3, 4, cell_type, new_style=True)
    V = FunctionSpace(mesh, ("Lagrange", 2))
    u = Function(V)
    u.vector.set(1.0)
    with XDMFFileNew(mesh.mpi_comm(), filename, encoding=encoding) as file:
        file.write(u)


@pytest.mark.parametrize("cell_type", celltypes_2D)
@pytest.mark.parametrize("encoding", encodings)
def test_save_2d_vector(tempdir, encoding, cell_type):
    filename = os.path.join(tempdir, "u_2dv.xdmf")
    mesh = UnitSquareMesh(MPI.comm_world, 16, 9, cell_type, new_style=True)
    V = VectorFunctionSpace(mesh, ("Lagrange", 2))
    u = Function(V)
    u.vector.set(1.0 + (1j if has_petsc_complex else 0))
    with XDMFFileNew(mesh.mpi_comm(), filename, encoding=encoding) as file:
        file.write(u)


@pytest.mark.parametrize("cell_type", celltypes_3D)
@pytest.mark.parametrize("encoding", encodings)
def test_save_3d_vector(tempdir, encoding, cell_type):
    filename = os.path.join(tempdir, "u_3Dv.xdmf")
    mesh = UnitCubeMesh(MPI.comm_world, 2, 2, 2, cell_type, new_style=True)
    u = Function(VectorFunctionSpace(mesh, ("Lagrange", 1)))
    u.vector.set(1.0 + (1j if has_petsc_complex else 0))
    with XDMFFileNew(mesh.mpi_comm(), filename, encoding=encoding) as file:
        file.write(u)


@pytest.mark.parametrize("cell_type", celltypes_3D)
@pytest.mark.parametrize("encoding", encodings)
def test_save_3d_vector_series(tempdir, encoding, cell_type):
    filename = os.path.join(tempdir, "u_3D.xdmf")
    mesh = UnitCubeMesh(MPI.comm_world, 2, 2, 2, cell_type, new_style=True)
    u = Function(VectorFunctionSpace(mesh, ("Lagrange", 2)))
    with XDMFFileNew(mesh.mpi_comm(), filename, encoding=encoding) as file:
        u.vector.set(1.0 + (1j if has_petsc_complex else 0))
        file.write(u, 0.1)
        u.vector.set(2.0 + (2j if has_petsc_complex else 0))
        file.write(u, 0.2)
        u.vector.set(3.0 + (3j if has_petsc_complex else 0))
        file.write(u, 0.3)


@pytest.mark.parametrize("cell_type", celltypes_2D)
@pytest.mark.parametrize("encoding", encodings)
def test_save_2d_tensor(tempdir, encoding, cell_type):
    filename = os.path.join(tempdir, "tensor.xdmf")
    mesh = UnitSquareMesh(MPI.comm_world, 16, 16, cell_type, new_style=True)
    u = Function(TensorFunctionSpace(mesh, ("Lagrange", 2)))
    u.vector.set(1.0 + (1j if has_petsc_complex else 0))
    with XDMFFileNew(mesh.mpi_comm(), filename, encoding=encoding) as file:
        file.write(u)


@pytest.mark.parametrize("cell_type", celltypes_3D)
@pytest.mark.parametrize("encoding", encodings)
def test_save_3d_tensor(tempdir, encoding, cell_type):
    filename = os.path.join(tempdir, "u3t.xdmf")
    mesh = UnitCubeMesh(MPI.comm_world, 4, 4, 4, cell_type, new_style=True)
    u = Function(TensorFunctionSpace(mesh, ("Lagrange", 2)))
    u.vector.set(1.0 + (1j if has_petsc_complex else 0))
    with XDMFFileNew(mesh.mpi_comm(), filename, encoding=encoding) as file:
        file.write(u)

# --- mesh data


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

# --- MeshFunctions


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
    num_vertices = cpp.mesh.cell_num_vertices(cell_type)
    for c in range(num_cells):
        dofs = x_dofmap.links(c)
        v = [dofs[i] for i in range(num_vertices)]
        x_r = np.linalg.norm(np.sum(x[v], axis=0) / len(v))
        if (x_r < 1):
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
        v = [dofs[i] for i in range(num_vertices)]
        x_r = np.linalg.norm(np.sum(x[v], axis=0) / len(v))
        if (x_r < 1.0):
            assert mf_in.values[c] == -1
        else:
            assert mf_in.values[c] == 1


# @pytest.mark.parametrize("cell_type", celltypes_3D)
# @pytest.mark.parametrize("encoding", encodings)
# @pytest.mark.parametrize("data_type", data_types)
# def test_save_2D_cell_function(tempdir, encoding, data_type, cell_type):
#     dtype_str, dtype = data_type
#     filename = os.path.join(tempdir, "mf_2D_{}.xdmf".format(dtype_str))
#     filename_msh = os.path.join(tempdir, "mf_2D_{}-mesh.xdmf".format(dtype_str))
#     mesh = UnitCubeMesh(MPI.comm_world, 7, 11, 8, cell_type, new_style=True)
#     mf = MeshFunction(dtype_str, mesh, mesh.topology.dim, 0)
#     mf.name = "cells"

#     tdim = mesh.topology.dim
#     map = mesh.topology.index_map(tdim)
#     num_cells = map.size_local + map.num_ghosts
#     mf.values[:] = np.arange(num_cells, dtype=dtype) + map.local_range[0]
#     x = mesh.geometry.x
#     x_dofmap = mesh.geometry.dofmap()
#     for c in range(num_cells):
#         dofs = x_dofmap.links(c)
#         x_mid = (x[dofs[0]] + x[dofs[1]] + x[dofs[2]]) / 3.0
#         if (x_mid < 0.49):
#             mf.values[c] = -1
#         else:
#             mf.values[c] = 1

#     # NOTE: We need to write the mesh and mesh function to handle
#     # re-odering of indices
#     # Write mesh and mesh function
#     with XDMFFileNew(mesh.mpi_comm(), filename_msh, encoding=encoding) as file:
#         file.write(mesh)
#     with XDMFFileNew(mesh.mpi_comm(), filename, encoding=encoding) as file:
#         file.write(mf)

#     with XDMFFileNew(mesh.mpi_comm(), filename_msh) as xdmf:
#         mesh2 = xdmf.read_mesh()
#     with XDMFFileNew(mesh.mpi_comm(), filename) as xdmf:
#         read_function = getattr(xdmf, "read_mf_" + dtype_str)
#         mf_in = read_function(mesh2, "cells")

#     map = mesh2.topology.index_map(tdim)
#     num_cells = map.size_local + map.num_ghosts
#     x = mesh2.geometry.x
#     x_dofmap = mesh2.geometry.dofmap()
#     for c in range(num_cells):
#         dofs = x_dofmap.links(c)
#         x_mid = (x[dofs[0], 0] + x[dofs[1], 0] + x[dofs[2], 0]) / 3.0
#         if (x_mid < 0.49):
#             assert mf_in.values[c] == -1
#         else:
#             assert mf_in.values[c] == 1
