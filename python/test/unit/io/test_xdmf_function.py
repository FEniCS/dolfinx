# Copyright (C) 2012-2019 Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from pathlib import Path

import numpy as np
import pytest

from dolfinx.fem import (Function, FunctionSpace, TensorFunctionSpace,
                         VectorFunctionSpace)
from dolfinx.io import XDMFFile
from dolfinx.mesh import (CellType, create_unit_cube, create_unit_interval,
                          create_unit_square)

from mpi4py import MPI

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


# --- Function


@pytest.mark.parametrize("use_pathlib", [True, False])
@pytest.mark.parametrize("encoding", encodings)
@pytest.mark.parametrize("dtype", [np.double, np.complex128])
def test_save_1d_scalar(tempdir, encoding, dtype, use_pathlib):
    xtype = np.real(dtype(0)).dtype
    filename2 = (Path(tempdir).joinpath("u1_.xdmf")if use_pathlib else Path(tempdir, "u1_.xdmf"))
    mesh = create_unit_interval(MPI.COMM_WORLD, 32, dtype=xtype)
    V = FunctionSpace(mesh, ("Lagrange", 2))
    u = Function(V, dtype=dtype)
    u.x.set(1.0 + (1j if np.issubdtype(dtype, np.complexfloating) else 0))
    with XDMFFile(mesh.comm, filename2, "w", encoding=encoding) as file:
        file.write_mesh(mesh)
        file.write_function(u)


@pytest.mark.parametrize("cell_type", celltypes_2D)
@pytest.mark.parametrize("encoding", encodings)
@pytest.mark.parametrize("dtype", [np.double, np.complex128])
def test_save_2d_scalar(tempdir, encoding, dtype, cell_type):
    xtype = np.real(dtype(0)).dtype
    filename = Path(tempdir, "u2.xdmf")
    mesh = create_unit_square(MPI.COMM_WORLD, 12, 12, cell_type, dtype=xtype)
    V = FunctionSpace(mesh, ("Lagrange", 2))
    u = Function(V, dtype=dtype)
    u.x.set(1.0)
    with XDMFFile(mesh.comm, filename, "w", encoding=encoding) as file:
        file.write_mesh(mesh)
        file.write_function(u)


@pytest.mark.parametrize("cell_type", celltypes_3D)
@pytest.mark.parametrize("encoding", encodings)
@pytest.mark.parametrize("dtype", [np.double, np.complex128])
def test_save_3d_scalar(tempdir, encoding, dtype, cell_type):
    xtype = np.real(dtype(0)).dtype
    filename = Path(tempdir, "u3.xdmf")
    mesh = create_unit_cube(MPI.COMM_WORLD, 4, 3, 4, cell_type, dtype=xtype)
    V = FunctionSpace(mesh, ("Lagrange", 2))
    u = Function(V, dtype=dtype)
    u.x.set(1.0)
    with XDMFFile(mesh.comm, filename, "w", encoding=encoding) as file:
        file.write_mesh(mesh)
        file.write_function(u)


@pytest.mark.parametrize("cell_type", celltypes_2D)
@pytest.mark.parametrize("encoding", encodings)
@pytest.mark.parametrize("dtype", [np.double, np.complex128])
def test_save_2d_vector(tempdir, encoding, dtype, cell_type):
    xtype = np.real(dtype(0)).dtype
    filename = Path(tempdir, "u_2dv.xdmf")
    mesh = create_unit_square(MPI.COMM_WORLD, 12, 13, cell_type, dtype=xtype)
    V = VectorFunctionSpace(mesh, ("Lagrange", 2))
    u = Function(V, dtype=dtype)
    u.x.set(1.0 + (1j if np.issubdtype(dtype, np.complexfloating) else 0))
    with XDMFFile(mesh.comm, filename, "w", encoding=encoding) as file:
        file.write_mesh(mesh)
        file.write_function(u)


@pytest.mark.parametrize("cell_type", celltypes_3D)
@pytest.mark.parametrize("encoding", encodings)
@pytest.mark.parametrize("dtype", [np.double, np.complex128])
def test_save_3d_vector(tempdir, encoding, dtype, cell_type):
    xtype = np.real(dtype(0)).dtype
    filename = Path(tempdir, "u_3Dv.xdmf")
    mesh = create_unit_cube(MPI.COMM_WORLD, 2, 2, 2, cell_type, dtype=xtype)
    u = Function(VectorFunctionSpace(mesh, ("Lagrange", 1)), dtype=dtype)
    u.x.set(1.0 + (1j if np.issubdtype(dtype, np.complexfloating) else 0))
    with XDMFFile(mesh.comm, filename, "w", encoding=encoding) as file:
        file.write_mesh(mesh)
        file.write_function(u)


@pytest.mark.parametrize("cell_type", celltypes_2D)
@pytest.mark.parametrize("encoding", encodings)
@pytest.mark.parametrize("dtype", [np.double, np.complex128])
def test_save_2d_tensor(tempdir, encoding, dtype, cell_type):
    xtype = np.real(dtype(0)).dtype
    filename = Path(tempdir, "tensor.xdmf")
    mesh = create_unit_square(MPI.COMM_WORLD, 16, 16, cell_type, dtype=xtype)
    u = Function(TensorFunctionSpace(mesh, ("Lagrange", 2)), dtype=dtype)
    u.x.set(1.0 + (1j if np.issubdtype(dtype, np.complexfloating) else 0))
    with XDMFFile(mesh.comm, filename, "w", encoding=encoding) as file:
        file.write_mesh(mesh)
        file.write_function(u)


@pytest.mark.parametrize("cell_type", celltypes_3D)
@pytest.mark.parametrize("encoding", encodings)
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_save_3d_tensor(tempdir, encoding, dtype, cell_type):
    xtype = np.real(dtype(0)).dtype
    filename = Path(tempdir, "u3t.xdmf")
    mesh = create_unit_cube(MPI.COMM_WORLD, 4, 4, 4, cell_type, dtype=xtype)
    u = Function(TensorFunctionSpace(mesh, ("Lagrange", 2)), dtype=dtype)
    u.x.set(1.0 + (1j if np.issubdtype(dtype, np.complexfloating) else 0))
    with XDMFFile(mesh.comm, filename, "w", encoding=encoding) as file:
        file.write_mesh(mesh)
        file.write_function(u)


@pytest.mark.parametrize("cell_type", celltypes_3D)
@pytest.mark.parametrize("encoding", encodings)
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_save_3d_vector_series(tempdir, encoding, dtype, cell_type):
    filename = Path(tempdir, "u_3D.xdmf")
    xtype = np.real(dtype(0)).dtype
    mesh = create_unit_cube(MPI.COMM_WORLD, 2, 2, 2, cell_type, dtype=xtype)
    u = Function(VectorFunctionSpace(mesh, ("Lagrange", 2)), dtype=dtype)
    with XDMFFile(mesh.comm, filename, "w", encoding=encoding) as file:
        file.write_mesh(mesh)
        u.x.set(1.0 + (1j if np.issubdtype(dtype, np.complexfloating) else 0))
        file.write_function(u, 0.1)
        u.x.set(2.0 + (2j if np.issubdtype(dtype, np.complexfloating) else 0))
        file.write_function(u, 0.2)

    with XDMFFile(mesh.comm, filename, "a", encoding=encoding) as file:
        u.x.set(3.0 + (3j if np.issubdtype(dtype, np.complexfloating) else 0))
        file.write_function(u, 0.3)
