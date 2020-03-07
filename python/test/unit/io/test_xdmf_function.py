# Copyright (C) 2012-2019 Garth N. Wells
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import os

import pytest

from dolfinx import (MPI, Function, FunctionSpace,
                     TensorFunctionSpace, UnitCubeMesh, UnitIntervalMesh,
                     UnitSquareMesh, VectorFunctionSpace, has_petsc_complex)
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


# --- Function


@pytest.mark.parametrize("encoding", encodings)
def test_save_1d_scalar(tempdir, encoding):
    filename2 = os.path.join(tempdir, "u1_.xdmf")
    mesh = UnitIntervalMesh(MPI.comm_world, 32)
    V = FunctionSpace(mesh, ("Lagrange", 2))
    u = Function(V)
    u.vector.set(1.0 + (1j if has_petsc_complex else 0))
    with XDMFFileNew(mesh.mpi_comm(), filename2, encoding=encoding) as file:
        file.write(u)


@pytest.mark.parametrize("cell_type", celltypes_2D)
@pytest.mark.parametrize("encoding", encodings)
def test_save_2d_scalar(tempdir, encoding, cell_type):
    filename = os.path.join(tempdir, "u2.xdmf")
    mesh = UnitSquareMesh(MPI.comm_world, 12, 12, cell_type)
    V = FunctionSpace(mesh, ("Lagrange", 2))
    u = Function(V)
    u.vector.set(1.0)
    with XDMFFileNew(mesh.mpi_comm(), filename, encoding=encoding) as file:
        file.write(u)


@pytest.mark.parametrize("cell_type", celltypes_3D)
@pytest.mark.parametrize("encoding", encodings)
def test_save_3d_scalar(tempdir, encoding, cell_type):
    filename = os.path.join(tempdir, "u3.xdmf")
    mesh = UnitCubeMesh(MPI.comm_world, 4, 3, 4, cell_type)
    V = FunctionSpace(mesh, ("Lagrange", 2))
    u = Function(V)
    u.vector.set(1.0)
    with XDMFFileNew(mesh.mpi_comm(), filename, encoding=encoding) as file:
        file.write(u)


@pytest.mark.parametrize("cell_type", celltypes_2D)
@pytest.mark.parametrize("encoding", encodings)
def test_save_2d_vector(tempdir, encoding, cell_type):
    filename = os.path.join(tempdir, "u_2dv.xdmf")
    mesh = UnitSquareMesh(MPI.comm_world, 12, 13, cell_type)
    V = VectorFunctionSpace(mesh, ("Lagrange", 2))
    u = Function(V)
    u.vector.set(1.0 + (1j if has_petsc_complex else 0))
    with XDMFFileNew(mesh.mpi_comm(), filename, encoding=encoding) as file:
        file.write(u)


@pytest.mark.parametrize("cell_type", celltypes_3D)
@pytest.mark.parametrize("encoding", encodings)
def test_save_3d_vector(tempdir, encoding, cell_type):
    filename = os.path.join(tempdir, "u_3Dv.xdmf")
    mesh = UnitCubeMesh(MPI.comm_world, 2, 2, 2, cell_type)
    u = Function(VectorFunctionSpace(mesh, ("Lagrange", 1)))
    u.vector.set(1.0 + (1j if has_petsc_complex else 0))
    with XDMFFileNew(mesh.mpi_comm(), filename, encoding=encoding) as file:
        file.write(u)


@pytest.mark.parametrize("cell_type", celltypes_2D)
@pytest.mark.parametrize("encoding", encodings)
def test_save_2d_tensor(tempdir, encoding, cell_type):
    filename = os.path.join(tempdir, "tensor.xdmf")
    mesh = UnitSquareMesh(MPI.comm_world, 16, 16, cell_type)
    u = Function(TensorFunctionSpace(mesh, ("Lagrange", 2)))
    u.vector.set(1.0 + (1j if has_petsc_complex else 0))
    with XDMFFileNew(mesh.mpi_comm(), filename, encoding=encoding) as file:
        file.write(u)


@pytest.mark.parametrize("cell_type", celltypes_3D)
@pytest.mark.parametrize("encoding", encodings)
def test_save_3d_tensor(tempdir, encoding, cell_type):
    filename = os.path.join(tempdir, "u3t.xdmf")
    mesh = UnitCubeMesh(MPI.comm_world, 4, 4, 4, cell_type)
    u = Function(TensorFunctionSpace(mesh, ("Lagrange", 2)))
    u.vector.set(1.0 + (1j if has_petsc_complex else 0))
    with XDMFFileNew(mesh.mpi_comm(), filename, encoding=encoding) as file:
        file.write(u)


@pytest.mark.parametrize("cell_type", celltypes_3D)
@pytest.mark.parametrize("encoding", encodings)
def test_save_3d_vector_series(tempdir, encoding, cell_type):
    filename = os.path.join(tempdir, "u_3D.xdmf")
    mesh = UnitCubeMesh(MPI.comm_world, 2, 2, 2, cell_type)
    u = Function(VectorFunctionSpace(mesh, ("Lagrange", 2)))
    with XDMFFileNew(mesh.mpi_comm(), filename, encoding=encoding) as file:
        u.vector.set(1.0 + (1j if has_petsc_complex else 0))
        file.write(u, 0.1)
        u.vector.set(2.0 + (2j if has_petsc_complex else 0))
        file.write(u, 0.2)
        u.vector.set(3.0 + (3j if has_petsc_complex else 0))
        file.write(u, 0.3)
