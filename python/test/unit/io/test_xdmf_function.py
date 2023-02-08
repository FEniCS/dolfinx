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
from petsc4py import PETSc

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
def test_save_1d_scalar(tempdir, encoding, use_pathlib):
    filename2 = (
        Path(tempdir).joinpath("u1_.xdmf")
        if use_pathlib else Path(tempdir, "u1_.xdmf")
    )
    mesh = create_unit_interval(MPI.COMM_WORLD, 32)
    V = FunctionSpace(mesh, ("Lagrange", 2))
    u = Function(V)
    u.vector.set(1.0 + (1j if np.issubdtype(PETSc.ScalarType, np.complexfloating) else 0))
    with XDMFFile(mesh.comm, filename2, "w", encoding=encoding) as file:
        file.write_mesh(mesh)
        file.write_function(u)


@pytest.mark.parametrize("cell_type", celltypes_2D)
@pytest.mark.parametrize("encoding", encodings)
def test_save_2d_scalar(tempdir, encoding, cell_type):
    filename = Path(tempdir, "u2.xdmf")
    mesh = create_unit_square(MPI.COMM_WORLD, 12, 12, cell_type)
    V = FunctionSpace(mesh, ("Lagrange", 2))
    u = Function(V)
    u.vector.set(1.0)
    with XDMFFile(mesh.comm, filename, "w", encoding=encoding) as file:
        file.write_mesh(mesh)
        file.write_function(u)


@pytest.mark.parametrize("cell_type", celltypes_3D)
@pytest.mark.parametrize("encoding", encodings)
def test_save_3d_scalar(tempdir, encoding, cell_type):
    filename = Path(tempdir, "u3.xdmf")
    mesh = create_unit_cube(MPI.COMM_WORLD, 4, 3, 4, cell_type)
    V = FunctionSpace(mesh, ("Lagrange", 2))
    u = Function(V)
    u.vector.set(1.0)
    with XDMFFile(mesh.comm, filename, "w", encoding=encoding) as file:
        file.write_mesh(mesh)
        file.write_function(u)


@pytest.mark.parametrize("cell_type", celltypes_2D)
@pytest.mark.parametrize("encoding", encodings)
def test_save_2d_vector(tempdir, encoding, cell_type):
    filename = Path(tempdir, "u_2dv.xdmf")
    mesh = create_unit_square(MPI.COMM_WORLD, 12, 13, cell_type)
    V = VectorFunctionSpace(mesh, ("Lagrange", 2))
    u = Function(V)
    u.vector.set(1.0 + (1j if np.issubdtype(PETSc.ScalarType, np.complexfloating) else 0))
    with XDMFFile(mesh.comm, filename, "w", encoding=encoding) as file:
        file.write_mesh(mesh)
        file.write_function(u)


@pytest.mark.parametrize("cell_type", celltypes_3D)
@pytest.mark.parametrize("encoding", encodings)
def test_save_3d_vector(tempdir, encoding, cell_type):
    filename = Path(tempdir, "u_3Dv.xdmf")
    mesh = create_unit_cube(MPI.COMM_WORLD, 2, 2, 2, cell_type)
    u = Function(VectorFunctionSpace(mesh, ("Lagrange", 1)))
    u.vector.set(1.0 + (1j if np.issubdtype(PETSc.ScalarType, np.complexfloating) else 0))
    with XDMFFile(mesh.comm, filename, "w", encoding=encoding) as file:
        file.write_mesh(mesh)
        file.write_function(u)


@pytest.mark.parametrize("cell_type", celltypes_2D)
@pytest.mark.parametrize("encoding", encodings)
def test_save_2d_tensor(tempdir, encoding, cell_type):
    filename = Path(tempdir, "tensor.xdmf")
    mesh = create_unit_square(MPI.COMM_WORLD, 16, 16, cell_type)
    u = Function(TensorFunctionSpace(mesh, ("Lagrange", 2)))
    u.vector.set(1.0 + (1j if np.issubdtype(PETSc.ScalarType, np.complexfloating) else 0))
    with XDMFFile(mesh.comm, filename, "w", encoding=encoding) as file:
        file.write_mesh(mesh)
        file.write_function(u)


@pytest.mark.parametrize("cell_type", celltypes_3D)
@pytest.mark.parametrize("encoding", encodings)
def test_save_3d_tensor(tempdir, encoding, cell_type):
    filename = Path(tempdir, "u3t.xdmf")
    mesh = create_unit_cube(MPI.COMM_WORLD, 4, 4, 4, cell_type)
    u = Function(TensorFunctionSpace(mesh, ("Lagrange", 2)))
    u.vector.set(1.0 + (1j if np.issubdtype(PETSc.ScalarType, np.complexfloating) else 0))
    with XDMFFile(mesh.comm, filename, "w", encoding=encoding) as file:
        file.write_mesh(mesh)
        file.write_function(u)


@pytest.mark.parametrize("cell_type", celltypes_3D)
@pytest.mark.parametrize("encoding", encodings)
def test_save_3d_vector_series(tempdir, encoding, cell_type):
    filename = Path(tempdir, "u_3D.xdmf")
    mesh = create_unit_cube(MPI.COMM_WORLD, 2, 2, 2, cell_type)
    u = Function(VectorFunctionSpace(mesh, ("Lagrange", 2)))
    with XDMFFile(mesh.comm, filename, "w", encoding=encoding) as file:
        file.write_mesh(mesh)
        u.vector.set(1.0 + (1j if np.issubdtype(PETSc.ScalarType, np.complexfloating) else 0))
        file.write_function(u, 0.1)
        u.vector.set(2.0 + (2j if np.issubdtype(PETSc.ScalarType, np.complexfloating) else 0))
        file.write_function(u, 0.2)

    with XDMFFile(mesh.comm, filename, "a", encoding=encoding) as file:
        u.vector.set(3.0 + (3j if np.issubdtype(PETSc.ScalarType, np.complexfloating) else 0))
        file.write_function(u, 0.3)
