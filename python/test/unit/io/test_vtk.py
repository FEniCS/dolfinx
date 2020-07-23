# Copyright (C) 2011 Garth N. Wells
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import os

import pytest
from dolfinx import (Function, FunctionSpace, TensorFunctionSpace,
                     UnitCubeMesh, UnitIntervalMesh, UnitSquareMesh,
                     VectorFunctionSpace)
from dolfinx.cpp.mesh import CellType
from dolfinx.io import VTKFile
from dolfinx_utils.test.fixtures import tempdir
from dolfinx_utils.test.skips import skip_in_parallel
from mpi4py import MPI

assert (tempdir)


@pytest.fixture
def cell_types_2D():
    return [CellType.triangle, CellType.quadrilateral]


@pytest.fixture
def cell_types_3D():
    return [CellType.tetrahedron, CellType.hexahedron]


@pytest.fixture
def file_options():
    return ["ascii", "base64", "compressed"]


@pytest.fixture
def type_conv():
    return dict(size_t=int, int=int, double=float)


@pytest.fixture(scope="function")
def tempfile(tempdir, request):
    return os.path.join(tempdir, request.function.__name__)


@pytest.mark.xfail(reason="file_option not added to VTK initializer")
def test_save_1d_mesh(tempfile, file_options):
    mesh = UnitIntervalMesh(MPI.COMM_WORLD, 32)
    VTKFile(tempfile + "mesh.pvd").write(mesh)
    f = VTKFile(tempfile + "mesh.pvd")
    f.write(mesh, 0.)
    f.write(mesh, 1.)
    for file_option in file_options:
        VTKFile(tempfile + "mesh.pvd", file_option).write(mesh)


@pytest.mark.xfail(reason="file_option not added to VTK initializer")
def test_save_2d_mesh(tempfile, file_options):
    mesh = UnitSquareMesh(MPI.COMM_WORLD, 32, 32)
    VTKFile(tempfile + "mesh.pvd").write(mesh)
    f = VTKFile(tempfile + "mesh.pvd")
    f.write(mesh, 0.)
    f.write(mesh, 1.)
    for file_option in file_options:
        VTKFile(tempfile + "mesh.pvd", file_option).write(mesh)


@pytest.mark.xfail(reason="file_option not added to VTK initializer")
def test_save_3d_mesh(tempfile, file_options):
    mesh = UnitCubeMesh(MPI.COMM_WORLD, 8, 8, 8)
    VTKFile(tempfile + "mesh.pvd").write(mesh)
    f = VTKFile(tempfile + "mesh.pvd")
    f.write(mesh, 0.)
    f.write(mesh, 1.)
    for file_option in file_options:
        VTKFile(tempfile + "mesh.pvd", file_option).write(mesh)


@pytest.mark.xfail(reason="file_option not added to VTK initializer")
def test_save_1d_scalar(tempfile, file_options):
    mesh = UnitIntervalMesh(MPI.COMM_WORLD, 32)
    u = Function(FunctionSpace(mesh, ("Lagrange", 2)))
    u.vector.set(1.0)
    VTKFile(tempfile + "u.pvd").write(u)
    f = VTKFile(tempfile + "u.pvd")
    f.write(u, 0.)
    f.write(u, 1.)
    for file_option in file_options:
        VTKFile(tempfile + "u.pvd", file_option).write(u)


@pytest.mark.xfail(reason="file_option not added to VTK initializer")
def test_save_2d_scalar(tempfile, file_options):
    mesh = UnitSquareMesh(MPI.COMM_WORLD, 16, 16)
    u = Function(FunctionSpace(mesh, ("Lagrange", 2)))
    u.vector.set(1.0)
    VTKFile(tempfile + "u.pvd").write(u)
    f = VTKFile(tempfile + "u.pvd")
    f.write(u, 0.)
    f.write(u, 1.)
    for file_option in file_options:
        VTKFile(tempfile + "u.pvd", file_option).write(u)


@pytest.mark.xfail(reason="file_option not added to VTK initializer")
def test_save_3d_scalar(tempfile, file_options):
    mesh = UnitCubeMesh(MPI.COMM_WORLD, 8, 8, 8)
    u = Function(FunctionSpace(mesh, ("Lagrange", 2)))
    u.vector.set(1.0)
    VTKFile(tempfile + "u.pvd").write(u)
    f = VTKFile(tempfile + "u.pvd")
    f.write(u, 0.)
    f.write(u, 1.)
    for file_option in file_options:
        VTKFile(tempfile + "u.pvd", file_option).write(u)


@pytest.mark.xfail(reason="FFCX fails for tensor spaces in 1D")
@skip_in_parallel
def test_save_1d_vector(tempfile, file_options):
    mesh = UnitIntervalMesh(MPI.COMM_WORLD, 32)
    u = Function(VectorFunctionSpace(mesh, ("Lagrange", 2)))
    u.vector.set(1.0)
    VTKFile(tempfile + "u.pvd").write(u)
    for file_option in file_options:
        VTKFile(tempfile + "u.pvd", file_option).write(u)


@pytest.mark.xfail(reason="file_option not added to VTK initializer")
def test_save_2d_vector(tempfile, file_options):
    mesh = UnitSquareMesh(MPI.COMM_WORLD, 16, 16)
    u = Function(VectorFunctionSpace(mesh, "Lagrange", 2))
    u.vector.set(1)
    VTKFile(tempfile + "u.pvd").write(u)
    f = VTKFile(tempfile + "u.pvd")
    f.write(u, 0.)
    f.write(u, 1.)
    for file_option in file_options:
        VTKFile(tempfile + "u.pvd", file_option).write(u)


@pytest.mark.xfail(reason="file_option not added to VTK initializer")
def test_save_3d_vector(tempfile, file_options):
    mesh = UnitCubeMesh(MPI.COMM_WORLD, 8, 8, 8)
    u = Function(VectorFunctionSpace(mesh, "Lagrange", 2))
    u.vector.set(1)
    VTKFile(tempfile + "u.pvd").write(u)
    f = VTKFile(tempfile + "u.pvd")
    f.write(u, 0.)
    f.write(u, 1.)
    for file_option in file_options:
        VTKFile(tempfile + "u.pvd", file_option).write(u)


@pytest.mark.xfail(reason="FFCX fails for tensor spaces in 1D")
@skip_in_parallel
def test_save_1d_tensor(tempfile, file_options):
    mesh = UnitIntervalMesh(MPI.COMM_WORLD, 32)
    u = Function(TensorFunctionSpace(mesh, ("Lagrange", 2)))
    u.vector.set(1)
    VTKFile(tempfile + "u.pvd").write(u)
    for file_option in file_options:
        VTKFile(tempfile + "u.pvd", file_option).write(u)


@pytest.mark.xfail(reason="file_option not added to VTK initializer")
def test_save_2d_tensor(tempfile, file_options):
    mesh = UnitSquareMesh(MPI.COMM_WORLD, 16, 16)
    u = Function(TensorFunctionSpace(mesh, ("Lagrange", 2)))
    u.vector.set(1)
    VTKFile(tempfile + "u.pvd").write(u)
    f = VTKFile(tempfile + "u.pvd")
    f.write(u, 0.)
    f.write(u, 1.)
    for file_option in file_options:
        VTKFile(tempfile + "u.pvd", file_option).write(u)


@pytest.mark.xfail(reason="file_option not added to VTK initializer")
def test_save_3d_tensor(tempfile, file_options):
    mesh = UnitCubeMesh(MPI.COMM_WORLD, 8, 8, 8)
    u = Function(TensorFunctionSpace(mesh, ("Lagrange", 2)))
    u.vector.set(1)
    VTKFile(tempfile + "u.pvd").write(u)
    f = VTKFile(tempfile + "u.pvd")
    f.write(u, 0.)
    f.write(u, 1.)
    for file_option in file_options:
        VTKFile(tempfile + "u.pvd", file_option).write(u)
