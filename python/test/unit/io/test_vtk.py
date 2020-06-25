# Copyright (C) 2011-2020 Garth N. Wells and Jørgen S. Dokken
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import os

import numpy as np
import pytest
import ufl
from mpi4py import MPI

from dolfinx import (Function, FunctionSpace, Mesh, TensorFunctionSpace,
                     UnitCubeMesh, UnitIntervalMesh, UnitSquareMesh,
                     VectorFunctionSpace)
from dolfinx.cpp.io import VTKFileNew
from dolfinx.cpp.mesh import CellType
from dolfinx.io import VTKFile
from dolfinx_utils.test.fixtures import tempdir
from dolfinx_utils.test.skips import skip_in_parallel

assert (tempdir)


@pytest.fixture
def cell_types_2D():
    return [CellType.triangle, CellType.quadrilateral]


@pytest.fixture
def cell_types_3D():
    return [CellType.tetrahedron, CellType.hexahedron]


# @pytest.fixture
# def file_options():
#     return ["ascii", "base64", "compressed"]


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
def test_save_2d_scalar_old(tempfile, file_options):
    mesh = UnitSquareMesh(MPI.COMM_WORLD, 16, 16)
    u = Function(FunctionSpace(mesh, ("Lagrange", 2)))
    u.vector.set(1.0)
    VTKFile(tempfile + "u.pvd").write(u)
    f = VTKFile(tempfile + "u.pvd")
    f.write(u, 0.)
    f.write(u, 1.)
    for file_option in file_options:
        VTKFile(tempfile + "u.pvd", file_option).write(u)


@pytest.mark.xfail(reason="P2->P1 interpolation not implemented")
def test_save_2d_scalar(tempfile):
    mesh = UnitSquareMesh(MPI.COMM_WORLD, 3, 3)
    u = Function(FunctionSpace(mesh, ("Lagrange", 2)))
    u.vector.set(1.0)
    # VTKFileNeww(tempfile + "u.pvd").write(u)
    # f = VTKFileNew(mesh.mpi_comm(), tempfile + "u.pvd", "w")
    f = VTKFileNew(mesh.mpi_comm(), "u.pvd", "w")
    f.write([u._cpp_object], 0.)
    # f.write(u, 1.)
    # for file_option in file_options:
    #     VTKFile(tempfile + "u.pvd", file_option).write(u)


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
def test_save_2d_vector_old(tempfile, file_options):
    mesh = UnitSquareMesh(MPI.COMM_WORLD, 16, 16)
    u = Function(VectorFunctionSpace(mesh, "Lagrange", 2))
    u.vector.set(1)
    VTKFile(tempfile + "u.pvd").write(u)
    f = VTKFile(tempfile + "u.pvd")
    f.write(u, 0.)
    f.write(u, 1.)
    for file_option in file_options:
        VTKFile(tempfile + "u.pvd", file_option).write(u)


def test_save_2d_vector(tempfile):
    points = np.array([[0, 0], [1, 0], [1, 2], [0, 2],
                       [1 / 2, 0], [1, 2 / 2], [1 / 2, 2],
                       [0, 2 / 2], [1 / 2, 2 / 2]])
    points = np.array([[0, 0], [1, 0], [0, 2], [0.5, 1], [0, 1], [0.5, 0],
                       [1, 2], [0.5, 2], [1, 1]])

    cells = np.array([[0, 1, 2, 3, 4, 5],
                      [1, 6, 2, 7, 3, 8]])
    mesh = Mesh(MPI.COMM_WORLD, CellType.triangle, points, cells, [], degree=2)
    u = Function(VectorFunctionSpace(mesh, ("Lagrange", 2)))
    u.vector.set(1)
    f = VTKFileNew(mesh.mpi_comm(), tempfile + "u2-new.pvd", "w")
    f.write([u._cpp_object], 0.)

    fold = VTKFile(tempfile + "u2-old.pvd")
    fold.write(u)

    # f.write(u, 1.)
    # for file_option in file_options:
    #     VTKFile(tempfile + "u.pvd", file_option).write(u)


def xtest_save_2d_mixed(tempfile):
    mesh = UnitCubeMesh(MPI.COMM_WORLD, 1, 1, 1)

    P2 = ufl.VectorElement("Lagrange", mesh.ufl_cell(), 2)
    P1 = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    TH = P2 * P1
    W = FunctionSpace(mesh, TH)

    U = Function(W)
    U.vector.set(1)
    # f = VTKFileNew(mesh.mpi_comm(), "u2.pvd", "w")
    # f.write(u._cpp_object, 0.)

    fold = VTKFile("U2-old.pvd")
    fold.write(U)


@ pytest.mark.xfail(reason="file_option not added to VTK initializer")
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


@ pytest.mark.xfail(reason="FFCX fails for tensor spaces in 1D")
@ skip_in_parallel
def test_save_1d_tensor(tempfile, file_options):
    mesh = UnitIntervalMesh(MPI.COMM_WORLD, 32)
    u = Function(TensorFunctionSpace(mesh, ("Lagrange", 2)))
    u.vector.set(1)
    VTKFile(tempfile + "u.pvd").write(u)
    for file_option in file_options:
        VTKFile(tempfile + "u.pvd", file_option).write(u)


def test_save_2d_tensor(tempfile):
    import time
    mesh = UnitSquareMesh(MPI.COMM_WORLD, 16, 16)
    u = Function(TensorFunctionSpace(mesh, ("Lagrange", 1)))
    u.vector.set(1)
    f = VTKFileNew(MPI.COMM_WORLD, tempfile + "u.pvd", "w")
    start = time.time()
    f.write([u._cpp_object], 0.)
    u.vector.set(2)
    f.write([u._cpp_object], 1.)
    end = time.time()
    print("NEW: {0:.2e}".format(end - start))

    f = VTKFile(tempfile + "u_old.pvd")
    start = time.time()
    f.write(u)
    u.vector.set(2)
    f.write(u)
    end = time.time()
    print("OLD: {0:.2e}".format(end - start))


@ pytest.mark.xfail(reason="file_option not added to VTK initializer")
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
