# Copyright (C) 2011-2021 Garth N. Wells and Jørgen S. Dokken
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import os

import numpy as np
import pytest
import ufl
from dolfinx import (Function, FunctionSpace, TensorFunctionSpace,
                     UnitCubeMesh, UnitIntervalMesh, UnitSquareMesh,
                     VectorFunctionSpace)
from dolfinx.cpp.io import VTKFileNew
from dolfinx.cpp.mesh import CellType
from dolfinx.io import VTKFile
from dolfinx.mesh import create_mesh
from dolfinx_utils.test.fixtures import tempdir
from dolfinx_utils.test.skips import skip_in_parallel
from mpi4py import MPI
from petsc4py import PETSc

assert (tempdir)


def test_save_1d_mesh(tempdir):
    filename = os.path.join(tempdir, "mesh.pvd")
    mesh = UnitIntervalMesh(MPI.COMM_WORLD, 32)
    with VTKFileNew(MPI.COMM_WORLD, filename, "w") as vtk:
        vtk.write(mesh, 0)
        vtk.write(mesh, 1)


@pytest.mark.parametrize("cell_type", [CellType.triangle, CellType.quadrilateral])
def test_save_2d_mesh(tempdir, cell_type):
    mesh = UnitSquareMesh(MPI.COMM_WORLD, 32, 32)
    filename = os.path.join(tempdir, f"mesh_{cpp.mesh.to_string(cell_type)}.pvd")
    with VTKFileNew(MPI.COMM_WORLD, filename, "w") as vtk:
        vtk.write(mesh, 0.)
        vtk.write(mesh, 2.)


@pytest.mark.parametrize("cell_type", [CellType.tetrahedron, CellType.hexahedron])
def test_save_3d_mesh(tempdir, cell_type):
    mesh = UnitCubeMesh(MPI.COMM_WORLD, 8, 8, 8, cell_type=cell_type)
    filename = os.path.join(tempdir, f"mesh_{cpp.mesh.to_string(cell_type)}.pvd")
    with VTKFileNew(MPI.COMM_WORLD, filename, "w") as vtk:
        vtk.write(mesh, 0.)
        vtk.write(mesh, 2.)


def test_save_1d_scalar(tempdir):
    mesh = UnitIntervalMesh(MPI.COMM_WORLD, 32)

    def f(x):
        return x[0]

    u = Function(FunctionSpace(mesh, ("CG", 2)))
    u.interpolate(f)
    u.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    filename = os.path.join(tempdir, "u.pvd")
    with VTKFileNew(MPI.COMM_WORLD, filename, "w") as vtk:
        vtk.write([u._cpp_object], 0.)
        vtk.write([u._cpp_object], 1.)


@pytest.mark.xfail(reason="file_option not added to VTK initializer")
def test_save_2d_scalar_old(tempdir, file_options):
    mesh = UnitSquareMesh(MPI.COMM_WORLD, 16, 16)
    u = Function(FunctionSpace(mesh, ("Lagrange", 2)))
    u.vector.set(1.0)
    filename = os.path.join(tempdir, "u.pvd")
    VTKFile(filename).write(u)
    f = VTKFile(filename)
    f.write(u, 0.)
    f.write(u, 1.)
    for file_option in file_options:
        VTKFile(filename, file_option).write(u)


@pytest.mark.xfail(reason="P2->P1 interpolation not implemented")
def test_save_2d_scalar(tempdir):
    mesh = UnitSquareMesh(MPI.COMM_WORLD, 3, 3)
    u = Function(FunctionSpace(mesh, ("Lagrange", 2)))
    u.vector.set(1.0)
    # VTKFileNeww(filename).write(u)
    # f = VTKFileNew(mesh.mpi_comm(), filename, "w")
    f = VTKFileNew(mesh.mpi_comm(), "u.pvd", "w")
    f.write([u._cpp_object], 0.)
    # f.write(u, 1.)
    # for file_option in file_options:
    #     VTKFile(filename, file_option).write(u)


@pytest.mark.xfail(reason="file_option not added to VTK initializer")
def test_save_3d_scalar(tempdir, file_options):
    mesh = UnitCubeMesh(MPI.COMM_WORLD, 8, 8, 8)
    u = Function(FunctionSpace(mesh, ("Lagrange", 2)))
    u.vector.set(1.0)
    filename = os.path.join(tempdir, "u.pvd")
    VTKFile(filename).write(u)
    f = VTKFile(filename)
    f.write(u, 0.)
    f.write(u, 1.)
    for file_option in file_options:
        VTKFile(filename, file_option).write(u)


@pytest.mark.xfail(reason="FFCX fails for tensor spaces in 1D")
@skip_in_parallel
def test_save_1d_vector(tempdir, file_options):
    mesh = UnitIntervalMesh(MPI.COMM_WORLD, 32)
    u = Function(VectorFunctionSpace(mesh, ("Lagrange", 2)))
    u.vector.set(1.0)
    filename = os.path.join(tempdir, "u.pvd")
    VTKFile(filename).write(u)
    for file_option in file_options:
        VTKFile(filename, file_option).write(u)


@pytest.mark.xfail(reason="file_option not added to VTK initializer")
def test_save_2d_vector_old(tempdir, file_options):
    mesh = UnitSquareMesh(MPI.COMM_WORLD, 16, 16)
    u = Function(VectorFunctionSpace(mesh, "Lagrange", 2))
    u.vector.set(1)
    filename = os.path.join(tempdir, "u.pvd")
    VTKFile(filename).write(u)
    f = VTKFile(filename)
    f.write(u, 0.)
    f.write(u, 1.)
    for file_option in file_options:
        VTKFile(filename, file_option).write(u)


@skip_in_parallel
def test_save_2d_vector(tempdir):
    points = np.array([[0, 0], [1, 0], [1, 2], [0, 2],
                       [1 / 2, 0], [1, 2 / 2], [1 / 2, 2],
                       [0, 2 / 2], [1 / 2, 2 / 2]])
    points = np.array([[0, 0], [1, 0], [0, 2], [0.5, 1], [0, 1], [0.5, 0],
                       [1, 2], [0.5, 2], [1, 1]])

    cells = np.array([[0, 1, 2, 3, 4, 5],
                      [1, 6, 2, 7, 3, 8]])
    domain = ufl.Mesh(ufl.VectorElement("Lagrange", "triangle", 2))
    mesh = create_mesh(MPI.COMM_WORLD, cells, points, domain)
    u = Function(VectorFunctionSpace(mesh, ("Lagrange", 2)))

    def func(x):
        vals = np.zeros((2, x.shape[1]))
        vals[0] = x[0]
        vals[1] = x[1]
        return vals
    u.interpolate(func)
    filename = os.path.join(tempdir, "u2-new.pvd")
    f = VTKFileNew(mesh.mpi_comm(), filename, "w")
    f.write([u._cpp_object], 0.)
    # filename = os.path.join(tempdir, "u2-old.pvd")
    # fold = VTKFile(filename)
    # fold.write(u)
    # # f.write(u, 1.)
    # for file_option in file_options:
    #     VTKFile(filename, file_option).write(u)


def xtest_save_2d_mixed(tempdir):
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


@pytest.mark.xfail(reason="file_option not added to VTK initializer")
def test_save_3d_vector(tempdir, file_options):
    mesh = UnitCubeMesh(MPI.COMM_WORLD, 8, 8, 8)
    u = Function(VectorFunctionSpace(mesh, "Lagrange", 2))
    u.vector.set(1)
    filename = os.path.join(tempdir, "u.pvd")
    VTKFile(filename).write(u)
    f = VTKFile(filename)
    f.write(u, 0.)
    f.write(u, 1.)
    for file_option in file_options:
        VTKFile(filename, file_option).write(u)


@pytest.mark.xfail(reason="FFCX fails for tensor spaces in 1D")
@skip_in_parallel
def test_save_1d_tensor(tempdir, file_options):
    mesh = UnitIntervalMesh(MPI.COMM_WORLD, 32)
    u = Function(TensorFunctionSpace(mesh, ("Lagrange", 2)))
    u.vector.set(1)
    filename = os.path.join(tempdir, "u.pvd")
    VTKFile(filename).write(u)
    for file_option in file_options:
        VTKFile(filename, file_option).write(u)


def test_save_2d_tensor(tempdir):
    import time
    mesh = UnitSquareMesh(MPI.COMM_WORLD, 16, 16)
    u = Function(TensorFunctionSpace(mesh, ("Lagrange", 1)))
    u.vector.set(1)
    filename = os.path.join(tempdir, "u.pvd")
    f = VTKFileNew(MPI.COMM_WORLD, filename, "w")
    start = time.time()
    f.write([u._cpp_object], 0.)
    u.vector.set(2)
    f.write([u._cpp_object], 1.)
    end = time.time()
    print("NEW: {0:.2e}".format(end - start))

    f = VTKFile(tempdir + "u_old.pvd")
    start = time.time()
    f.write(u)
    u.vector.set(2)
    f.write(u)
    end = time.time()
    print("OLD: {0:.2e}".format(end - start))


@pytest.mark.xfail(reason="file_option not added to VTK initializer")
def test_save_3d_tensor(tempdir, file_options):
    mesh = UnitCubeMesh(MPI.COMM_WORLD, 8, 8, 8)
    u = Function(TensorFunctionSpace(mesh, ("Lagrange", 2)))
    u.vector.set(1)
    filename = os.path.join(tempdir, "u.pvd")
    VTKFile(filename).write(u)
    f = VTKFile(filename)
    f.write(u, 0.)
    f.write(u, 1.)
    for file_option in file_options:
        VTKFile(filename, file_option).write(u)
