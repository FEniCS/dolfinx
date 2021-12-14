# Copyright (C) 2011-2021 Garth N. Wells and Jørgen S. Dokken
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import os

import numpy as np
import pytest

import ufl
from dolfinx.fem import (Function, FunctionSpace, TensorFunctionSpace,
                         VectorFunctionSpace)
from dolfinx.generation import UnitCubeMesh, UnitIntervalMesh, UnitSquareMesh
from dolfinx.io import VTKFile
from dolfinx.mesh import CellType, create_mesh
from dolfinx_utils.test.fixtures import tempdir
from dolfinx_utils.test.skips import skip_in_parallel

from mpi4py import MPI
from petsc4py import PETSc

assert (tempdir)

cell_types_2D = [CellType.triangle, CellType.quadrilateral]
cell_types_3D = [CellType.tetrahedron, CellType.hexahedron]


def test_save_1d_mesh(tempdir):
    filename = os.path.join(tempdir, "mesh.pvd")
    mesh = UnitIntervalMesh(MPI.COMM_WORLD, 32)
    with VTKFile(MPI.COMM_WORLD, filename, "w") as vtk:
        vtk.write_mesh(mesh)
        vtk.write_mesh(mesh, 1)


@pytest.mark.parametrize("cell_type", cell_types_2D)
def test_save_2d_mesh(tempdir, cell_type):
    mesh = UnitSquareMesh(MPI.COMM_WORLD, 32, 32, cell_type=cell_type)
    filename = os.path.join(tempdir, f"mesh_{cell_type.name}.pvd")
    with VTKFile(MPI.COMM_WORLD, filename, "w") as vtk:
        vtk.write_mesh(mesh, 0.)
        vtk.write_mesh(mesh, 2.)


@pytest.mark.parametrize("cell_type", cell_types_3D)
def test_save_3d_mesh(tempdir, cell_type):
    mesh = UnitCubeMesh(MPI.COMM_WORLD, 8, 8, 8, cell_type=cell_type)
    filename = os.path.join(tempdir, f"mesh_{cell_type.name}.pvd")
    with VTKFile(MPI.COMM_WORLD, filename, "w") as vtk:
        vtk.write_mesh(mesh, 0.)
        vtk.write_mesh(mesh, 2.)


def test_save_1d_scalar(tempdir):
    mesh = UnitIntervalMesh(MPI.COMM_WORLD, 32)

    def f(x):
        return x[0]

    u = Function(FunctionSpace(mesh, ("Lagrange", 2)))
    u.interpolate(f)
    u.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    filename = os.path.join(tempdir, "u.pvd")
    with VTKFile(MPI.COMM_WORLD, filename, "w") as vtk:
        vtk.write_function(u, 0.)
        vtk.write_function(u, 1.)


@pytest.mark.parametrize("cell_type", cell_types_2D)
def test_save_2d_scalar(tempdir, cell_type):
    mesh = UnitSquareMesh(MPI.COMM_WORLD, 16, 16, cell_type=cell_type)
    u = Function(FunctionSpace(mesh, ("Lagrange", 2)))
    u.x.array[:] = 1.0

    filename = os.path.join(tempdir, "u.pvd")
    with VTKFile(MPI.COMM_WORLD, filename, "w") as vtk:
        vtk.write_function(u, 0.)
        vtk.write_function(u, 1.)


@pytest.mark.parametrize("cell_type", cell_types_3D)
def test_save_3d_scalar(tempdir, cell_type):
    mesh = UnitCubeMesh(MPI.COMM_WORLD, 8, 8, 8, cell_type=cell_type)
    u = Function(FunctionSpace(mesh, ("Lagrange", 2)))
    u.x.array[:] = 1.0

    filename = os.path.join(tempdir, "u.pvd")
    with VTKFile(MPI.COMM_WORLD, filename, "w") as vtk:
        vtk.write_function(u, 0.)
        vtk.write_function(u, 1.)


def test_save_1d_vector(tempdir):
    mesh = UnitIntervalMesh(MPI.COMM_WORLD, 32)

    def f(x):
        vals = np.zeros((2, x.shape[1]))
        vals[0] = x[0]
        vals[1] = 2 * x[0] * x[0]
        return vals

    element = ufl.VectorElement("Lagrange", mesh.ufl_cell(), 2, dim=2)
    u = Function(FunctionSpace(mesh, element))
    u.interpolate(f)
    u.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    filename = os.path.join(tempdir, "u.pvd")
    with VTKFile(MPI.COMM_WORLD, filename, "w") as vtk:
        vtk.write_function(u, 0.)


@pytest.mark.parametrize("cell_type", cell_types_2D)
def test_save_2d_vector(tempdir, cell_type):
    mesh = UnitSquareMesh(MPI.COMM_WORLD, 16, 16, cell_type=cell_type)
    u = Function(VectorFunctionSpace(mesh, ("Lagrange", 1)))

    def f(x):
        vals = np.zeros((2, x.shape[1]))
        vals[0] = x[0]
        vals[1] = 2 * x[0] * x[1]
        return vals

    u.interpolate(f)
    u.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    filename = os.path.join(tempdir, "u.pvd")
    with VTKFile(MPI.COMM_WORLD, filename, "w") as vtk:
        vtk.write_function(u, 0.)


@skip_in_parallel
def test_save_2d_vector_CG2(tempdir):
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
    filename = os.path.join(tempdir, "u.pvd")
    with VTKFile(mesh.comm, filename, "w") as vtk:
        vtk.write_function(u, 0.)


def test_save_2d_mixed(tempdir):
    mesh = UnitCubeMesh(MPI.COMM_WORLD, 3, 3, 3)

    P2 = ufl.VectorElement("Lagrange", mesh.ufl_cell(), 2)
    P1 = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    TH = P2 * P1
    W = FunctionSpace(mesh, TH)

    def vec_func(x):
        vals = np.zeros((3, x.shape[1]))
        vals[0] = x[0]
        vals[1] = 0.2 * x[1]
        return vals

    def scal_func(x):
        return 0.5 * x[0]

    U = Function(W)
    U.sub(0).interpolate(vec_func)
    U.sub(1).interpolate(scal_func)
    U.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    filename = os.path.join(tempdir, "u.pvd")
    with VTKFile(mesh.comm, filename, "w") as vtk:
        vtk.write_function([U.sub(i) for i in range(W.num_sub_spaces())], 0.)


def test_save_1d_tensor(tempdir):
    mesh = UnitIntervalMesh(MPI.COMM_WORLD, 32)
    element = ufl.TensorElement("Lagrange", mesh.ufl_cell(), 2, shape=(2, 2))
    u = Function(FunctionSpace(mesh, element))
    u.x.array[:] = 1.0
    filename = os.path.join(tempdir, "u.pvd")
    with VTKFile(mesh.comm, filename, "w") as vtk:
        vtk.write_function(u, 0.)


def test_save_2d_tensor(tempdir):
    mesh = UnitSquareMesh(MPI.COMM_WORLD, 16, 16)
    u = Function(TensorFunctionSpace(mesh, ("Lagrange", 2)))
    u.x.array[:] = 1.0
    filename = os.path.join(tempdir, "u.pvd")
    with VTKFile(mesh.comm, filename, "w") as vtk:
        vtk.write_function(u, 0.)
        u.x.array[:] = 2.0
        vtk.write_function(u, 1.)


def test_save_3d_tensor(tempdir):
    mesh = UnitCubeMesh(MPI.COMM_WORLD, 8, 8, 8)
    u = Function(TensorFunctionSpace(mesh, ("Lagrange", 2)))
    u.x.array[:] = 1.0
    filename = os.path.join(tempdir, "u.pvd")
    with VTKFile(mesh.comm, filename, "w") as vtk:
        vtk.write_function(u, 0.)
