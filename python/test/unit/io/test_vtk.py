# Copyright (C) 2011-2021 Garth N. Wells and Jørgen S. Dokken
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from pathlib import Path

import numpy as np
import pytest

import ufl
from dolfinx.fem import (Function, FunctionSpace, TensorFunctionSpace,
                         VectorFunctionSpace)
from dolfinx.io import VTKFile
from dolfinx.mesh import (CellType, create_mesh, create_unit_cube,
                          create_unit_interval, create_unit_square)
from dolfinx.plot import create_vtk_mesh


from mpi4py import MPI

cell_types_2D = [CellType.triangle, CellType.quadrilateral]
cell_types_3D = [CellType.tetrahedron, CellType.hexahedron]


def test_save_1d_mesh_subdir(tempdir):
    filename = Path(tempdir, "mesh.pvd")
    mesh = create_unit_interval(MPI.COMM_WORLD, 32)
    with VTKFile(MPI.COMM_WORLD, filename, "w") as vtk:
        vtk.write_mesh(mesh)
        vtk.write_mesh(mesh, 1)


@pytest.mark.parametrize("cell_type", cell_types_2D)
def test_save_2d_mesh(tempdir, cell_type):
    mesh = create_unit_square(MPI.COMM_WORLD, 32, 32, cell_type=cell_type)
    filename = Path(tempdir, f"mesh_{cell_type.name}.pvd")
    with VTKFile(MPI.COMM_WORLD, filename, "w") as vtk:
        vtk.write_mesh(mesh, 0.)
        vtk.write_mesh(mesh, 2.)


@pytest.mark.parametrize("cell_type", cell_types_3D)
def test_save_3d_mesh(tempdir, cell_type):
    mesh = create_unit_cube(MPI.COMM_WORLD, 8, 8, 8, cell_type=cell_type)
    filename = Path(tempdir, f"mesh_{cell_type.name}.pvd")
    with VTKFile(MPI.COMM_WORLD, filename, "w") as vtk:
        vtk.write_mesh(mesh, 0.)
        vtk.write_mesh(mesh, 2.)


def test_save_1d_scalar(tempdir):
    mesh = create_unit_interval(MPI.COMM_WORLD, 32)
    u = Function(FunctionSpace(mesh, ("Lagrange", 2)))
    u.interpolate(lambda x: x[0])
    filename = Path(tempdir, "u.pvd")
    with VTKFile(MPI.COMM_WORLD, filename, "w") as vtk:
        vtk.write_function(u, 0.)


@pytest.mark.parametrize("cell_type", cell_types_2D)
def test_save_2d_scalar(tempdir, cell_type):
    mesh = create_unit_square(MPI.COMM_WORLD, 16, 16, cell_type=cell_type)
    u = Function(FunctionSpace(mesh, ("Lagrange", 2)))
    u.x.array[:] = 1.0

    filename = Path(tempdir, "u.pvd")
    with VTKFile(MPI.COMM_WORLD, filename, "w") as vtk:
        vtk.write_function(u, 0.)
        vtk.write_function(u, 1.)


@pytest.mark.parametrize("cell_type", cell_types_3D)
def test_save_3d_scalar(tempdir, cell_type):
    mesh = create_unit_cube(MPI.COMM_WORLD, 8, 8, 8, cell_type=cell_type)
    u = Function(FunctionSpace(mesh, ("Lagrange", 2)))
    u.x.array[:] = 1.0

    filename = Path(tempdir, "u.pvd")
    with VTKFile(MPI.COMM_WORLD, filename, "w") as vtk:
        vtk.write_function(u, 0.)
        vtk.write_function(u, 1.)


def test_save_1d_vector(tempdir):
    mesh = create_unit_interval(MPI.COMM_WORLD, 32)

    def f(x):
        vals = np.zeros((2, x.shape[1]))
        vals[0] = x[0]
        vals[1] = 2 * x[0] * x[0]
        return vals

    element = ufl.VectorElement("Lagrange", mesh.ufl_cell(), 2, dim=2)
    u = Function(FunctionSpace(mesh, element))
    u.interpolate(f)
    filename = Path(tempdir, "u.pvd")
    with VTKFile(MPI.COMM_WORLD, filename, "w") as vtk:
        vtk.write_function(u, 0.)


@pytest.mark.parametrize("cell_type", cell_types_2D)
def test_save_2d_vector(tempdir, cell_type):
    mesh = create_unit_square(MPI.COMM_WORLD, 16, 16, cell_type=cell_type)
    u = Function(VectorFunctionSpace(mesh, ("Lagrange", 1)))

    def f(x):
        vals = np.zeros((2, x.shape[1]))
        vals[0] = x[0]
        vals[1] = 2 * x[0] * x[1]
        return vals

    u.interpolate(f)
    filename = Path(tempdir, "u.pvd")
    with VTKFile(MPI.COMM_WORLD, filename, "w") as vtk:
        vtk.write_function(u, 0.)
        vtk.write_function(u, 1.)


@pytest.mark.skip_in_parallel
def test_save_2d_vector_CG2(tempdir):
    points = np.array([[0, 0], [1, 0], [1, 2], [0, 2],
                       [1 / 2, 0], [1, 1], [1 / 2, 2],
                       [0, 1], [1 / 2, 1]])
    points = np.array([[0, 0], [1, 0], [0, 2], [0.5, 1], [0, 1], [0.5, 0],
                       [1, 2], [0.5, 2], [1, 1]])
    cells = np.array([[0, 1, 2, 3, 4, 5],
                      [1, 6, 2, 7, 3, 8]])
    domain = ufl.Mesh(ufl.VectorElement("Lagrange", "triangle", 2))
    mesh = create_mesh(MPI.COMM_WORLD, cells, points, domain)
    u = Function(VectorFunctionSpace(mesh, ("Lagrange", 2)))
    u.interpolate(lambda x: np.vstack((x[0], x[1])))
    filename = Path(tempdir, "u.pvd")
    with VTKFile(mesh.comm, filename, "w") as vtk:
        vtk.write_function(u, 0.)


def test_save_vtk_mixed(tempdir):
    mesh = create_unit_cube(MPI.COMM_WORLD, 3, 3, 3)
    P2 = ufl.VectorElement("Lagrange", mesh.ufl_cell(), 1)
    P1 = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    W = FunctionSpace(mesh, P2 * P1)
    V1 = FunctionSpace(mesh, P1)
    V2 = FunctionSpace(mesh, P2)

    U = Function(W)
    U.sub(0).interpolate(lambda x: np.vstack((x[0], 0.2 * x[1], np.zeros_like(x[0]))))
    U.sub(1).interpolate(lambda x: 0.5 * x[0])

    U1, U2 = Function(V1), Function(V2)
    U1.interpolate(U.sub(1))
    U2.interpolate(U.sub(0))
    U2.name = "u"
    U1.name = "p"

    filename = Path(tempdir, "u.pvd")
    with VTKFile(mesh.comm, filename, "w") as vtk:
        vtk.write_function([U2, U1], 0.)
    with VTKFile(mesh.comm, filename, "w") as vtk:
        vtk.write_function([U1, U2], 0.)

    Up = U.sub(1)
    Up.name = "psub"
    with pytest.raises(RuntimeError):
        with VTKFile(mesh.comm, filename, "w") as vtk:
            vtk.write_function([U2, Up, U1], 0)
    with pytest.raises(RuntimeError):
        with VTKFile(mesh.comm, filename, "w") as vtk:
            vtk.write_function([U.sub(i) for i in range(W.num_sub_spaces)], 0)


def test_save_vtk_cell_point(tempdir):
    """Test writing cell-wise and point-wise data"""
    mesh = create_unit_cube(MPI.COMM_WORLD, 3, 3, 3)
    P2 = ufl.VectorElement("Lagrange", mesh.ufl_cell(), 1)
    P1 = ufl.FiniteElement("Discontinuous Lagrange", mesh.ufl_cell(), 0)

    V2, V1 = FunctionSpace(mesh, P2), FunctionSpace(mesh, P1)
    U2, U1 = Function(V2), Function(V1)
    U2.interpolate(lambda x: np.vstack((x[0], 0.2 * x[1], np.zeros_like(x[0]))))
    U1.interpolate(lambda x: 0.5 * x[0])
    U2.name = "A"
    U1.name = "B"

    filename = Path(tempdir, "u.pvd")
    with VTKFile(mesh.comm, filename, "w") as vtk:
        vtk.write_function([U2, U1], 0.)
    with VTKFile(mesh.comm, filename, "w") as vtk:
        vtk.write_function((U1, U2), 0.)


def test_save_1d_tensor(tempdir):
    mesh = create_unit_interval(MPI.COMM_WORLD, 32)
    element = ufl.TensorElement("Lagrange", mesh.ufl_cell(), 2, shape=(2, 2))
    u = Function(FunctionSpace(mesh, element))
    u.x.array[:] = 1.0
    filename = Path(tempdir, "u.pvd")
    with VTKFile(mesh.comm, filename, "w") as vtk:
        vtk.write_function(u, 0.)


def test_save_2d_tensor(tempdir):
    mesh = create_unit_square(MPI.COMM_WORLD, 16, 16)
    u = Function(TensorFunctionSpace(mesh, ("Lagrange", 2)))
    u.x.array[:] = 1.0
    filename = Path(tempdir, "u.pvd")
    with VTKFile(mesh.comm, filename, "w") as vtk:
        vtk.write_function(u, 0.)
        u.x.array[:] = 2.0
        vtk.write_function(u, 1.)


def test_save_3d_tensor(tempdir):
    mesh = create_unit_cube(MPI.COMM_WORLD, 8, 8, 8)
    u = Function(TensorFunctionSpace(mesh, ("Lagrange", 2)))
    u.x.array[:] = 1.0
    filename = Path(tempdir, "u.pvd")
    with VTKFile(mesh.comm, filename, "w") as vtk:
        vtk.write_function(u, 0.)


def test_vtk_mesh():
    comm = MPI.COMM_WORLD
    mesh = create_unit_square(comm, 2 * comm.size, 2 * comm.size)
    V = FunctionSpace(mesh, ("Lagrange", 1))
    create_vtk_mesh(V)
