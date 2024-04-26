# Copyright (C) 2011-2021 Garth N. Wells and Jørgen S. Dokken
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from pathlib import Path

from mpi4py import MPI

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import ufl
from basix.ufl import element, mixed_element
from dolfinx import default_real_type
from dolfinx.fem import Function, functionspace
from dolfinx.io import VTKFile
from dolfinx.io.utils import cell_perm_vtk  # F401
from dolfinx.mesh import (
    CellType,
    create_mesh,
    create_unit_cube,
    create_unit_interval,
    create_unit_square,
)
from dolfinx.plot import vtk_mesh

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
        vtk.write_mesh(mesh, 0.0)
        vtk.write_mesh(mesh, 2.0)


@pytest.mark.parametrize("cell_type", cell_types_3D)
def test_save_3d_mesh(tempdir, cell_type):
    mesh = create_unit_cube(MPI.COMM_WORLD, 8, 8, 8, cell_type=cell_type)
    filename = Path(tempdir, f"mesh_{cell_type.name}.pvd")
    with VTKFile(MPI.COMM_WORLD, filename, "w") as vtk:
        vtk.write_mesh(mesh, 0.0)
        vtk.write_mesh(mesh, 2.0)


def test_save_1d_scalar(tempdir):
    mesh = create_unit_interval(MPI.COMM_WORLD, 32)
    u = Function(functionspace(mesh, ("Lagrange", 2)))
    u.interpolate(lambda x: x[0])
    filename = Path(tempdir, "u.pvd")
    with VTKFile(MPI.COMM_WORLD, filename, "w") as vtk:
        vtk.write_function(u, 0.0)


@pytest.mark.parametrize("cell_type", cell_types_2D)
def test_save_2d_scalar(tempdir, cell_type):
    mesh = create_unit_square(MPI.COMM_WORLD, 16, 16, cell_type=cell_type)
    u = Function(functionspace(mesh, ("Lagrange", 2)))
    u.x.array[:] = 1.0

    filename = Path(tempdir, "u.pvd")
    with VTKFile(MPI.COMM_WORLD, filename, "w") as vtk:
        vtk.write_function(u, 0.0)
        vtk.write_function(u, 1.0)


@pytest.mark.parametrize("cell_type", cell_types_3D)
def test_save_3d_scalar(tempdir, cell_type):
    mesh = create_unit_cube(MPI.COMM_WORLD, 8, 8, 8, cell_type=cell_type)
    u = Function(functionspace(mesh, ("Lagrange", 2)))
    u.x.array[:] = 1.0

    filename = Path(tempdir, "u.pvd")
    with VTKFile(MPI.COMM_WORLD, filename, "w") as vtk:
        vtk.write_function(u, 0.0)
        vtk.write_function(u, 1.0)


def test_save_1d_vector(tempdir):
    mesh = create_unit_interval(MPI.COMM_WORLD, 32)

    def f(x):
        vals = np.zeros((2, x.shape[1]))
        vals[0] = x[0]
        vals[1] = 2 * x[0] * x[0]
        return vals

    e = element("Lagrange", mesh.basix_cell(), 2, shape=(2,), dtype=default_real_type)
    u = Function(functionspace(mesh, e))
    u.interpolate(f)
    filename = Path(tempdir, "u.pvd")
    with VTKFile(MPI.COMM_WORLD, filename, "w") as vtk:
        vtk.write_function(u, 0.0)


@pytest.mark.parametrize("cell_type", cell_types_2D)
def test_save_2d_vector(tempdir, cell_type):
    mesh = create_unit_square(MPI.COMM_WORLD, 16, 16, cell_type=cell_type)
    gdim = mesh.geometry.dim
    u = Function(functionspace(mesh, ("Lagrange", 1, (gdim,))))

    def f(x):
        vals = np.zeros((2, x.shape[1]))
        vals[0] = x[0]
        vals[1] = 2 * x[0] * x[1]
        return vals

    u.interpolate(f)
    filename = Path(tempdir, "u.pvd")
    with VTKFile(MPI.COMM_WORLD, filename, "w") as vtk:
        vtk.write_function(u, 0.0)
        vtk.write_function(u, 1.0)


@pytest.mark.skip_in_parallel
def test_save_2d_vector_CG2(tempdir):
    points = np.array(
        [[0, 0], [1, 0], [1, 2], [0, 2], [1 / 2, 0], [1, 1], [1 / 2, 2], [0, 1], [1 / 2, 1]],
        dtype=default_real_type,
    )
    points = np.array(
        [[0, 0], [1, 0], [0, 2], [0.5, 1], [0, 1], [0.5, 0], [1, 2], [0.5, 2], [1, 1]],
        dtype=default_real_type,
    )
    cells = np.array([[0, 1, 2, 3, 4, 5], [1, 6, 2, 7, 3, 8]])
    domain = ufl.Mesh(element("Lagrange", "triangle", 2, shape=(2,), dtype=default_real_type))
    mesh = create_mesh(MPI.COMM_WORLD, cells, points, domain)
    gdim = mesh.geometry.dim
    u = Function(functionspace(mesh, ("Lagrange", 2, (gdim,))))
    u.interpolate(lambda x: np.vstack((x[0], x[1])))
    filename = Path(tempdir, "u.pvd")
    with VTKFile(mesh.comm, filename, "w") as vtk:
        vtk.write_function(u, 0.0)


def test_save_vtk_mixed(tempdir):
    mesh = create_unit_cube(MPI.COMM_WORLD, 3, 3, 3)
    P2 = element(
        "Lagrange", mesh.basix_cell(), 1, shape=(mesh.geometry.dim,), dtype=default_real_type
    )
    P1 = element("Lagrange", mesh.basix_cell(), 1, dtype=default_real_type)
    W = functionspace(mesh, mixed_element([P2, P1]))
    V1 = functionspace(mesh, P1)
    V2 = functionspace(mesh, P2)

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
        vtk.write_function([U2, U1], 0.0)
    with VTKFile(mesh.comm, filename, "w") as vtk:
        vtk.write_function([U1, U2], 0.0)

    Up = U.sub(1)
    Up.name = "psub"
    with pytest.raises(RuntimeError):
        with VTKFile(mesh.comm, filename, "w") as vtk:
            vtk.write_function([U2, Up, U1], 0)
    with pytest.raises(RuntimeError):
        with VTKFile(mesh.comm, filename, "w") as vtk:
            vtk.write_function([U.sub(i) for i in range(W.num_sub_spaces)], 0)


@pytest.mark.parametrize("cell_type", cell_types_2D)
def test_save_vector_element(tempdir, cell_type):
    mesh = create_unit_square(MPI.COMM_WORLD, 16, 16, cell_type=cell_type)
    u = Function(functionspace(mesh, ("RT", 1)))

    def f(x):
        vals = np.zeros((2, x.shape[1]))
        vals[0] = x[0]
        vals[1] = 2 * x[0] * x[1]
        return vals

    u.interpolate(f)
    filename = Path(tempdir, "u.pvd")
    with pytest.raises(RuntimeError):
        with VTKFile(MPI.COMM_WORLD, filename, "w") as vtk:
            vtk.write_function(u, 0.0)
            vtk.write_function(u, 1.0)


def test_save_vtk_cell_point(tempdir):
    """Test writing cell-wise and point-wise data"""
    mesh = create_unit_cube(MPI.COMM_WORLD, 3, 3, 3)
    P2 = element("Lagrange", mesh.basix_cell(), 1, shape=(3,), dtype=default_real_type)
    P1 = element("Discontinuous Lagrange", mesh.basix_cell(), 0, dtype=default_real_type)

    V2, V1 = functionspace(mesh, P2), functionspace(mesh, P1)
    U2, U1 = Function(V2), Function(V1)
    U2.interpolate(lambda x: np.vstack((x[0], 0.2 * x[1], np.zeros_like(x[0]))))
    U1.interpolate(lambda x: 0.5 * x[0])
    U2.name = "A"
    U1.name = "B"

    filename = Path(tempdir, "u.pvd")
    with VTKFile(mesh.comm, filename, "w") as vtk:
        vtk.write_function([U2, U1], 0.0)
    with VTKFile(mesh.comm, filename, "w") as vtk:
        vtk.write_function((U1, U2), 0.0)


def test_save_1d_tensor(tempdir):
    mesh = create_unit_interval(MPI.COMM_WORLD, 32)
    e = element("Lagrange", mesh.basix_cell(), 2, shape=(2, 2), dtype=default_real_type)
    u = Function(functionspace(mesh, e))
    u.x.array[:] = 1.0
    filename = Path(tempdir, "u.pvd")
    with VTKFile(mesh.comm, filename, "w") as vtk:
        vtk.write_function(u, 0.0)


def test_save_2d_tensor(tempdir):
    mesh = create_unit_square(MPI.COMM_WORLD, 16, 16)
    gdim = mesh.geometry.dim
    u = Function(functionspace(mesh, ("Lagrange", 2, (gdim, gdim))))
    u.x.array[:] = 1.0
    filename = Path(tempdir, "u.pvd")
    with VTKFile(mesh.comm, filename, "w") as vtk:
        vtk.write_function(u, 0.0)
        u.x.array[:] = 2.0
        vtk.write_function(u, 1.0)


def test_save_3d_tensor(tempdir):
    mesh = create_unit_cube(MPI.COMM_WORLD, 8, 8, 8)
    gdim = mesh.geometry.dim
    u = Function(functionspace(mesh, ("Lagrange", 2, (gdim, gdim))))
    u.x.array[:] = 1.0
    filename = Path(tempdir, "u.pvd")
    with VTKFile(mesh.comm, filename, "w") as vtk:
        vtk.write_function(u, 0.0)


def test_triangle_perm_vtk():
    higher_order_triangle_perm = {
        10: np.array([0, 1, 2, 5, 6, 8, 7, 3, 4, 9]),
        15: np.array([0, 1, 2, 6, 7, 8, 11, 10, 9, 3, 4, 5, 12, 13, 14]),
        21: np.array([0, 1, 2, 7, 8, 9, 10, 14, 13, 12, 11, 3, 4, 5, 6, 15, 18, 16, 20, 19, 17]),
        28: np.array(
            [
                0,
                1,
                2,
                8,
                9,
                10,
                11,
                12,
                17,
                16,
                15,
                14,
                13,
                3,
                4,
                5,
                6,
                7,
                18,
                21,
                22,
                19,
                26,
                27,
                23,
                25,
                24,
                20,
            ]
        ),
        36: np.array(
            [
                0,
                1,
                2,
                9,
                10,
                11,
                12,
                13,
                14,
                20,
                19,
                18,
                17,
                16,
                15,
                3,
                4,
                5,
                6,
                7,
                8,
                21,
                24,
                25,
                26,
                22,
                32,
                33,
                34,
                27,
                31,
                35,
                28,
                30,
                29,
                23,
            ]
        ),
        45: np.array(
            [
                0,
                1,
                2,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                23,
                22,
                21,
                20,
                19,
                18,
                17,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                24,
                27,
                28,
                29,
                30,
                25,
                38,
                39,
                42,
                40,
                31,
                37,
                44,
                43,
                32,
                36,
                41,
                33,
                35,
                34,
                26,
            ]
        ),
        55: np.array(
            [
                0,
                1,
                2,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                26,
                25,
                24,
                23,
                22,
                21,
                20,
                19,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                27,
                30,
                31,
                32,
                33,
                34,
                28,
                44,
                45,
                48,
                49,
                46,
                35,
                43,
                53,
                54,
                50,
                36,
                42,
                52,
                51,
                37,
                41,
                47,
                38,
                40,
                39,
                29,
            ]
        ),
    }
    for p_test, v_test in higher_order_triangle_perm.items():
        v = cell_perm_vtk(CellType.triangle, p_test)
        assert_array_equal(v, v_test)


def test_vtk_mesh():
    comm = MPI.COMM_WORLD
    mesh = create_unit_square(comm, 2 * comm.size, 2 * comm.size)
    V = functionspace(mesh, ("Lagrange", 1))
    vtk_mesh(V)
