# Copyright (C) 2009-2020 Garth N. Wells, Matthew W. Scroggs and Jorgen S. Dokken
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Test that interpolation is done correctly"""

import random

import numpy as np
import pytest

import ufl
from dolfinx.fem import (Function, FunctionSpace, VectorFunctionSpace,
                         assemble_scalar)
from dolfinx.mesh import create_unit_cube_mesh, create_unit_square_mesh
from dolfinx.mesh import CellType, create_mesh
from dolfinx_utils.test.skips import skip_in_parallel

from mpi4py import MPI

parametrize_cell_types = pytest.mark.parametrize(
    "cell_type", [
        CellType.interval,
        CellType.triangle,
        CellType.tetrahedron,
        CellType.quadrilateral,
        CellType.hexahedron
    ])


def random_point_in_reference(cell_type):
    if cell_type == CellType.interval:
        return (random.random(), 0, 0)
    elif cell_type == CellType.triangle:
        x, y = random.random(), random.random()
        # If point is outside cell, move it back inside
        if x + y > 1:
            x, y = 1 - x, 1 - y
        return (x, y, 0)
    elif cell_type == CellType.tetrahedron:
        x, y, z = random.random(), random.random(), random.random()
        # If point is outside cell, move it back inside
        if x + y > 1:
            x, y = 1 - x, 1 - y
        if y + z > 1:
            y, z = 1 - z, 1 - x - y
        if x + y + z > 1:
            x, z = 1 - x - y, x + y + z - 1
        return (x, y, z)
    elif cell_type == CellType.quadrilateral:
        x, y = random.random(), random.random()
        return (x, y, 0)
    elif cell_type == CellType.hexahedron:
        return (random.random(), random.random(), random.random())


def random_point_in_cell(mesh):
    cell_type = mesh.topology.cell_type
    point = random_point_in_reference(cell_type)

    if cell_type == CellType.interval:
        origin = mesh.geometry.x[0]
        axes = (mesh.geometry.x[1], )
    elif cell_type == CellType.triangle:
        origin = mesh.geometry.x[0]
        axes = (mesh.geometry.x[1], mesh.geometry.x[2])
    elif cell_type == CellType.tetrahedron:
        origin = mesh.geometry.x[0]
        axes = (mesh.geometry.x[1], mesh.geometry.x[2], mesh.geometry.x[3])
    elif cell_type == CellType.quadrilateral:
        origin = mesh.geometry.x[0]
        axes = (mesh.geometry.x[1], mesh.geometry.x[2])
    elif cell_type == CellType.hexahedron:
        origin = mesh.geometry.x[0]
        axes = (mesh.geometry.x[1], mesh.geometry.x[2], mesh.geometry.x[4])

    return tuple(origin[i] + sum((axis[i] - origin[i]) * p for axis, p in zip(axes, point)) for i in range(3))


def one_cell_mesh(cell_type):
    if cell_type == CellType.interval:
        points = np.array([[-1.], [2.]])
    if cell_type == CellType.triangle:
        points = np.array([[-1., -1.], [2., 0.], [0., 0.5]])
    elif cell_type == CellType.tetrahedron:
        points = np.array([[-1., -1., -1.], [2., 0., 0.], [0., 0.5, 0.], [0., 0., 1.]])
    elif cell_type == CellType.quadrilateral:
        points = np.array([[-1., 0.], [1., 0.], [-1., 1.5], [1., 1.5]])
    elif cell_type == CellType.hexahedron:
        points = np.array([[-1., -0.5, 0.], [1., -0.5, 0.], [-1., 1.5, 0.],
                           [1., 1.5, 0.], [0., -0.5, 1.], [1., -0.5, 1.],
                           [-1., 1.5, 1.], [1., 1.5, 1.]])
    num_points = len(points)

    # Randomly number the points and create the mesh
    order = list(range(num_points))
    random.shuffle(order)
    ordered_points = np.zeros(points.shape)
    for i, j in enumerate(order):
        ordered_points[j] = points[i]
    cells = np.array([order])

    domain = ufl.Mesh(ufl.VectorElement("Lagrange", cell_type.name, 1))
    return create_mesh(MPI.COMM_WORLD, cells, ordered_points, domain)


def run_scalar_test(V, poly_order):
    """Test that interpolation is correct in a scalar valued space."""
    random.seed(13)
    tdim = V.mesh.topology.dim

    if tdim == 1:
        def f(x):
            return x[0] ** poly_order
    elif tdim == 2:
        def f(x):
            return x[1] ** poly_order + 2 * x[0] ** min(poly_order, 1)
    else:
        def f(x):
            return x[1] ** poly_order + 2 * x[0] ** min(poly_order, 1) - 3 * x[2] ** min(poly_order, 2)

    v = Function(V)
    v.interpolate(f)
    points = [random_point_in_cell(V.mesh) for count in range(5)]
    cells = [0 for count in range(5)]
    values = v.eval(points, cells)
    for p, val in zip(points, values):
        assert np.allclose(val, f(p))


def run_vector_test(V, poly_order):
    """Test that interpolation is correct in a scalar valued space."""
    random.seed(12)
    tdim = V.mesh.topology.dim

    if tdim == 1:
        def f(x):
            return x[0] ** poly_order
    elif tdim == 2:
        def f(x):
            return (x[1] ** min(poly_order, 1), 2 * x[0] ** poly_order)
    else:
        def f(x):
            return (x[1] ** min(poly_order, 1), 2 * x[0] ** poly_order, 3 * x[2] ** min(poly_order, 2))

    v = Function(V)
    v.interpolate(f)
    points = [random_point_in_cell(V.mesh) for count in range(5)]
    cells = [0 for count in range(5)]
    values = v.eval(points, cells)
    for p, val in zip(points, values):
        assert np.allclose(val, f(p))


@skip_in_parallel
@parametrize_cell_types
@pytest.mark.parametrize("order", range(1, 5))
def test_Lagrange_interpolation(cell_type, order):
    """Test that interpolation is correct in a FunctionSpace"""
    mesh = one_cell_mesh(cell_type)
    V = FunctionSpace(mesh, ("Lagrange", order))
    run_scalar_test(V, order)


@skip_in_parallel
@parametrize_cell_types
@pytest.mark.parametrize('order', range(1, 5))
def test_vector_interpolation(cell_type, order):
    """Test that interpolation is correct in a VectorFunctionSpace."""
    mesh = one_cell_mesh(cell_type)
    V = VectorFunctionSpace(mesh, ("Lagrange", order))
    run_vector_test(V, order)


@skip_in_parallel
@pytest.mark.parametrize(
    "cell_type", [CellType.triangle, CellType.tetrahedron])
@pytest.mark.parametrize("order", range(1, 5))
def test_N1curl_interpolation(cell_type, order):
    random.seed(8)
    mesh = one_cell_mesh(cell_type)
    V = FunctionSpace(mesh, ("Nedelec 1st kind H(curl)", order))
    run_vector_test(V, order - 1)


@skip_in_parallel
@pytest.mark.parametrize("cell_type", [CellType.triangle])
@pytest.mark.parametrize("order", [1, 2])
def test_N2curl_interpolation(cell_type, order):
    mesh = one_cell_mesh(cell_type)
    V = FunctionSpace(mesh, ("Nedelec 2nd kind H(curl)", order))
    run_vector_test(V, order)


@skip_in_parallel
@pytest.mark.parametrize(
    "cell_type", [CellType.quadrilateral])
@pytest.mark.parametrize("order", range(1, 5))
def test_RTCE_interpolation(cell_type, order):
    random.seed(8)
    mesh = one_cell_mesh(cell_type)
    V = FunctionSpace(mesh, ("RTCE", order))
    run_vector_test(V, order - 1)


@skip_in_parallel
@pytest.mark.parametrize(
    "cell_type", [CellType.hexahedron])
@pytest.mark.parametrize("order", range(1, 5))
def test_NCE_interpolation(cell_type, order):
    random.seed(8)
    mesh = one_cell_mesh(cell_type)
    V = FunctionSpace(mesh, ("NCE", order))
    run_vector_test(V, order - 1)


@skip_in_parallel
def test_mixed_interpolation():
    """Test that interpolation raised an exception."""
    mesh = one_cell_mesh(CellType.triangle)
    A = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    B = ufl.VectorElement("Lagrange", mesh.ufl_cell(), 1)
    v = Function(FunctionSpace(mesh, ufl.MixedElement([A, B])))
    with pytest.raises(RuntimeError):
        v.interpolate(lambda x: (x[1], 2 * x[0], 3 * x[1]))


@pytest.mark.parametrize("order1", [2, 3, 4])
@pytest.mark.parametrize("order2", [2, 3, 4])
def test_interpolation_nedelec(order1, order2):
    mesh = create_unit_cube_mesh(MPI.COMM_WORLD, 2, 2, 2)
    V = FunctionSpace(mesh, ("N1curl", order1))
    V1 = FunctionSpace(mesh, ("N1curl", order2))

    u = Function(V)
    v = Function(V1)

    # The expression "lambda x: x" is contained in the N1curl function
    # space order>1
    u.interpolate(lambda x: x)
    v.interpolate(u)

    assert np.isclose(assemble_scalar(ufl.inner(u - v, u - v) * ufl.dx), 0)

    # The target expression is also contained in N2curl space of any
    # order
    V2 = FunctionSpace(mesh, ("N2curl", 1))
    w = Function(V2)
    w.interpolate(u)

    assert np.isclose(assemble_scalar(ufl.inner(u - w, u - w) * ufl.dx), 0)


@pytest.mark.parametrize("tdim", [2, 3])
@pytest.mark.parametrize("order", [1, 2, 3])
def test_interpolation_dg_to_n1curl(tdim, order):
    if tdim == 2:
        mesh = create_unit_square_mesh(MPI.COMM_WORLD, 5, 5)
    else:
        mesh = create_unit_cube_mesh(MPI.COMM_WORLD, 2, 2, 2)
    V = VectorFunctionSpace(mesh, ("DG", order))
    V1 = FunctionSpace(mesh, ("N1curl", order + 1))

    u = Function(V)
    v = Function(V1)

    u.interpolate(lambda x: x[:tdim] ** order)
    v.interpolate(u)
    s = assemble_scalar(ufl.inner(u - v, u - v) * ufl.dx)
    assert np.isclose(s, 0)


@pytest.mark.parametrize("tdim", [2, 3])
@pytest.mark.parametrize("order", [1, 2, 3])
def test_interpolation_n1curl_to_dg(tdim, order):
    if tdim == 2:
        mesh = create_unit_square_mesh(MPI.COMM_WORLD, 5, 5)
    else:
        mesh = create_unit_cube_mesh(MPI.COMM_WORLD, 2, 2, 2)
    V = FunctionSpace(mesh, ("N1curl", order + 1))
    V1 = VectorFunctionSpace(mesh, ("DG", order))

    u = Function(V)
    v = Function(V1)

    u.interpolate(lambda x: x[:tdim] ** order)
    v.interpolate(u)
    s = assemble_scalar(ufl.inner(u - v, u - v) * ufl.dx)
    assert np.isclose(s, 0)


@pytest.mark.parametrize("tdim", [2, 3])
@pytest.mark.parametrize("order", [1, 2, 3])
def test_interpolation_n2curl_to_bdm(tdim, order):
    if tdim == 2:
        mesh = create_unit_square_mesh(MPI.COMM_WORLD, 5, 5)
    else:
        mesh = create_unit_cube_mesh(MPI.COMM_WORLD, 2, 2, 2)
    V = FunctionSpace(mesh, ("N2curl", order))
    V1 = FunctionSpace(mesh, ("BDM", order))

    u = Function(V)
    v = Function(V1)

    u.interpolate(lambda x: x[:tdim] ** order)
    v.interpolate(u)
    s = assemble_scalar(ufl.inner(u - v, u - v) * ufl.dx)
    assert np.isclose(s, 0)


@pytest.mark.parametrize("order1", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("order2", [1, 2, 3])
def test_interpolation_p2p(order1, order2):
    mesh = create_unit_cube_mesh(MPI.COMM_WORLD, 2, 2, 2)
    V = FunctionSpace(mesh, ("Lagrange", order1))
    V1 = FunctionSpace(mesh, ("Lagrange", order2))

    u = Function(V)
    v = Function(V1)

    u.interpolate(lambda x: x[0])
    v.interpolate(u)

    s = assemble_scalar(ufl.inner(u - v, u - v) * ufl.dx)
    assert np.isclose(s, 0)

    DG = FunctionSpace(mesh, ("DG", order2))
    w = Function(DG)
    w.interpolate(u)
    s = assemble_scalar(ufl.inner(u - w, u - w) * ufl.dx)
    assert np.isclose(s, 0)


@pytest.mark.parametrize("order1", [1, 2, 3])
@pytest.mark.parametrize("order2", [1, 2])
def test_interpolation_vector_elements(order1, order2):
    mesh = create_unit_cube_mesh(MPI.COMM_WORLD, 2, 2, 2)
    V = VectorFunctionSpace(mesh, ("Lagrange", order1))
    V1 = VectorFunctionSpace(mesh, ("Lagrange", order2))

    u = Function(V)
    v = Function(V1)

    u.interpolate(lambda x: x)
    v.interpolate(u)

    s = assemble_scalar(ufl.inner(u - v, u - v) * ufl.dx)
    assert np.isclose(s, 0)

    DG = VectorFunctionSpace(mesh, ("DG", order2))
    w = Function(DG)
    w.interpolate(u)
    s = assemble_scalar(ufl.inner(u - w, u - w) * ufl.dx)
    assert np.isclose(s, 0)


@skip_in_parallel
def test_interpolation_non_affine():
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0], [1, 2, 0],
                       [0, 0, 3], [1, 0, 3], [0, 2, 3], [1, 2, 3],
                       [0.5, 0, 0], [0, 1, 0], [0, 0, 1.5], [1, 1, 0],
                       [1, 0, 1.5], [0.5, 2, 0], [0, 2, 1.5], [1, 2, 1.5],
                       [0.5, 0, 3], [0, 1, 3], [1, 1, 3], [0.5, 2, 3],
                       [0.5, 1, 0], [0.5, 0, 1.5], [0, 1, 1.5], [1, 1, 1.5],
                       [0.5, 2, 1.5], [0.5, 1, 3], [0.5, 1, 1.5]], dtype=np.float64)

    cells = np.array([range(len(points))], dtype=np.int32)
    cell_type = CellType.hexahedron
    domain = ufl.Mesh(ufl.VectorElement("Lagrange", cell_type.name, 2))
    mesh = create_mesh(MPI.COMM_WORLD, cells, points, domain)

    W = FunctionSpace(mesh, ("NCE", 1))
    V = FunctionSpace(mesh, ("NCE", 2))

    w = Function(W)
    v = Function(V)

    w.interpolate(lambda x: x)
    v.interpolate(w)
    s = assemble_scalar(ufl.inner(w - v, w - v) * ufl.dx)
    assert np.isclose(s, 0)
