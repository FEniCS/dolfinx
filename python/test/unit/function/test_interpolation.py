# Copyright (C) 2009-2020 Garth N. Wells, Matthew W. Scroggs and Jorgen S. Dokken
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Test that interpolation is done correctly"""

import random

import numpy as np
import pytest
import ufl
from dolfinx import Function, FunctionSpace, VectorFunctionSpace, cpp
from dolfinx.cpp.mesh import CellType
from dolfinx.mesh import create_mesh
from dolfinx_utils.test.skips import skip_in_parallel
from mpi4py import MPI

parametrize_cell_types = pytest.mark.parametrize(
    "cell_type", [CellType.interval, CellType.triangle, CellType.tetrahedron,
                  CellType.quadrilateral, CellType.hexahedron])


def random_point_in_cell(cell_type):
    if cell_type == CellType.interval:
        return (random.random(), 0, 0)
    if cell_type == CellType.triangle:
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


def one_cell_mesh(cell_type):
    if cell_type == CellType.interval:
        points = np.array([[0.], [1.]])
    if cell_type == CellType.triangle:
        points = np.array([[0., 0.], [1., 0.], [0., 1.]])
    elif cell_type == CellType.tetrahedron:
        points = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    elif cell_type == CellType.quadrilateral:
        points = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
    elif cell_type == CellType.hexahedron:
        points = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
                           [1., 1., 0.], [0., 0., 1.], [1., 0., 1.],
                           [0., 1., 1.], [1., 1., 1.]])
    num_points = len(points)

    # Randomly number the points and create the mesh
    order = list(range(num_points))
    random.shuffle(order)
    ordered_points = np.zeros(points.shape)
    ordered_points = np.zeros(points.shape)
    for i, j in enumerate(order):
        ordered_points[j] = points[i]
    cells = np.array([order])
    domain = ufl.Mesh(ufl.VectorElement("Lagrange", cpp.mesh.to_string(cell_type), 1))
    mesh = create_mesh(MPI.COMM_WORLD, cells, ordered_points, domain)

    mesh.topology.create_connectivity_all()
    return mesh


@skip_in_parallel
@parametrize_cell_types
@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_scalar_interpolation(cell_type, order):
    """Test that interpolation is correct in a FunctionSpace"""
    mesh = one_cell_mesh(cell_type)
    tdim = mesh.topology.dim
    V = FunctionSpace(mesh, ("Lagrange", order))
    v = Function(V)

    if tdim == 1:
        def f(x):
            return x[0] ** order
    elif tdim == 2:
        def f(x):
            return x[1] ** order + 2 * x[0]
    else:
        def f(x):
            return x[1] ** order + 2 * x[0] - 3 * x[2]

    v.interpolate(f)
    points = [random_point_in_cell(cell_type) for count in range(5)]
    cells = [0 for count in range(5)]
    values = v.eval(points, cells)
    for p, v in zip(points, values):
        assert np.allclose(v, f(p))


@skip_in_parallel
@pytest.mark.parametrize('order', [1, 2, 3, 4])
@pytest.mark.parametrize(
    "cell_type", [
        CellType.interval,
        CellType.triangle,
        CellType.tetrahedron,
        CellType.quadrilateral,
        CellType.hexahedron
    ])
def test_vector_interpolation(cell_type, order):
    """Test that interpolation is correct in a VectorFunctionSpace."""
    mesh = one_cell_mesh(cell_type)
    tdim = mesh.topology.dim

    V = VectorFunctionSpace(mesh, ("Lagrange", order))
    v = Function(V)

    if tdim == 1:
        def f(x):
            return x[0] ** order
    elif tdim == 2:
        def f(x):
            return (x[1], 2 * x[0] ** order)
    else:
        def f(x):
            return (x[1], 2 * x[0] ** order, 3 * x[2])

    v.interpolate(f)
    points = [random_point_in_cell(cell_type) for count in range(5)]
    cells = [0 for count in range(5)]
    values = v.eval(points, cells)
    for p, v in zip(points, values):
        assert np.allclose(f(p), v)


@skip_in_parallel
def test_mixed_interpolation():
    """Test that interpolation is correct in a MixedElement."""
    mesh = one_cell_mesh(CellType.triangle)
    A = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    B = ufl.VectorElement("Lagrange", mesh.ufl_cell(), 1)
    v = Function(FunctionSpace(mesh, ufl.MixedElement([A, B])))
    with pytest.raises(RuntimeError):
        v.interpolate(lambda x: (x[1], 2 * x[0], 3 * x[1]))


@skip_in_parallel
@pytest.mark.parametrize("cell_type",
                         [
                             CellType.triangle,
                             CellType.tetrahedron
                         ])
@pytest.mark.parametrize("order", [1, 2, 3])
def test_N1curl_interpolation(cell_type, order):
    mesh = one_cell_mesh(cell_type)
    tdim = mesh.topology.dim

    # TODO: fix higher order elements
    if tdim == 2 and order > 2:
        pytest.skip()
    if tdim == 3 and order > 1:
        pytest.skip()

    V = FunctionSpace(mesh, ("Nedelec 1st kind H(curl)", order))
    v = Function(V)

    if tdim == 2:
        def f(x):
            return (x[0] ** (order - 1), 2 * x[0] ** (order - 1) + x[1] ** (order - 1))
    else:
        def f(x):
            return (x[1] ** (order - 1), x[2] ** (order - 1), x[0] ** (order - 1) - 2 * x[1] ** (order - 1))

    v.interpolate(f)
    points = [random_point_in_cell(cell_type) for count in range(5)]
    cells = [0 for count in range(5)]
    values = v.eval(points, cells)
    assert np.allclose(values, [f(p) for p in points])


@skip_in_parallel
@pytest.mark.parametrize("cell_type", [CellType.triangle])
@pytest.mark.parametrize("order", [1, 2])
def test_N2curl_interpolation(cell_type, order):
    mesh = one_cell_mesh(cell_type)
    tdim = mesh.topology.dim

    # TODO: fix higher order elements
    if tdim == 2 and order > 1:
        pytest.skip()

    V = FunctionSpace(mesh, ("Nedelec 2nd kind H(curl)", order))
    v = Function(V)

    if tdim == 2:
        def f(x):
            return (x[1] ** order, 2 * x[0])
    else:
        def f(x):
            return (x[1] ** order + 2 * x[0], x[2] ** order, - 3 * x[2])

    v.interpolate(f)
    points = [random_point_in_cell(cell_type) for count in range(5)]
    cells = [0 for count in range(5)]
    values = v.eval(points, cells)
    assert np.allclose(values, [f(p) for p in points])
