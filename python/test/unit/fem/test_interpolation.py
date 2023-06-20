# Copyright (C) 2009-2020 Garth N. Wells, Matthew W. Scroggs and Jorgen S. Dokken
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Test that interpolation is done correctly"""

import random

import numpy as np
import pytest

import basix
import ufl
from basix.ufl import (blocked_element, custom_element, element,
                       enriched_element, mixed_element)
from dolfinx.fem import (Expression, Function, FunctionSpace,
                         VectorFunctionSpace, assemble_scalar, form)
from dolfinx.mesh import (CellType, create_mesh, create_unit_cube,
                          create_unit_square, locate_entities, meshtags)
from dolfinx import default_real_type

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
    assert len(mesh.topology.cell_types) == 1
    cell_type = mesh.topology.cell_types[0]
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
        points = np.array([[-1.], [2.]], dtype=default_real_type)
    if cell_type == CellType.triangle:
        points = np.array([[-1., -1.], [2., 0.], [0., 0.5]], dtype=default_real_type)
    elif cell_type == CellType.tetrahedron:
        points = np.array([[-1., -1., -1.], [2., 0., 0.], [0., 0.5, 0.], [0., 0., 1.]], dtype=default_real_type)
    elif cell_type == CellType.quadrilateral:
        points = np.array([[-1., 0.], [1., 0.], [-1., 1.5], [1., 1.5]], dtype=default_real_type)
    elif cell_type == CellType.hexahedron:
        points = np.array([[-1., -0.5, 0.], [1., -0.5, 0.], [-1., 1.5, 0.],
                           [1., 1.5, 0.], [0., -0.5, 1.], [1., -0.5, 1.],
                           [-1., 1.5, 1.], [1., 1.5, 1.]], dtype=default_real_type)
    num_points = len(points)

    # Randomly number the points and create the mesh
    order = list(range(num_points))
    random.shuffle(order)
    ordered_points = np.zeros(points.shape, dtype=default_real_type)
    for i, j in enumerate(order):
        ordered_points[j] = points[i]
    cells = np.array([order])

    domain = ufl.Mesh(element("Lagrange", cell_type.name, 1, rank=1))
    return create_mesh(MPI.COMM_WORLD, cells, ordered_points, domain)


def two_cell_mesh(cell_type):
    if cell_type == CellType.interval:
        points = np.array([[0.], [1.], [-1.]], dtype=default_real_type)
        cells = [[0, 1], [0, 2]]
    if cell_type == CellType.triangle:
        # Define equilateral triangles with area 1
        root = 3 ** 0.25  # 4th root of 3
        points = np.array([[0., 0.], [2 / root, 0.], [1 / root, root], [1 / root, -root]], dtype=default_real_type)
        cells = [[0, 1, 2], [1, 0, 3]]
    elif cell_type == CellType.tetrahedron:
        # Define regular tetrahedra with volume 1
        s = 2 ** 0.5 * 3 ** (1 / 3)  # side length
        points = np.array([[0., 0., 0.], [s, 0., 0.],
                           [s / 2, s * np.sqrt(3) / 2, 0.],
                           [s / 2, s / 2 / np.sqrt(3), s * np.sqrt(2 / 3)],
                           [s / 2, s / 2 / np.sqrt(3), -s * np.sqrt(2 / 3)]], dtype=default_real_type)
        cells = [[0, 1, 2, 3], [0, 2, 1, 4]]
    elif cell_type == CellType.quadrilateral:
        # Define unit quadrilaterals (area 1)
        points = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.], [0., -1.], [1., -1.]], dtype=default_real_type)
        cells = [[0, 1, 2, 3], [5, 1, 4, 0]]
    elif cell_type == CellType.hexahedron:
        # Define unit hexahedra (volume 1)
        points = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
                           [1., 1., 0.], [0., 0., 1.], [1., 0., 1.],
                           [0., 1., 1.], [1., 1., 1.], [0., 0., -1.],
                           [1., 0., -1.], [0., 1., -1.], [1., 1., -1.]], dtype=default_real_type)
        cells = [[0, 1, 2, 3, 4, 5, 6, 7], [9, 11, 8, 10, 1, 3, 0, 2]]

    domain = ufl.Mesh(element("Lagrange", cell_type.name, 1, rank=1))
    mesh = create_mesh(MPI.COMM_WORLD, cells, points, domain)
    return mesh


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
    points = np.asarray(points, dtype=default_real_type)
    cells = [0 for count in range(5)]
    values = v.eval(points, cells)
    for p, val in zip(points, values):
        assert np.allclose(val, f(p), atol=1.0e-5)


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
        assert np.allclose(val, f(p), atol=1.0e-5)


@pytest.mark.skip_in_parallel
@parametrize_cell_types
@pytest.mark.parametrize("order", range(1, 5))
def test_Lagrange_interpolation(cell_type, order):
    """Test that interpolation is correct in a FunctionSpace"""
    mesh = one_cell_mesh(cell_type)
    V = FunctionSpace(mesh, ("Lagrange", order))
    run_scalar_test(V, order)


@pytest.mark.skip_in_parallel
@pytest.mark.parametrize("cell_type", [CellType.interval, CellType.quadrilateral, CellType.hexahedron])
@pytest.mark.parametrize("order", range(1, 5))
def test_serendipity_interpolation(cell_type, order):
    """Test that interpolation is correct in a FunctionSpace"""
    mesh = one_cell_mesh(cell_type)
    V = FunctionSpace(mesh, ("S", order))
    run_scalar_test(V, order)


@pytest.mark.skip_in_parallel
@parametrize_cell_types
@pytest.mark.parametrize('order', range(1, 5))
def test_vector_interpolation(cell_type, order):
    """Test that interpolation is correct in a VectorFunctionSpace."""
    mesh = one_cell_mesh(cell_type)
    V = VectorFunctionSpace(mesh, ("Lagrange", order))
    run_vector_test(V, order)


@pytest.mark.skip_in_parallel
@pytest.mark.parametrize("cell_type", [CellType.triangle, CellType.tetrahedron])
@pytest.mark.parametrize("order", range(1, 5))
def test_N1curl_interpolation(cell_type, order):
    random.seed(8)
    mesh = one_cell_mesh(cell_type)
    V = FunctionSpace(mesh, ("Nedelec 1st kind H(curl)", order))
    run_vector_test(V, order - 1)


@pytest.mark.skip_in_parallel
@pytest.mark.parametrize("cell_type", [CellType.triangle])
@pytest.mark.parametrize("order", [1, 2])
def test_N2curl_interpolation(cell_type, order):
    mesh = one_cell_mesh(cell_type)
    V = FunctionSpace(mesh, ("Nedelec 2nd kind H(curl)", order))
    run_vector_test(V, order)


@pytest.mark.skip_in_parallel
@pytest.mark.parametrize("cell_type", [CellType.quadrilateral])
@pytest.mark.parametrize("order", range(1, 5))
def test_RTCE_interpolation(cell_type, order):
    random.seed(8)
    mesh = one_cell_mesh(cell_type)
    V = FunctionSpace(mesh, ("RTCE", order))
    run_vector_test(V, order - 1)


@pytest.mark.skip_in_parallel
@pytest.mark.parametrize("cell_type", [CellType.hexahedron])
@pytest.mark.parametrize("order", range(1, 5))
def test_NCE_interpolation(cell_type, order):
    random.seed(8)
    mesh = one_cell_mesh(cell_type)
    V = FunctionSpace(mesh, ("NCE", order))
    run_vector_test(V, order - 1)


def test_mixed_sub_interpolation():
    """Test interpolation of sub-functions"""
    mesh = create_unit_cube(MPI.COMM_WORLD, 3, 3, 3)

    def f(x):
        return np.vstack((10 + x[0], -10 - x[1], 25 + x[0]))

    P2 = element("Lagrange", mesh.basix_cell(), 2, rank=1)
    P1 = element("Lagrange", mesh.basix_cell(), 1)
    for i, P in enumerate((mixed_element([P2, P1]), mixed_element([P1, P2]))):
        W = FunctionSpace(mesh, P)
        U = Function(W)
        U.sub(i).interpolate(f)

        # Same element
        V = FunctionSpace(mesh, P2)
        u, v = Function(V), Function(V)
        u.interpolate(U.sub(i))
        v.interpolate(f)
        assert np.allclose(u.vector.array, v.vector.array)

        # Same map, different elements
        V = VectorFunctionSpace(mesh, ("Lagrange", 1))
        u, v = Function(V), Function(V)
        u.interpolate(U.sub(i))
        v.interpolate(f)
        assert np.allclose(u.vector.array, v.vector.array)

        # Different maps (0)
        V = FunctionSpace(mesh, ("N1curl", 1))
        u, v = Function(V), Function(V)
        u.interpolate(U.sub(i))
        v.interpolate(f)
        assert np.allclose(u.vector.array, v.vector.array, atol=1.0e-6)

        # Different maps (1)
        V = FunctionSpace(mesh, ("RT", 2))
        u, v = Function(V), Function(V)
        u.interpolate(U.sub(i))
        v.interpolate(f)
        assert np.allclose(u.vector.array, v.vector.array, atol=1.0e-6)

        # Test with wrong shape
        V0 = FunctionSpace(mesh, P.sub_elements()[0])
        V1 = FunctionSpace(mesh, P.sub_elements()[1])
        v0, v1 = Function(V0), Function(V1)
        with pytest.raises(RuntimeError):
            v0.interpolate(U.sub(1))
        with pytest.raises(RuntimeError):
            v1.interpolate(U.sub(0))


@pytest.mark.skip_in_parallel
def test_mixed_interpolation():
    """Test that mixed interpolation raised an exception."""
    mesh = one_cell_mesh(CellType.triangle)
    A = element("Lagrange", mesh.basix_cell(), 1)
    B = element("Lagrange", mesh.basix_cell(), 1, rank=1)
    v = Function(FunctionSpace(mesh, mixed_element([A, B])))
    with pytest.raises(RuntimeError):
        v.interpolate(lambda x: (x[1], 2 * x[0], 3 * x[1]))


@pytest.mark.parametrize("order1", [2, 3, 4])
@pytest.mark.parametrize("order2", [2, 3, 4])
def test_interpolation_nedelec(order1, order2):
    mesh = create_unit_cube(MPI.COMM_WORLD, 2, 2, 2)
    V = FunctionSpace(mesh, ("N1curl", order1))
    V1 = FunctionSpace(mesh, ("N1curl", order2))
    u, v = Function(V), Function(V1)

    # The expression "lambda x: x" is contained in the N1curl function
    # space for order > 1
    u.interpolate(lambda x: x)
    v.interpolate(u)
    assert np.isclose(assemble_scalar(form(ufl.inner(u - v, u - v) * ufl.dx)), 0)

    # The target expression is also contained in N2curl space of any
    # order
    V2 = FunctionSpace(mesh, ("N2curl", 1))
    w = Function(V2)
    w.interpolate(u)
    assert np.isclose(assemble_scalar(form(ufl.inner(u - w, u - w) * ufl.dx)), 0)


@pytest.mark.parametrize("tdim", [2, 3])
@pytest.mark.parametrize("order", [1, 2, 3])
def test_interpolation_dg_to_n1curl(tdim, order):
    if tdim == 2:
        mesh = create_unit_square(MPI.COMM_WORLD, 5, 5)
    else:
        mesh = create_unit_cube(MPI.COMM_WORLD, 2, 2, 2)
    V = VectorFunctionSpace(mesh, ("DG", order))
    V1 = FunctionSpace(mesh, ("N1curl", order + 1))
    u, v = Function(V), Function(V1)
    u.interpolate(lambda x: x[:tdim] ** order)
    v.interpolate(u)
    s = assemble_scalar(form(ufl.inner(u - v, u - v) * ufl.dx))
    assert np.isclose(s, 0)


@pytest.mark.parametrize("tdim", [2, 3])
@pytest.mark.parametrize("order", [1, 2, 3])
def test_interpolation_n1curl_to_dg(tdim, order):
    if tdim == 2:
        mesh = create_unit_square(MPI.COMM_WORLD, 5, 5)
    else:
        mesh = create_unit_cube(MPI.COMM_WORLD, 2, 2, 2)
    V = FunctionSpace(mesh, ("N1curl", order + 1))
    V1 = VectorFunctionSpace(mesh, ("DG", order))
    u, v = Function(V), Function(V1)
    u.interpolate(lambda x: x[:tdim] ** order)
    v.interpolate(u)
    s = assemble_scalar(form(ufl.inner(u - v, u - v) * ufl.dx))
    assert np.isclose(s, 0)


@pytest.mark.parametrize("tdim", [2, 3])
@pytest.mark.parametrize("order", [1, 2, 3])
def test_interpolation_n2curl_to_bdm(tdim, order):
    if tdim == 2:
        mesh = create_unit_square(MPI.COMM_WORLD, 5, 5)
    else:
        mesh = create_unit_cube(MPI.COMM_WORLD, 2, 2, 2)
    V = FunctionSpace(mesh, ("N2curl", order))
    V1 = FunctionSpace(mesh, ("BDM", order))
    u, v = Function(V), Function(V1)
    u.interpolate(lambda x: x[:tdim] ** order)
    v.interpolate(u)
    s = assemble_scalar(form(ufl.inner(u - v, u - v) * ufl.dx))
    assert np.isclose(s, 0)


@pytest.mark.parametrize("order1", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("order2", [1, 2, 3])
def test_interpolation_p2p(order1, order2):
    mesh = create_unit_cube(MPI.COMM_WORLD, 2, 2, 2)
    V = FunctionSpace(mesh, ("Lagrange", order1))
    V1 = FunctionSpace(mesh, ("Lagrange", order2))
    u, v = Function(V), Function(V1)

    u.interpolate(lambda x: x[0])
    v.interpolate(u)

    s = assemble_scalar(form(ufl.inner(u - v, u - v) * ufl.dx))
    assert np.isclose(s, 0)

    DG = FunctionSpace(mesh, ("DG", order2))
    w = Function(DG)
    w.interpolate(u)
    s = assemble_scalar(form(ufl.inner(u - w, u - w) * ufl.dx))
    assert np.isclose(s, 0)


@pytest.mark.parametrize("order1", [1, 2, 3])
@pytest.mark.parametrize("order2", [1, 2])
def test_interpolation_vector_elements(order1, order2):
    mesh = create_unit_cube(MPI.COMM_WORLD, 2, 2, 2)
    V = VectorFunctionSpace(mesh, ("Lagrange", order1))
    V1 = VectorFunctionSpace(mesh, ("Lagrange", order2))
    u, v = Function(V), Function(V1)

    u.interpolate(lambda x: x)
    v.interpolate(u)

    s = assemble_scalar(form(ufl.inner(u - v, u - v) * ufl.dx))
    assert np.isclose(s, 0)

    DG = VectorFunctionSpace(mesh, ("DG", order2))
    w = Function(DG)
    w.interpolate(u)
    s = assemble_scalar(form(ufl.inner(u - w, u - w) * ufl.dx))
    assert np.isclose(s, 0)


@pytest.mark.skip_in_parallel
def test_interpolation_non_affine():
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0], [1, 2, 0],
                       [0, 0, 3], [1, 0, 3], [0, 2, 3], [1, 2, 3],
                       [0.5, 0, 0], [0, 1, 0], [0, 0, 1.5], [1, 1, 0],
                       [1, 0, 1.5], [0.5, 2, 0], [0, 2, 1.5], [1, 2, 1.5],
                       [0.5, 0, 3], [0, 1, 3], [1, 1, 3], [0.5, 2, 3],
                       [0.5, 1, 0], [0.5, 0, 1.5], [0, 1, 1.5], [1, 1, 1.5],
                       [0.5, 2, 1.5], [0.5, 1, 3], [0.5, 1, 1.5]], dtype=default_real_type)

    cells = np.array([range(len(points))], dtype=np.int32)
    domain = ufl.Mesh(element("Lagrange", "hexahedron", 2, rank=1))
    mesh = create_mesh(MPI.COMM_WORLD, cells, points, domain)
    W = FunctionSpace(mesh, ("NCE", 1))
    V = FunctionSpace(mesh, ("NCE", 2))
    w, v = Function(W), Function(V)
    w.interpolate(lambda x: x)
    v.interpolate(w)
    s = assemble_scalar(form(ufl.inner(w - v, w - v) * ufl.dx))
    assert np.isclose(s, 0)


@pytest.mark.skip_in_parallel
def test_interpolation_non_affine_nonmatching_maps():
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0], [1, 2, 0],
                       [0, 0, 3], [1, 0, 3], [0, 2, 3], [1, 2, 3],
                       [0.5, 0, 0], [0, 1, 0], [0, 0, 1.5], [1, 1, 0],
                       [1, 0, 1.5], [0.5, 2, 0], [0, 2, 1.5], [1, 2, 1.5],
                       [0.5, 0, 3], [0, 1, 3], [1, 1, 3], [0.5, 2, 3],
                       [0.5, 1, 0], [0.5, -0.1, 1.5], [0, 1, 1.5], [1, 1, 1.5],
                       [0.5, 2, 1.5], [0.5, 1, 3], [0.5, 1, 1.5]], dtype=default_real_type)

    cells = np.array([range(len(points))], dtype=np.int32)
    domain = ufl.Mesh(element("Lagrange", "hexahedron", 2, rank=1))
    mesh = create_mesh(MPI.COMM_WORLD, cells, points, domain)
    W = VectorFunctionSpace(mesh, ("DG", 1))
    V = FunctionSpace(mesh, ("NCE", 4))
    w, v = Function(W), Function(V)
    w.interpolate(lambda x: x)
    v.interpolate(w)
    s = assemble_scalar(form(ufl.inner(w - v, w - v) * ufl.dx))
    assert np.isclose(s, 0)


@pytest.mark.parametrize("order", [2, 3, 4])
@pytest.mark.parametrize("dim", [2, 3])
def test_nedelec_spatial(order, dim):
    if dim == 2:
        mesh = create_unit_square(MPI.COMM_WORLD, 4, 4)
    elif dim == 3:
        mesh = create_unit_cube(MPI.COMM_WORLD, 2, 2, 2)

    V = FunctionSpace(mesh, ("N1curl", order))
    u = Function(V)
    x = ufl.SpatialCoordinate(mesh)

    # The expression (x,y,z) is contained in the N1curl function space
    # order>1
    f_ex = x
    f = Expression(f_ex, V.element.interpolation_points())
    u.interpolate(f)
    assert np.isclose(np.abs(assemble_scalar(form(ufl.inner(u - f_ex, u - f_ex) * ufl.dx))), 0)

    # The target expression is also contained in N2curl space of any
    # order
    V2 = FunctionSpace(mesh, ("N2curl", 1))
    w = Function(V2)
    f2 = Expression(f_ex, V2.element.interpolation_points())
    w.interpolate(f2)
    assert np.isclose(np.abs(assemble_scalar(form(ufl.inner(w - f_ex, w - f_ex) * ufl.dx))), 0)


@pytest.mark.parametrize("order", [1, 2, 3, 4])
@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("affine", [True, False])
def test_vector_interpolation_spatial(order, dim, affine):
    if dim == 2:
        ct = CellType.triangle if affine else CellType.quadrilateral
        mesh = create_unit_square(MPI.COMM_WORLD, 3, 4, ct)
    elif dim == 3:
        ct = CellType.tetrahedron if affine else CellType.hexahedron
        mesh = create_unit_cube(MPI.COMM_WORLD, 3, 2, 2, ct)

    V = VectorFunctionSpace(mesh, ("Lagrange", order))
    u = Function(V)
    x = ufl.SpatialCoordinate(mesh)

    # The expression (x,y,z)^n is contained in space
    f = ufl.as_vector([x[i]**order for i in range(dim)])
    u.interpolate(Expression(f, V.element.interpolation_points()))
    assert np.isclose(np.abs(assemble_scalar(form(ufl.inner(u - f, u - f) * ufl.dx))), 0)


@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_2D_lagrange_to_curl(order):
    mesh = create_unit_square(MPI.COMM_WORLD, 3, 4)
    V = FunctionSpace(mesh, ("N1curl", order))
    u = Function(V)

    W = FunctionSpace(mesh, ("Lagrange", order))
    u0 = Function(W)
    u0.interpolate(lambda x: -x[1])
    u1 = Function(W)
    u1.interpolate(lambda x: x[0])

    f = ufl.as_vector((u0, u1))
    f_expr = Expression(f, V.element.interpolation_points())
    u.interpolate(f_expr)
    x = ufl.SpatialCoordinate(mesh)
    f_ex = ufl.as_vector((-x[1], x[0]))
    assert np.isclose(np.abs(assemble_scalar(form(ufl.inner(u - f_ex, u - f_ex) * ufl.dx))), 0)


@pytest.mark.parametrize("order", [2, 3, 4])
def test_de_rahm_2D(order):
    mesh = create_unit_square(MPI.COMM_WORLD, 3, 4)
    W = FunctionSpace(mesh, ("Lagrange", order))
    w = Function(W)
    w.interpolate(lambda x: x[0] + x[0] * x[1] + 2 * x[1]**2)

    g = ufl.grad(w)
    Q = FunctionSpace(mesh, ("N2curl", order - 1))
    q = Function(Q)
    q.interpolate(Expression(g, Q.element.interpolation_points()))

    x = ufl.SpatialCoordinate(mesh)
    g_ex = ufl.as_vector((1 + x[1], 4 * x[1] + x[0]))
    assert np.isclose(np.abs(assemble_scalar(form(ufl.inner(q - g_ex, q - g_ex) * ufl.dx))), 0)

    V = FunctionSpace(mesh, ("BDM", order - 1))
    v = Function(V)

    def curl2D(u):
        return ufl.as_vector((ufl.Dx(u[1], 0), - ufl.Dx(u[0], 1)))

    v.interpolate(Expression(curl2D(ufl.grad(w)), V.element.interpolation_points()))
    h_ex = ufl.as_vector((1, -1))
    assert np.isclose(np.abs(assemble_scalar(form(ufl.inner(v - h_ex, v - h_ex) * ufl.dx))), 0, atol=1.0e-6)


@pytest.mark.parametrize("order", [1, 2, 3, 4])
@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("affine", [True, False])
@pytest.mark.parametrize("callable_", [True, False])
def test_interpolate_subset(order, dim, affine, callable_):
    if dim == 2:
        ct = CellType.triangle if affine else CellType.quadrilateral
        mesh = create_unit_square(MPI.COMM_WORLD, 3, 4, ct)
    elif dim == 3:
        ct = CellType.tetrahedron if affine else CellType.hexahedron
        mesh = create_unit_cube(MPI.COMM_WORLD, 3, 2, 2, ct)

    V = FunctionSpace(mesh, ("DG", order))
    u = Function(V)

    cells = locate_entities(mesh, mesh.topology.dim, lambda x: x[1] <= 0.5 + 1e-10)
    num_local_cells = mesh.topology.index_map(mesh.topology.dim).size_local
    cells_local = cells[cells < num_local_cells]

    x = ufl.SpatialCoordinate(mesh)
    f = x[1]**order
    if not callable_:
        expr = Expression(f, V.element.interpolation_points())
        u.interpolate(expr, cells_local)
    else:
        u.interpolate(lambda x: x[1]**order, cells_local)
    mt = meshtags(mesh, mesh.topology.dim, cells_local, np.ones(cells_local.size, dtype=np.int32))
    dx = ufl.Measure("dx", domain=mesh, subdomain_data=mt)
    assert np.isclose(np.abs(form(assemble_scalar(form(ufl.inner(u - f, u - f) * dx(1))))), 0)
    integral = mesh.comm.allreduce(assemble_scalar(form(u * dx)), op=MPI.SUM)
    assert np.isclose(integral, 1 / (order + 1) * 0.5**(order + 1), 0, atol=1.0e-6)


def test_interpolate_callable():
    """Test interpolation with callables"""
    mesh = create_unit_square(MPI.COMM_WORLD, 2, 1)
    V = FunctionSpace(mesh, ("Lagrange", 2))
    u0, u1 = Function(V), Function(V)

    numba = pytest.importorskip("numba")

    @numba.njit
    def f(x):
        return x[0]

    u0.interpolate(lambda x: x[0])
    u1.interpolate(f)
    assert np.allclose(u0.x.array, u1.x.array)
    with pytest.raises(RuntimeError):
        u0.interpolate(lambda x: np.vstack([x[0], x[1]]))


@pytest.mark.parametrize("bound", [1.5, 0.5])
def test_interpolate_callable_subset(bound):
    """Test interpolation on subsets with callables"""
    mesh = create_unit_square(MPI.COMM_WORLD, 3, 4)
    cells = locate_entities(mesh, mesh.topology.dim, lambda x: x[1] <= bound + 1e-10)
    num_local_cells = mesh.topology.index_map(mesh.topology.dim).size_local
    cells_local = cells[cells < num_local_cells]

    V = FunctionSpace(mesh, ("DG", 2))
    u0, u1 = Function(V), Function(V)

    x = ufl.SpatialCoordinate(mesh)
    f = x[0]
    expr = Expression(f, V.element.interpolation_points())
    u0.interpolate(lambda x: x[0], cells_local)
    u1.interpolate(expr, cells_local)
    assert np.allclose(u0.x.array, u1.x.array)


@pytest.mark.parametrize("scalar_element", [
    element("P", "triangle", 1),
    element("P", "triangle", 2),
    element("P", "triangle", 3),
    element("Q", "quadrilateral", 1),
    element("Q", "quadrilateral", 2),
    element("Q", "quadrilateral", 3),
    element("S", "quadrilateral", 1),
    element("S", "quadrilateral", 2),
    element("S", "quadrilateral", 3),
    enriched_element([element("P", "triangle", 1), element("Bubble", "triangle", 3)]),
    enriched_element([element("P", "quadrilateral", 1), element("Bubble", "quadrilateral", 2)]),
])
def test_vector_element_interpolation(scalar_element):
    """Test interpolation into a range of vector elements."""
    mesh = create_unit_square(MPI.COMM_WORLD, 10, 10, getattr(CellType, scalar_element.cell().cellname()))
    V = FunctionSpace(mesh, blocked_element(scalar_element, shape=(2, )))
    u = Function(V)
    u.interpolate(lambda x: (x[0], x[1]))
    u2 = Function(V)
    u2.sub(0).interpolate(lambda x: x[0])
    u2.sub(1).interpolate(lambda x: x[1])
    assert np.allclose(u2.x.array, u.x.array)


def test_custom_vector_element():
    """Test interpolation into an element with a value size that uses an identity map."""
    mesh = create_unit_square(MPI.COMM_WORLD, 10, 10)

    wcoeffs = np.eye(6)

    x = [[], [], [], []]
    x[0].append(np.array([[0., 0.]]))
    x[0].append(np.array([[1., 0.]]))
    x[0].append(np.array([[0., 1.]]))
    for _ in range(3):
        x[1].append(np.zeros((0, 2)))
    x[2].append(np.zeros((0, 2)))

    M = [[], [], [], []]
    for _ in range(3):
        M[0].append(np.array([[[[1.]], [[0.]]], [[[0.]], [[1.]]]]))
    for _ in range(3):
        M[1].append(np.zeros((0, 2, 0, 1)))
    M[2].append(np.zeros((0, 2, 0, 1)))

    e = custom_element(
        basix.CellType.triangle, [2], wcoeffs, x, M, 0, basix.MapType.identity,
        basix.SobolevSpace.H1, False, 1, 1)

    V = FunctionSpace(mesh, e)
    W = VectorFunctionSpace(mesh, ("Lagrange", 1))

    v = Function(V)
    w = Function(W)
    v.interpolate(lambda x: (x[0], x[1]))
    w.interpolate(lambda x: (x[0], x[1]))

    assert np.isclose(np.abs(assemble_scalar(form(ufl.inner(v - w, v - w) * ufl.dx))), 0)


@pytest.mark.skip_in_parallel
@pytest.mark.parametrize("cell_type", [CellType.triangle, CellType.tetrahedron])
@pytest.mark.parametrize("order", range(1, 5))
def test_mixed_interpolation_permuting(cell_type, order):
    random.seed(8)
    mesh = two_cell_mesh(cell_type)

    def g(x):
        return np.sin(x[1]) + 2 * x[0]

    x = ufl.SpatialCoordinate(mesh)
    dgdy = ufl.cos(x[1])

    curl_el = element("N1curl", mesh.basix_cell(), 1)
    vlag_el = element("Lagrange", mesh.basix_cell(), 1, rank=1)
    lagr_el = element("Lagrange", mesh.basix_cell(), order)

    V = FunctionSpace(mesh, mixed_element([curl_el, lagr_el]))
    Eb_m = Function(V)
    Eb_m.sub(1).interpolate(g)
    diff = Eb_m[2].dx(1) - dgdy
    error = assemble_scalar(form(ufl.dot(diff, diff) * ufl.dx))

    V = FunctionSpace(mesh, mixed_element([vlag_el, lagr_el]))
    Eb_m = Function(V)
    Eb_m.sub(1).interpolate(g)
    diff = Eb_m[2].dx(1) - dgdy
    error2 = assemble_scalar(form(ufl.dot(diff, diff) * ufl.dx))
    assert np.isclose(error, error2)
