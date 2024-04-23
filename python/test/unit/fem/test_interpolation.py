# Copyright (C) 2009-2020 Garth N. Wells, Matthew W. Scroggs and Jorgen S. Dokken
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Test that interpolation is done correctly"""

import random

from mpi4py import MPI

import numpy as np
import pytest

import basix
import ufl
from basix.ufl import blocked_element, custom_element, element, enriched_element, mixed_element
from dolfinx import default_real_type, default_scalar_type
from dolfinx.fem import (
    Expression,
    Function,
    assemble_scalar,
    create_nonmatching_meshes_interpolation_data,
    form,
    functionspace,
)
from dolfinx.geometry import bb_tree, compute_collisions_points
from dolfinx.mesh import (
    CellType,
    create_mesh,
    create_rectangle,
    create_submesh,
    create_unit_cube,
    create_unit_square,
    locate_entities,
    locate_entities_boundary,
    meshtags,
)

parametrize_cell_types = pytest.mark.parametrize(
    "cell_type",
    [
        CellType.interval,
        CellType.triangle,
        CellType.tetrahedron,
        CellType.quadrilateral,
        CellType.hexahedron,
    ],
)


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
        axes = (mesh.geometry.x[1],)
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

    return tuple(
        origin[i] + sum((axis[i] - origin[i]) * p for axis, p in zip(axes, point)) for i in range(3)
    )


def one_cell_mesh(cell_type):
    if cell_type == CellType.interval:
        points = np.array([[-1.0], [2.0]], dtype=default_real_type)
    if cell_type == CellType.triangle:
        points = np.array([[-1.0, -1.0], [2.0, 0.0], [0.0, 0.5]], dtype=default_real_type)
    elif cell_type == CellType.tetrahedron:
        points = np.array(
            [[-1.0, -1.0, -1.0], [2.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0]],
            dtype=default_real_type,
        )
    elif cell_type == CellType.quadrilateral:
        points = np.array(
            [[-1.0, 0.0], [1.0, 0.0], [-1.0, 1.5], [1.0, 1.5]], dtype=default_real_type
        )
    elif cell_type == CellType.hexahedron:
        points = np.array(
            [
                [-1.0, -0.5, 0.0],
                [1.0, -0.5, 0.0],
                [-1.0, 1.5, 0.0],
                [1.0, 1.5, 0.0],
                [0.0, -0.5, 1.0],
                [1.0, -0.5, 1.0],
                [-1.0, 1.5, 1.0],
                [1.0, 1.5, 1.0],
            ],
            dtype=default_real_type,
        )
    num_points = len(points)

    # Randomly number the points and create the mesh
    order = list(range(num_points))
    random.shuffle(order)
    ordered_points = np.zeros(points.shape, dtype=default_real_type)
    for i, j in enumerate(order):
        ordered_points[j] = points[i]
    cells = np.array([order])
    domain = ufl.Mesh(
        element(
            "Lagrange", cell_type.name, 1, shape=(ordered_points.shape[1],), dtype=default_real_type
        )
    )
    return create_mesh(MPI.COMM_WORLD, cells, ordered_points, domain)


def two_cell_mesh(cell_type):
    if cell_type == CellType.interval:
        points = np.array([[0.0], [1.0], [-1.0]], dtype=default_real_type)
        cells = [[0, 1], [0, 2]]
    if cell_type == CellType.triangle:
        # Define equilateral triangles with area 1
        root = 3**0.25  # 4th root of 3
        points = np.array(
            [[0.0, 0.0], [2 / root, 0.0], [1 / root, root], [1 / root, -root]],
            dtype=default_real_type,
        )
        cells = [[0, 1, 2], [1, 0, 3]]
    elif cell_type == CellType.tetrahedron:
        # Define regular tetrahedra with volume 1
        s = 2**0.5 * 3 ** (1 / 3)  # side length
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [s, 0.0, 0.0],
                [s / 2, s * np.sqrt(3) / 2, 0.0],
                [s / 2, s / 2 / np.sqrt(3), s * np.sqrt(2 / 3)],
                [s / 2, s / 2 / np.sqrt(3), -s * np.sqrt(2 / 3)],
            ],
            dtype=default_real_type,
        )
        cells = [[0, 1, 2, 3], [0, 2, 1, 4]]
    elif cell_type == CellType.quadrilateral:
        # Define unit quadrilaterals (area 1)
        points = np.array(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, -1.0], [1.0, -1.0]],
            dtype=default_real_type,
        )
        cells = [[0, 1, 2, 3], [5, 1, 4, 0]]
    elif cell_type == CellType.hexahedron:
        # Define unit hexahedra (volume 1)
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [0.0, 0.0, -1.0],
                [1.0, 0.0, -1.0],
                [0.0, 1.0, -1.0],
                [1.0, 1.0, -1.0],
            ],
            dtype=default_real_type,
        )
        cells = [[0, 1, 2, 3, 4, 5, 6, 7], [9, 11, 8, 10, 1, 3, 0, 2]]

    domain = ufl.Mesh(
        element("Lagrange", cell_type.name, 1, shape=(points.shape[1],), dtype=default_real_type)
    )
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
            return (
                x[1] ** poly_order + 2 * x[0] ** min(poly_order, 1) - 3 * x[2] ** min(poly_order, 2)
            )

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
            return (
                x[1] ** min(poly_order, 1),
                2 * x[0] ** poly_order,
                3 * x[2] ** min(poly_order, 2),
            )

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
    """Test that interpolation is correct in a function space"""
    mesh = one_cell_mesh(cell_type)
    V = functionspace(mesh, ("Lagrange", order))
    run_scalar_test(V, order)


@pytest.mark.skip_in_parallel
@pytest.mark.parametrize(
    "cell_type", [CellType.interval, CellType.quadrilateral, CellType.hexahedron]
)
@pytest.mark.parametrize("order", range(1, 5))
def test_serendipity_interpolation(cell_type, order):
    """Test that interpolation is correct in a function space"""
    mesh = one_cell_mesh(cell_type)
    V = functionspace(mesh, ("S", order))
    run_scalar_test(V, order)


@pytest.mark.skip_in_parallel
@parametrize_cell_types
@pytest.mark.parametrize("order", range(1, 5))
def test_vector_interpolation(cell_type, order):
    """Test that interpolation is correct in a blocked (vector) function space."""
    mesh = one_cell_mesh(cell_type)
    gdim = mesh.geometry.dim
    V = functionspace(mesh, ("Lagrange", order, (gdim,)))
    run_vector_test(V, order)


@pytest.mark.skip_in_parallel
@pytest.mark.parametrize("cell_type", [CellType.triangle, CellType.tetrahedron])
@pytest.mark.parametrize("order", range(1, 5))
def test_N1curl_interpolation(cell_type, order):
    random.seed(8)
    mesh = one_cell_mesh(cell_type)
    V = functionspace(mesh, ("Nedelec 1st kind H(curl)", order))
    run_vector_test(V, order - 1)


@pytest.mark.skip_in_parallel
@pytest.mark.parametrize("cell_type", [CellType.triangle])
@pytest.mark.parametrize("order", [1, 2])
def test_N2curl_interpolation(cell_type, order):
    mesh = one_cell_mesh(cell_type)
    V = functionspace(mesh, ("Nedelec 2nd kind H(curl)", order))
    run_vector_test(V, order)


@pytest.mark.skip_in_parallel
@pytest.mark.parametrize("cell_type", [CellType.quadrilateral])
@pytest.mark.parametrize("order", range(1, 5))
def test_RTCE_interpolation(cell_type, order):
    random.seed(8)
    mesh = one_cell_mesh(cell_type)
    V = functionspace(mesh, ("RTCE", order))
    run_vector_test(V, order - 1)


@pytest.mark.skip_in_parallel
@pytest.mark.parametrize("cell_type", [CellType.hexahedron])
@pytest.mark.parametrize("order", range(1, 5))
def test_NCE_interpolation(cell_type, order):
    random.seed(8)
    mesh = one_cell_mesh(cell_type)
    V = functionspace(mesh, ("NCE", order))
    run_vector_test(V, order - 1)


def test_mixed_sub_interpolation():
    """Test interpolation of sub-functions"""
    mesh = create_unit_cube(MPI.COMM_WORLD, 3, 3, 3)

    def f(x):
        return np.vstack((10 + x[0], -10 - x[1], 25 + x[0]))

    P2 = element("Lagrange", mesh.basix_cell(), 2, shape=(mesh.geometry.dim,))
    P1 = element("Lagrange", mesh.basix_cell(), 1)
    for i, P in enumerate((mixed_element([P2, P1]), mixed_element([P1, P2]))):
        W = functionspace(mesh, P)
        U = Function(W)
        U.sub(i).interpolate(f)

        # Same element
        V = functionspace(mesh, P2)
        u, v = Function(V), Function(V)
        u.interpolate(U.sub(i))
        v.interpolate(f)
        assert np.allclose(u.x.array, v.x.array)

        # Same map, different elements
        gdim = mesh.geometry.dim
        V = functionspace(mesh, ("Lagrange", 1, (gdim,)))
        u, v = Function(V), Function(V)
        u.interpolate(U.sub(i))
        v.interpolate(f)
        assert np.allclose(u.x.array, v.x.array)

        # Different maps (0)
        V = functionspace(mesh, ("N1curl", 1))
        u, v = Function(V), Function(V)
        u.interpolate(U.sub(i))
        v.interpolate(f)
        atol = 5 * np.finfo(u.x.array.dtype).resolution
        assert np.allclose(u.x.array, v.x.array, atol=atol)

        # Different maps (1)
        V = functionspace(mesh, ("RT", 2))
        u, v = Function(V), Function(V)
        u.interpolate(U.sub(i))
        v.interpolate(f)
        atol = 5 * np.finfo(u.x.array.dtype).resolution
        assert np.allclose(u.x.array, v.x.array, atol=atol)

        # Test with wrong shape
        V0 = functionspace(mesh, P.sub_elements[0])
        V1 = functionspace(mesh, P.sub_elements[1])
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
    B = element("Lagrange", mesh.basix_cell(), 1, shape=(mesh.geometry.dim,))
    v = Function(functionspace(mesh, mixed_element([A, B])))
    with pytest.raises(RuntimeError):
        v.interpolate(lambda x: (x[1], 2 * x[0], 3 * x[1]))


@pytest.mark.parametrize("order1", [2, 3, 4])
@pytest.mark.parametrize("order2", [2, 3, 4])
def test_interpolation_nedelec(order1, order2):
    mesh = create_unit_cube(MPI.COMM_WORLD, 2, 2, 2)
    V = functionspace(mesh, ("N1curl", order1))
    V1 = functionspace(mesh, ("N1curl", order2))
    u, v = Function(V), Function(V1)

    # The expression "lambda x: x" is contained in the N1curl function
    # space for order > 1
    u.interpolate(lambda x: x)
    v.interpolate(u)
    assert assemble_scalar(form(ufl.inner(u - v, u - v) * ufl.dx)) == pytest.approx(0, abs=1.0e-10)

    # The target expression is also contained in N2curl space of any
    # order
    V2 = functionspace(mesh, ("N2curl", 1))
    w = Function(V2)
    w.interpolate(u)
    assert assemble_scalar(form(ufl.inner(u - w, u - w) * ufl.dx)) == pytest.approx(0, abs=1.0e-10)


@pytest.mark.parametrize("tdim", [2, 3])
@pytest.mark.parametrize("order", [1, 2, 3])
def test_interpolation_dg_to_n1curl(tdim, order):
    if tdim == 2:
        mesh = create_unit_square(MPI.COMM_WORLD, 5, 5)
    else:
        mesh = create_unit_cube(MPI.COMM_WORLD, 2, 2, 2)
    V = functionspace(mesh, ("DG", order, (tdim,)))
    V1 = functionspace(mesh, ("N1curl", order + 1))
    u, v = Function(V), Function(V1)
    u.interpolate(lambda x: x[:tdim] ** order)
    v.interpolate(u)
    assert assemble_scalar(form(ufl.inner(u - v, u - v) * ufl.dx)) == pytest.approx(0.0, abs=1.0e-8)


@pytest.mark.parametrize("tdim", [2, 3])
@pytest.mark.parametrize("order", [1, 2, 3])
def test_interpolation_n1curl_to_dg(tdim, order):
    if tdim == 2:
        mesh = create_unit_square(MPI.COMM_WORLD, 5, 5)
    else:
        mesh = create_unit_cube(MPI.COMM_WORLD, 2, 2, 2)
    V = functionspace(mesh, ("N1curl", order + 1))
    V1 = functionspace(mesh, ("DG", order, (tdim,)))
    u, v = Function(V), Function(V1)
    u.interpolate(lambda x: x[:tdim] ** order)
    v.interpolate(u)
    assert assemble_scalar(form(ufl.inner(u - v, u - v) * ufl.dx)) == pytest.approx(0.0, abs=1e-10)


@pytest.mark.parametrize("tdim", [2, 3])
@pytest.mark.parametrize("order", [1, 2, 3])
def test_interpolation_n2curl_to_bdm(tdim, order):
    if tdim == 2:
        mesh = create_unit_square(MPI.COMM_WORLD, 5, 5)
    else:
        mesh = create_unit_cube(MPI.COMM_WORLD, 2, 2, 2)
    V = functionspace(mesh, ("N2curl", order))
    V1 = functionspace(mesh, ("BDM", order))
    u, v = Function(V), Function(V1)
    u.interpolate(lambda x: x[:tdim] ** order)
    v.interpolate(u)
    assert assemble_scalar(form(ufl.inner(u - v, u - v) * ufl.dx)) == pytest.approx(
        0.0, abs=1.0e-10
    )


@pytest.mark.parametrize("order1", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("order2", [1, 2, 3])
def test_interpolation_p2p(order1, order2):
    mesh = create_unit_cube(MPI.COMM_WORLD, 2, 2, 2)
    V = functionspace(mesh, ("Lagrange", order1))
    V1 = functionspace(mesh, ("Lagrange", order2))
    u, v = Function(V), Function(V1)
    u.interpolate(lambda x: x[0])
    v.interpolate(u)
    assert assemble_scalar(form(ufl.inner(u - v, u - v) * ufl.dx)) == pytest.approx(0.0, abs=1e-10)

    DG = functionspace(mesh, ("DG", order2))
    w = Function(DG)
    w.interpolate(u)
    assert assemble_scalar(form(ufl.inner(u - w, u - w) * ufl.dx)) == pytest.approx(0.0, abs=1e-10)


@pytest.mark.parametrize("order1", [1, 2, 3])
@pytest.mark.parametrize("order2", [1, 2])
def test_interpolation_vector_elements(order1, order2):
    mesh = create_unit_cube(MPI.COMM_WORLD, 2, 2, 2)
    gdim = mesh.geometry.dim
    V = functionspace(mesh, ("Lagrange", order1, (gdim,)))
    V1 = functionspace(mesh, ("Lagrange", order2, (gdim,)))
    u, v = Function(V), Function(V1)
    u.interpolate(lambda x: x)
    v.interpolate(u)
    assert assemble_scalar(form(ufl.inner(u - v, u - v) * ufl.dx)) == pytest.approx(0)

    DG = functionspace(mesh, ("DG", order2, (gdim,)))
    w = Function(DG)
    w.interpolate(u)
    assert assemble_scalar(form(ufl.inner(u - w, u - w) * ufl.dx)) == pytest.approx(0)


@pytest.mark.skip_in_parallel
def test_interpolation_non_affine():
    points = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 2, 0],
            [1, 2, 0],
            [0, 0, 3],
            [1, 0, 3],
            [0, 2, 3],
            [1, 2, 3],
            [0.5, 0, 0],
            [0, 1, 0],
            [0, 0, 1.5],
            [1, 1, 0],
            [1, 0, 1.5],
            [0.5, 2, 0],
            [0, 2, 1.5],
            [1, 2, 1.5],
            [0.5, 0, 3],
            [0, 1, 3],
            [1, 1, 3],
            [0.5, 2, 3],
            [0.5, 1, 0],
            [0.5, 0, 1.5],
            [0, 1, 1.5],
            [1, 1, 1.5],
            [0.5, 2, 1.5],
            [0.5, 1, 3],
            [0.5, 1, 1.5],
        ],
        dtype=default_real_type,
    )
    cells = np.array([range(len(points))], dtype=np.int32)
    domain = ufl.Mesh(element("Lagrange", "hexahedron", 2, shape=(3,), dtype=default_real_type))
    mesh = create_mesh(MPI.COMM_WORLD, cells, points, domain)
    W = functionspace(mesh, ("NCE", 1))
    V = functionspace(mesh, ("NCE", 2))
    w, v = Function(W), Function(V)
    w.interpolate(lambda x: x)
    v.interpolate(w)
    assert assemble_scalar(form(ufl.inner(w - v, w - v) * ufl.dx)) == pytest.approx(0, abs=1e-10)


@pytest.mark.skip_in_parallel
def test_interpolation_non_affine_nonmatching_maps():
    points = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 2, 0],
            [1, 2, 0],
            [0, 0, 3],
            [1, 0, 3],
            [0, 2, 3],
            [1, 2, 3],
            [0.5, 0, 0],
            [0, 1, 0],
            [0, 0, 1.5],
            [1, 1, 0],
            [1, 0, 1.5],
            [0.5, 2, 0],
            [0, 2, 1.5],
            [1, 2, 1.5],
            [0.5, 0, 3],
            [0, 1, 3],
            [1, 1, 3],
            [0.5, 2, 3],
            [0.5, 1, 0],
            [0.5, -0.1, 1.5],
            [0, 1, 1.5],
            [1, 1, 1.5],
            [0.5, 2, 1.5],
            [0.5, 1, 3],
            [0.5, 1, 1.5],
        ],
        dtype=default_real_type,
    )
    cells = np.array([range(len(points))], dtype=np.int32)
    domain = ufl.Mesh(element("Lagrange", "hexahedron", 2, shape=(3,), dtype=default_real_type))
    mesh = create_mesh(MPI.COMM_WORLD, cells, points, domain)
    gdim = mesh.geometry.dim
    W = functionspace(mesh, ("DG", 1, (gdim,)))
    V = functionspace(mesh, ("NCE", 4))
    w, v = Function(W), Function(V)
    w.interpolate(lambda x: x)
    v.interpolate(w)
    assert assemble_scalar(form(ufl.inner(w - v, w - v) * ufl.dx)) == pytest.approx(0, abs=1e-8)


@pytest.mark.parametrize("order", [2, 3, 4])
@pytest.mark.parametrize("dim", [2, 3])
def test_nedelec_spatial(order, dim):
    if dim == 2:
        mesh = create_unit_square(MPI.COMM_WORLD, 4, 4)
    elif dim == 3:
        mesh = create_unit_cube(MPI.COMM_WORLD, 2, 2, 2)

    V = functionspace(mesh, ("N1curl", order))
    u = Function(V)
    x = ufl.SpatialCoordinate(mesh)

    # The expression (x,y,z) is contained in the N1curl function space
    # order>1
    f_ex = x
    f = Expression(f_ex, V.element.interpolation_points())
    u.interpolate(f)
    assert np.abs(assemble_scalar(form(ufl.inner(u - f_ex, u - f_ex) * ufl.dx))) == pytest.approx(
        0, abs=1e-10
    )

    # The target expression is also contained in N2curl space of any
    # order
    V2 = functionspace(mesh, ("N2curl", 1))
    w = Function(V2)
    f2 = Expression(f_ex, V2.element.interpolation_points())
    w.interpolate(f2)
    assert np.abs(assemble_scalar(form(ufl.inner(w - f_ex, w - f_ex) * ufl.dx))) == pytest.approx(0)


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

    V = functionspace(mesh, ("Lagrange", order, (dim,)))
    u = Function(V)
    x = ufl.SpatialCoordinate(mesh)

    # The expression (x,y,z)^n is contained in space
    f = ufl.as_vector([x[i] ** order for i in range(dim)])
    u.interpolate(Expression(f, V.element.interpolation_points()))
    assert np.abs(assemble_scalar(form(ufl.inner(u - f, u - f) * ufl.dx))) == pytest.approx(0)


@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_2D_lagrange_to_curl(order):
    mesh = create_unit_square(MPI.COMM_WORLD, 3, 4)
    V, W = functionspace(mesh, ("N1curl", order)), functionspace(mesh, ("Lagrange", order))
    u, u0 = Function(V), Function(W)
    u0.interpolate(lambda x: -x[1])
    u1 = Function(W)
    u1.interpolate(lambda x: x[0])
    f = ufl.as_vector((u0, u1))
    f_expr = Expression(f, V.element.interpolation_points())
    u.interpolate(f_expr)
    x = ufl.SpatialCoordinate(mesh)
    f_ex = ufl.as_vector((-x[1], x[0]))
    assert np.abs(assemble_scalar(form(ufl.inner(u - f_ex, u - f_ex) * ufl.dx))) == pytest.approx(0)


@pytest.mark.parametrize("order", [2, 3, 4])
def test_de_rahm_2D(order):
    mesh = create_unit_square(MPI.COMM_WORLD, 3, 4)
    W = functionspace(mesh, ("Lagrange", order))
    w = Function(W)
    w.interpolate(lambda x: x[0] + x[0] * x[1] + 2 * x[1] ** 2)
    g = ufl.grad(w)
    Q = functionspace(mesh, ("N2curl", order - 1))
    q = Function(Q)
    q.interpolate(Expression(g, Q.element.interpolation_points()))
    x = ufl.SpatialCoordinate(mesh)
    g_ex = ufl.as_vector((1 + x[1], 4 * x[1] + x[0]))
    assert np.abs(assemble_scalar(form(ufl.inner(q - g_ex, q - g_ex) * ufl.dx))) == pytest.approx(
        0, abs=np.sqrt(np.finfo(mesh.geometry.x.dtype).eps)
    )

    V = functionspace(mesh, ("BDM", order - 1))
    v = Function(V)

    def curl2D(u):
        return ufl.as_vector((ufl.Dx(u[1], 0), -ufl.Dx(u[0], 1)))

    v.interpolate(Expression(curl2D(ufl.grad(w)), V.element.interpolation_points()))
    h_ex = ufl.as_vector((1, -1))
    assert np.abs(assemble_scalar(form(ufl.inner(v - h_ex, v - h_ex) * ufl.dx))) == pytest.approx(
        0, abs=np.sqrt(np.finfo(mesh.geometry.x.dtype).eps)
    )


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

    V = functionspace(mesh, ("DG", order))
    u = Function(V)

    cells = locate_entities(mesh, mesh.topology.dim, lambda x: x[1] <= 0.5 + 1e-10)
    num_local_cells = mesh.topology.index_map(mesh.topology.dim).size_local
    cells_local = cells[cells < num_local_cells]
    x = ufl.SpatialCoordinate(mesh)
    f = x[1] ** order
    if not callable_:
        expr = Expression(f, V.element.interpolation_points())
        u.interpolate(expr, cells_local)
    else:
        u.interpolate(lambda x: x[1] ** order, cells_local)
    mt = meshtags(mesh, mesh.topology.dim, cells_local, np.ones(cells_local.size, dtype=np.int32))
    dx = ufl.Measure("dx", domain=mesh, subdomain_data=mt)
    assert np.abs(form(assemble_scalar(form(ufl.inner(u - f, u - f) * dx(1))))) == pytest.approx(0)
    integral = mesh.comm.allreduce(assemble_scalar(form(u * dx)), op=MPI.SUM)
    assert integral == pytest.approx(1 / (order + 1) * 0.5 ** (order + 1), abs=1.0e-6)


def test_interpolate_callable():
    """Test interpolation with callables"""
    numba = pytest.importorskip("numba")
    mesh = create_unit_square(MPI.COMM_WORLD, 2, 1)
    V = functionspace(mesh, ("Lagrange", 2))
    u0, u1 = Function(V), Function(V)

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
    V = functionspace(mesh, ("DG", 2))
    u0, u1 = Function(V), Function(V)
    x = ufl.SpatialCoordinate(mesh)
    f = x[0]
    expr = Expression(f, V.element.interpolation_points())
    u0.interpolate(lambda x: x[0], cells_local)
    u1.interpolate(expr, cells_local)
    assert np.allclose(u0.x.array, u1.x.array, rtol=1.0e-6, atol=1.0e-6)


@pytest.mark.parametrize(
    "scalar_element",
    [
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
    ],
)
def test_vector_element_interpolation(scalar_element):
    """Test interpolation into a range of vector elements."""
    mesh = create_unit_square(
        MPI.COMM_WORLD, 10, 10, getattr(CellType, scalar_element.cell.cellname())
    )
    V = functionspace(mesh, blocked_element(scalar_element, shape=(2,)))
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
    x[0].append(np.array([[0.0, 0.0]]))
    x[0].append(np.array([[1.0, 0.0]]))
    x[0].append(np.array([[0.0, 1.0]]))
    for _ in range(3):
        x[1].append(np.zeros((0, 2)))
    x[2].append(np.zeros((0, 2)))
    M = [[], [], [], []]
    for _ in range(3):
        M[0].append(np.array([[[[1.0]], [[0.0]]], [[[0.0]], [[1.0]]]]))
    for _ in range(3):
        M[1].append(np.zeros((0, 2, 0, 1)))
    M[2].append(np.zeros((0, 2, 0, 1)))
    e = custom_element(
        basix.CellType.triangle,
        [2],
        wcoeffs,
        x,
        M,
        0,
        basix.MapType.identity,
        basix.SobolevSpace.H1,
        False,
        1,
        1,
    )

    V = functionspace(mesh, e)
    gdim = mesh.geometry.dim
    W = functionspace(mesh, ("Lagrange", 1, (gdim,)))
    v = Function(V)
    w = Function(W)
    v.interpolate(lambda x: (x[0], x[1]))
    w.interpolate(lambda x: (x[0], x[1]))
    assert np.abs(assemble_scalar(form(ufl.inner(v - w, v - w) * ufl.dx))) == pytest.approx(0)


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
    vlag_el = element("Lagrange", mesh.basix_cell(), 1, shape=(mesh.geometry.dim,))
    lagr_el = element("Lagrange", mesh.basix_cell(), order)

    V = functionspace(mesh, mixed_element([curl_el, lagr_el]))
    Eb_m = Function(V)
    Eb_m.sub(1).interpolate(g)
    diff = Eb_m[2].dx(1) - dgdy
    error = assemble_scalar(form(ufl.dot(diff, diff) * ufl.dx))

    V = functionspace(mesh, mixed_element([vlag_el, lagr_el]))
    Eb_m = Function(V)
    Eb_m.sub(1).interpolate(g)
    diff = Eb_m[2].dx(1) - dgdy
    assert assemble_scalar(form(ufl.dot(diff, diff) * ufl.dx)) == pytest.approx(error)


@pytest.mark.parametrize("xtype", [np.float64])
@pytest.mark.parametrize("cell_type0", [CellType.hexahedron, CellType.tetrahedron])
@pytest.mark.parametrize("cell_type1", [CellType.triangle, CellType.quadrilateral])
def test_nonmatching_mesh_interpolation(xtype, cell_type0, cell_type1):
    mesh0 = create_unit_cube(MPI.COMM_WORLD, 5, 6, 7, cell_type=cell_type0, dtype=xtype)
    mesh1 = create_unit_square(MPI.COMM_WORLD, 5, 4, cell_type=cell_type1, dtype=xtype)

    def f(x):
        return (7 * x[1], 3 * x[0], x[2] + 0.4)

    el0 = element("Lagrange", mesh0.basix_cell(), 1, shape=(3,), dtype=xtype)
    V0 = functionspace(mesh0, el0)
    el1 = element("Lagrange", mesh1.basix_cell(), 1, shape=(3,), dtype=xtype)
    V1 = functionspace(mesh1, el1)

    # Interpolate on 3D mesh
    u0 = Function(V0, dtype=xtype)
    u0.interpolate(f)
    u0.x.scatter_forward()
    padding = 1e-14

    # Check that both interfaces of create nonmatching meshes interpolation data returns the same
    fine_mesh_cell_map = mesh1.topology.index_map(mesh1.topology.dim)
    num_cells_on_proc = fine_mesh_cell_map.size_local + fine_mesh_cell_map.num_ghosts
    cells = np.arange(num_cells_on_proc, dtype=np.int32)
    interpolation_data = create_nonmatching_meshes_interpolation_data(
        V1.mesh.geometry, V1.element, V0.mesh, cells, padding=padding
    )
    other_interpolation_data = create_nonmatching_meshes_interpolation_data(
        V1.mesh,
        V1.element,
        V0.mesh,
        padding=padding,
    )
    for data_0, data_1 in zip(interpolation_data, other_interpolation_data):
        np.testing.assert_allclose(data_0, data_1)

    # Interpolate 3D->2D
    u1 = Function(V1, dtype=xtype)

    u1.interpolate(u0, nmm_interpolation_data=interpolation_data)
    u1.x.scatter_forward()

    # Exact interpolation on 2D mesh
    u1_ex = Function(V1, dtype=xtype)
    u1_ex.interpolate(f)
    u1_ex.x.scatter_forward()

    assert np.allclose(
        u1_ex.x.array,
        u1.x.array,
        rtol=np.sqrt(np.finfo(xtype).eps),
        atol=np.sqrt(np.finfo(xtype).eps),
    )

    # Interpolate 2D->3D
    u0_2 = Function(V0, dtype=xtype)
    u0_2.interpolate(
        u1,
        nmm_interpolation_data=create_nonmatching_meshes_interpolation_data(
            u0_2.function_space.mesh,
            u0_2.function_space.element,
            u1.function_space.mesh,
            padding=padding,
        ),
    )

    # Check that function values over facets of 3D mesh of the twice
    # interpolated property is preserved
    def locate_bottom_facets(x):
        return np.isclose(x[2], 0)

    facets = locate_entities_boundary(mesh0, mesh0.topology.dim - 1, locate_bottom_facets)
    facet_tag = meshtags(
        mesh0, mesh0.topology.dim - 1, facets, np.full(len(facets), 1, dtype=np.int32)
    )
    residual = ufl.inner(u0 - u0_2, u0 - u0_2) * ufl.ds(
        domain=mesh0, subdomain_data=facet_tag, subdomain_id=1
    )
    assert np.isclose(assemble_scalar(form(residual, dtype=xtype)), 0)


@pytest.mark.parametrize("xtype", [np.float64])
def test_nonmatching_mesh_single_cell_overlap_interpolation(xtype):
    # mesh2 is contained by a single cell of mesh1. Here we test
    # interpolation when *not* every process has data to communicate

    # Test interpolation from mesh1 to mesh2
    n_mesh1 = 2
    mesh1 = create_rectangle(
        MPI.COMM_WORLD,
        [[0.0, 0.0], [1.0, 1.0]],
        [n_mesh1, n_mesh1],
        cell_type=CellType.quadrilateral,
        dtype=xtype,
    )

    n_mesh2 = 2
    p0_mesh2 = 1.0 / n_mesh1
    mesh2 = create_rectangle(
        MPI.COMM_WORLD,
        [[0.0, 0.0], [p0_mesh2, p0_mesh2]],
        [n_mesh2, n_mesh2],
        cell_type=CellType.triangle,
        dtype=xtype,
    )

    u1 = Function(functionspace(mesh1, ("Lagrange", 1)), name="u1", dtype=xtype)
    u2 = Function(functionspace(mesh2, ("Lagrange", 1)), name="u2", dtype=xtype)

    def f_test1(x):
        return 1.0 - x[0] * x[1]

    u1.interpolate(f_test1)
    u1.x.scatter_forward()
    padding = 1e-14
    u1_2_u2_nmm_data = create_nonmatching_meshes_interpolation_data(
        u2.function_space.mesh, u2.function_space.element, u1.function_space.mesh, padding=padding
    )

    u2.interpolate(u1, nmm_interpolation_data=u1_2_u2_nmm_data)
    u2.x.scatter_forward()

    # interpolate f which is exactly represented on the element
    u2_exact = Function(u2.function_space, dtype=xtype)
    u2_exact.interpolate(f_test1)
    u2_exact.x.scatter_forward()

    l2_error = assemble_scalar(form((u2 - u2_exact) ** 2 * ufl.dx, dtype=xtype))
    assert np.isclose(l2_error, 0.0, rtol=np.finfo(xtype).eps, atol=np.finfo(xtype).eps)

    # Test interpolation from mesh2 to mesh1
    def f_test2(x):
        return x[0] * x[1]

    u1.x.array[:] = 0.0
    u1.x.scatter_forward()

    u2.interpolate(f_test2)
    u2.x.scatter_forward()
    padding = 1e-14
    u2_2_u1_nmm_data = create_nonmatching_meshes_interpolation_data(
        u1.function_space.mesh, u1.function_space.element, u2.function_space.mesh, padding=padding
    )

    u1.interpolate(u2, nmm_interpolation_data=u2_2_u1_nmm_data)
    u1.x.scatter_forward()

    u1_exact = Function(u1.function_space, dtype=xtype)
    u1_exact.interpolate(f_test2)
    u1_exact.x.scatter_forward()

    # Find the single cell in mesh1 which is overlapped by mesh2
    tree1 = bb_tree(mesh1, mesh1.topology.dim)
    cells_overlapped1 = compute_collisions_points(
        tree1, np.array([p0_mesh2, p0_mesh2, 0.0]) / 2
    ).array
    assert cells_overlapped1.shape[0] <= 1

    # Construct the error measure on the overlapped cell
    cell_label = 1
    cts = meshtags(
        mesh1, mesh1.topology.dim, cells_overlapped1, np.full_like(cells_overlapped1, cell_label)
    )
    dx_cell = ufl.Measure("dx", subdomain_data=cts)

    l2_error = assemble_scalar(form((u1 - u1_exact) ** 2 * dx_cell(cell_label), dtype=xtype))
    assert np.isclose(l2_error, 0.0, rtol=np.finfo(xtype).eps, atol=np.finfo(xtype).eps)


@pytest.mark.parametrize("dtype", [np.float64])
def test_symmetric_tensor_interpolation(dtype):
    mesh = create_unit_square(MPI.COMM_WORLD, 10, 10, dtype=dtype)

    def tensor(x):
        mat = np.array(
            [
                [0],
                [1],
                [2],
                [3],
                [4],
                [5],
                [1],
                [6],
                [7],
                [8],
                [9],
                [10],
                [2],
                [7],
                [11],
                [12],
                [13],
                [14],
                [3],
                [8],
                [12],
                [15],
                [16],
                [17],
                [4],
                [9],
                [13],
                [16],
                [18],
                [19],
                [5],
                [10],
                [14],
                [17],
                [19],
                [20],
            ]
        )
        return np.broadcast_to(mat, (36, x.shape[1]))

    element = basix.ufl.element("DG", mesh.basix_cell(), 0, shape=(6, 6), dtype=dtype)
    symm_element = basix.ufl.element(
        "DG", mesh.basix_cell(), 0, shape=(6, 6), symmetry=True, dtype=dtype
    )
    space = functionspace(mesh, element)
    symm_space = functionspace(mesh, symm_element)
    f = Function(space)
    symm_f = Function(symm_space)

    f.interpolate(lambda x: tensor(x))
    symm_f.interpolate(lambda x: tensor(x))

    l2_error = assemble_scalar(form((f - symm_f) ** 2 * ufl.dx, dtype=dtype))
    assert np.isclose(l2_error, 0.0, rtol=np.finfo(dtype).eps, atol=np.finfo(dtype).eps)


def test_submesh_interpolation():
    """Test interpolation of a function between a submesh and its parent"""
    mesh = create_unit_square(MPI.COMM_WORLD, 6, 7)

    def left_locator(x):
        return x[0] <= 0.5 + 1e-14

    def ref_func(x):
        return x[0] + x[1] ** 2

    tdim = mesh.topology.dim
    cells = locate_entities(mesh, tdim, left_locator)
    submesh, sub_to_parent, _, _ = create_submesh(mesh, tdim, cells)

    V = functionspace(mesh, ("Lagrange", 2))
    u = Function(V)
    u.interpolate(ref_func)

    V_sub = functionspace(submesh, ("DG", 3))
    u_sub = Function(V_sub)

    # Map from parent to sub mesh
    u_sub.interpolate(u, cell_map=sub_to_parent)

    u_sub_exact = Function(V_sub)
    u_sub_exact.interpolate(ref_func)
    atol = 5 * np.finfo(default_scalar_type).resolution
    np.testing.assert_allclose(u_sub_exact.x.array, u_sub.x.array, atol=atol)

    # Map from sub to parent
    W = functionspace(mesh, ("DG", 4))
    w = Function(W)

    cell_imap = mesh.topology.index_map(tdim)
    num_cells = cell_imap.size_local + cell_imap.num_ghosts
    parent_to_sub = np.full(num_cells, -1, dtype=np.int32)
    parent_to_sub[sub_to_parent] = np.arange(len(sub_to_parent))

    # Mapping back needs to be restricted to the subset of cells in the submesh
    w.interpolate(u_sub_exact, cells=sub_to_parent, cell_map=parent_to_sub)

    w_exact = Function(W)
    w_exact.interpolate(ref_func, cells=cells)

    np.testing.assert_allclose(w.x.array, w_exact.x.array, atol=atol)


def test_submesh_expression_interpolation():
    """Test interpolation of an expression between a submesh and its parent"""
    mesh = create_unit_square(MPI.COMM_WORLD, 10, 8, cell_type=CellType.quadrilateral)

    def left_locator(x):
        return x[0] <= 0.5 + 1e-14

    def ref_func(x):
        return -3 * x[0] ** 2 + x[1] ** 2

    def grad_ref_func(x):
        values = np.zeros((2, x.shape[1]), dtype=default_scalar_type)
        values[0] = -6 * x[0]
        values[1] = 2 * x[1]
        return values

    def modified_grad(x):
        grad = grad_ref_func(x)
        return -0.2 * grad[1], 0.1 * grad[0]

    tdim = mesh.topology.dim
    cells = locate_entities(mesh, tdim, left_locator)
    submesh, sub_to_parent, _, _ = create_submesh(mesh, tdim, cells)

    V = functionspace(mesh, ("Lagrange", 2))
    u = Function(V)
    u.interpolate(ref_func)

    V_sub = functionspace(submesh, ("N2curl", 1))
    u_sub = Function(V_sub)

    parent_expr = Expression(ufl.grad(u), V_sub.element.interpolation_points())

    # Map from parent to sub mesh

    u_sub.interpolate(parent_expr, expr_mesh=mesh, cell_map=sub_to_parent)

    u_sub_exact = Function(V_sub)
    u_sub_exact.interpolate(grad_ref_func)
    atol = 10 * np.finfo(default_scalar_type).resolution
    np.testing.assert_allclose(u_sub_exact.x.array, u_sub.x.array, atol=atol)

    # Map from sub to parent
    W = functionspace(mesh, ("DQ", 1, (mesh.geometry.dim,)))
    w = Function(W)

    cell_imap = mesh.topology.index_map(tdim)
    num_cells = cell_imap.size_local + cell_imap.num_ghosts
    parent_to_sub = np.full(num_cells, -1, dtype=np.int32)
    parent_to_sub[sub_to_parent] = np.arange(len(sub_to_parent))

    # Map exact solution (based on quadrature points) back to parent mesh
    sub_vec = ufl.as_vector((-0.2 * u_sub_exact[1], 0.1 * u_sub_exact[0]))
    sub_expr = Expression(sub_vec, W.element.interpolation_points())

    # Mapping back needs to be restricted to the subset of cells in the submesh
    w.interpolate(sub_expr, cells=sub_to_parent, expr_mesh=submesh, cell_map=parent_to_sub)

    w_exact = Function(W)
    w_exact.interpolate(modified_grad, cells=cells)
    w_exact.x.scatter_forward()
    np.testing.assert_allclose(w.x.array, w_exact.x.array, atol=atol)
