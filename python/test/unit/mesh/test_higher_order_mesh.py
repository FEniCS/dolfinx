# Copyright (C) 2019 Jørgen Schartum Dokken & Matthew Scroggs
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
""" Unit-tests for higher order meshes """

import numpy as np
import pytest
import sympy as sp
import scipy.integrate
from sympy.vector import CoordSys3D, matrix_to_vector

from dolfin import MPI, Function, FunctionSpace, Mesh, fem
from dolfin.cpp.mesh import CellType, GhostMode
from dolfin.fem import assemble_scalar
from dolfin_utils.test.skips import skip_in_parallel
from ufl import dx


def sympy_scipy(points, nodes, L, H):
    """
    Approximated integration of z + x*y over a surface where the z-coordinate
    is only dependent of the y-component of the box.
    x in [0,L], y in [0,H]
    Input:
      points: All points of defining the geometry
      nodes:  Points on one of the outer boundaries varying in the y-direction
    """
    degree = len(nodes) - 1

    x, y, z = sp.symbols("x y z")
    a = [sp.Symbol("a{0:d}".format(i)) for i in range(degree + 1)]

    # Find polynomial for variation in z-direction
    poly = 0
    for deg in range(degree + 1):
        poly += a[deg] * y**deg
    eqs = []
    for node in nodes:
        eqs.append(poly.subs(y, points[node][-2]) - points[node][-1])
    coeffs = sp.solve(eqs, a)
    transform = poly
    for i in range(len(a)):
        transform = transform.subs(a[i], coeffs[a[i]])

    # Compute integral
    C = CoordSys3D("C")
    para = sp.Matrix([x, y, transform])
    vec = matrix_to_vector(para, C)
    cross = (vec.diff(x) ^ vec.diff(y)).magnitude()

    expr = (transform + x * y) * cross
    approx = sp.lambdify((x, y), expr)

    ref = scipy.integrate.nquad(approx, [[0, L], [0, H]])[0]
    # Slow and only works for simple integrals
    # integral = sp.integrate(expr, (y, 0, H))
    # integral = sp.integrate(integral, (x, 0, L))
    # ex = integral.evalf()

    return ref


@skip_in_parallel
@pytest.mark.parametrize('H', [0.3, 2])
@pytest.mark.parametrize('Z', [0.8, 1])
def test_second_order_mesh(H, Z):
    # Test second order mesh by computing volume of two cells
    #  *-----*-----*   3----6-----2
    #  | \         |   | \        |
    #  |   \       |   |   \      |
    #  *     *     *   7     8    5
    #  |       \   |   |      \   |
    #  |         \ |   |        \ |
    #  *-----*-----*   0----4-----1

    # Perturbation of nodes 4,5,6,7 while keeping volume constant
    L = 1
    points = np.array([[0, 0, 0], [L, 0, 0], [L, H, Z], [0, H, Z],
                       [L / 2, 0, 0], [L, H / 2, 0], [L / 2, H, Z],
                       [0, H / 2, 0], [L / 2, H / 2, 0]])
    cells = np.array([[0, 1, 3, 4, 8, 7],
                      [1, 2, 3, 5, 6, 8]])
    mesh = Mesh(MPI.comm_world, CellType.triangle, points, cells,
                [], GhostMode.none)

    def e2(x):
        values = np.empty((x.shape[0], 1))
        values[:, 0] = x[:, 2] + x[:, 0] * x[:, 1]
        return values
    degree = mesh.degree()
    # Interpolate function
    V = FunctionSpace(mesh, ("CG", degree))
    u = Function(V)
    cmap = fem.create_coordinate_map(mesh.ufl_domain())
    mesh.geometry.coord_mapping = cmap
    u.interpolate(e2)
    intu = assemble_scalar(u * dx(metadata={"quadrature_degree": 40}))
    nodes = [0, 7, 3]
    ref = sympy_scipy(points, nodes, L, H)
    assert ref == pytest.approx(intu, rel=1e-6)


@skip_in_parallel
@pytest.mark.parametrize('H', [1, 0.3])
@pytest.mark.parametrize('Z', [1, 0.5])
def test_third_order_mesh(H, Z):
    #  *---*---*---*   3--11--10--2
    #  | \         |   | \        |
    #  *   *   *   *   8   7  15  13
    #  |     \     |   |    \     |
    #  *  *    *   *   9  14  6   12
    #  |         \ |   |        \ |
    #  *---*---*---*   0--4---5---1
    L = 1
    points = np.array([[0, 0, 0], [L, 0, 0], [L, H, Z], [0, H, Z],  # 0, 1, 2, 3
                       [L / 3, 0, 0], [2 * L / 3, 0, 0],            # 4, 5
                       [2 * L / 3, H / 3, 0], [L / 3, 2 * H / 3, 0],  # 6, 7
                       [0, 2 * H / 3, 0], [0, H / 3, 0],        # 8, 9
                       [2 * L / 3, H, Z], [L / 3, H, Z],              # 10, 11
                       [L, H / 3, 0], [L, 2 * H / 3, 0],  # 12, 13
                       [L / 3, H / 3, 0],                         # 14
                       [2 * L / 3, 2 * H / 3, 0]])            # 15
    cells = np.array([[0, 1, 3, 4, 5, 6, 7, 8, 9, 14],
                      [1, 2, 3, 12, 13, 10, 11, 7, 6, 15]])
    mesh = Mesh(MPI.comm_world, CellType.triangle, points, cells,
                [], GhostMode.none)

    def e2(x):
        values = np.empty((x.shape[0], 1))
        values[:, 0] = x[:, 2] + x[:, 0] * x[:, 1]
        return values
    degree = mesh.degree()
    # Interpolate function
    V = FunctionSpace(mesh, ("CG", degree))
    u = Function(V)
    cmap = fem.create_coordinate_map(mesh.ufl_domain())
    mesh.geometry.coord_mapping = cmap
    u.interpolate(e2)
    intu = assemble_scalar(u * dx(metadata={"quadrature_degree": 40}))
    nodes = [0, 9, 8, 3]
    ref = sympy_scipy(points, nodes, L, H)
    assert ref == pytest.approx(intu, rel=1e-6)


@skip_in_parallel
@pytest.mark.parametrize('H', [1, 0.3])
@pytest.mark.parametrize('Z', [0.2, 0.1])
def test_fourth_order_mesh(H, Z):
    L = 1
    #  *--*--*--*--*   3-21-20-19--2
    #  | \         |   | \         |
    #  *   *  * *  *   10 9 24 23  18
    #  |     \     |   |    \      |
    #  *  *   *  * *   11 15  8 22 17
    #  |       \   |   |       \   |
    #  *  * *   *  *   12 13 14 7  16
    #  |         \ |   |         \ |
    #  *--*--*--*--*   0--4--5--6--1
    points = np.array(
        [[0, 0, 0], [L, 0, 0], [L, H, Z], [0, H, Z],   # 0, 1, 2, 3
         [L / 4, 0, 0], [L / 2, 0, 0], [3 * L / 4, 0, 0],  # 4, 5, 6
         [3 / 4 * L, H / 4, Z / 2], [L / 2, H / 2, 0],         # 7, 8
         [L / 4, 3 * H / 4, 0], [0, 3 * H / 4, 0],         # 9, 10
         [0, H / 2, 0], [0, H / 4, Z / 2],                     # 11, 12
         [L / 4, H / 4, Z / 2], [L / 2, H / 4, Z / 2], [L / 4, H / 2, 0],  # 13, 14, 15
         [L, H / 4, Z / 2], [L, H / 2, 0], [L, 3 * H / 4, 0],          # 16, 17, 18
         [3 * L / 4, H, Z], [L / 2, H, Z], [L / 4, H, Z],          # 19, 20, 21
         [3 * L / 4, H / 2, 0], [3 * L / 4, 3 * H / 4, 0],         # 22, 23
         [L / 2, 3 * H / 4, 0]]                                    # 24
    )

    cells = np.array([[0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                      [1, 2, 3, 16, 17, 18, 19, 20, 21, 9, 8, 7, 22, 23, 24]])
    mesh = Mesh(MPI.comm_world, CellType.triangle, points, cells,
                [], GhostMode.none)

    def e2(x):
        values = np.empty((x.shape[0], 1))
        values[:, 0] = x[:, 2] + x[:, 0] * x[:, 1]
        return values
    degree = mesh.degree()
    # Interpolate function
    V = FunctionSpace(mesh, ("CG", degree))
    u = Function(V)
    cmap = fem.create_coordinate_map(mesh.ufl_domain())
    mesh.geometry.coord_mapping = cmap
    u.interpolate(e2)
    intu = assemble_scalar(u * dx(metadata={"quadrature_degree": 50}))
    nodes = [0, 3, 10, 11, 12]
    ref = sympy_scipy(points, nodes, L, H)
    assert ref == pytest.approx(intu, rel=1e-4)


def scipy_one_cell(points, nodes):
    degree = len(nodes) - 1

    x, y, z = sp.symbols("x y z")
    a = [sp.Symbol("a{0:d}".format(i)) for i in range(degree + 1)]

    # Find polynomial for variation in z-direction
    poly = 0
    for deg in range(degree + 1):
        poly += a[deg] * y**deg
    eqs = []
    for node in nodes:
        eqs.append(poly.subs(y, points[node][-2]) - points[node][-1])
    coeffs = sp.solve(eqs, a)
    transform = poly

    for i in range(len(a)):
        transform = transform.subs(a[i], coeffs[a[i]])
    # Compute integral
    C = CoordSys3D("C")
    para = sp.Matrix([x, y, transform])
    vec = matrix_to_vector(para, C)
    cross = (vec.diff(x) ^ vec.diff(y)).magnitude()

    expr = (transform + x * y) * cross

    approx = sp.lambdify((x, y), expr)
    ref = scipy.integrate.dblquad(approx, 0, 1, lambda x: 0, lambda x: 1 - x)[0]
    return ref


# FIXME: Higher order tests are too slow, need to find a better test
@skip_in_parallel
@pytest.mark.parametrize("order", range(1, 6))
def test_nth_order_triangle(order):
    num_nodes = (order + 1) * (order + 2) / 2
    cells = np.array([range(int(num_nodes))])

    if order == 1:
        points = np.array([[0.00000, 0.00000, 0.00000], [1.00000, 0.00000, 0.00000],
                           [0.00000, 1.00000, 0.00000]])
    elif order == 2:
        points = np.array([[0.00000, 0.00000, 0.00000], [1.00000, 0.00000, 0.00000],
                           [0.00000, 1.00000, 0.00000], [0.50000, 0.00000, 0.00000],
                           [0.50000, 0.50000, -0.25000], [0.00000, 0.50000, -0.25000]])

    elif order == 3:
        points = np.array([[0.00000, 0.00000, 0.00000], [1.00000, 0.00000, 0.00000],
                           [0.00000, 1.00000, 0.00000], [0.33333, 0.00000, 0.00000],
                           [0.66667, 0.00000, 0.00000], [0.66667, 0.33333, -0.11111],
                           [0.33333, 0.66667, 0.11111], [0.00000, 0.66667, 0.11111],
                           [0.00000, 0.33333, -0.11111], [0.33333, 0.33333, -0.11111]])
    elif order == 4:
        points = np.array([[0.00000, 0.00000, 0.00000], [1.00000, 0.00000, 0.00000],
                           [0.00000, 1.00000, 0.00000], [0.25000, 0.00000, 0.00000],
                           [0.50000, 0.00000, 0.00000], [0.75000, 0.00000, 0.00000],
                           [0.75000, 0.25000, -0.06250], [0.50000, 0.50000, 0.06250],
                           [0.25000, 0.75000, -0.06250], [0.00000, 0.75000, -0.06250],
                           [0.00000, 0.50000, 0.06250], [0.00000, 0.25000, -0.06250],
                           [0.25000, 0.25000, -0.06250], [0.50000, 0.25000, -0.06250],
                           [0.25000, 0.50000, 0.06250]])

    elif order == 5:
        points = np.array([[0.00000, 0.00000, 0.00000], [1.00000, 0.00000, 0.00000],
                           [0.00000, 1.00000, 0.00000], [0.20000, 0.00000, 0.00000],
                           [0.40000, 0.00000, 0.00000], [0.60000, 0.00000, 0.00000],
                           [0.80000, 0.00000, 0.00000], [0.80000, 0.20000, -0.04000],
                           [0.60000, 0.40000, 0.04000], [0.40000, 0.60000, -0.04000],
                           [0.20000, 0.80000, 0.04000], [0.00000, 0.80000, 0.04000],
                           [0.00000, 0.60000, -0.04000], [0.00000, 0.40000, 0.04000],
                           [0.00000, 0.20000, -0.04000], [0.20000, 0.20000, -0.04000],
                           [0.60000, 0.20000, -0.04000], [0.20000, 0.60000, -0.04000],
                           [0.40000, 0.20000, -0.04000], [0.40000, 0.40000, 0.04000],
                           [0.20000, 0.40000, 0.04000]])

    elif order == 6:
        points = np.array([[0.00000, 0.00000, 0.00000], [1.00000, 0.00000, 0.00000],
                           [0.00000, 1.00000, 0.00000], [0.16667, 0.00000, 0.00000],
                           [0.33333, 0.00000, 0.00000], [0.50000, 0.00000, 0.00000],
                           [0.66667, 0.00000, 0.00000], [0.83333, 0.00000, 0.00000],
                           [0.83333, 0.16667, -0.00463], [0.66667, 0.33333, 0.00463],
                           [0.50000, 0.50000, -0.00463], [0.33333, 0.66667, 0.00463],
                           [0.16667, 0.83333, -0.00463], [0.00000, 0.83333, -0.00463],
                           [0.00000, 0.66667, 0.00463], [0.00000, 0.50000, -0.00463],
                           [0.00000, 0.33333, 0.00463], [0.00000, 0.16667, -0.00463],
                           [0.16667, 0.16667, -0.00463], [0.66667, 0.16667, -0.00463],
                           [0.16667, 0.66667, 0.00463], [0.33333, 0.16667, -0.00463],
                           [0.50000, 0.16667, -0.00463], [0.50000, 0.33333, 0.00463],
                           [0.33333, 0.50000, -0.00463], [0.16667, 0.50000, -0.00463],
                           [0.16667, 0.33333, 0.00463], [0.33333, 0.33333, 0.00463]])
    elif order == 7:
        points = np.array([[0.00000, 0.00000, 0.00000], [1.00000, 0.00000, 0.00000],
                           [0.00000, 1.00000, 0.00000], [0.14286, 0.00000, 0.00000],
                           [0.28571, 0.00000, 0.00000], [0.42857, 0.00000, 0.00000],
                           [0.57143, 0.00000, 0.00000], [0.71429, 0.00000, 0.00000],
                           [0.85714, 0.00000, 0.00000], [0.85714, 0.14286, -0.02041],
                           [0.71429, 0.28571, 0.02041], [0.57143, 0.42857, -0.02041],
                           [0.42857, 0.57143, 0.02041], [0.28571, 0.71429, -0.02041],
                           [0.14286, 0.85714, 0.02041], [0.00000, 0.85714, 0.02041],
                           [0.00000, 0.71429, -0.02041], [0.00000, 0.57143, 0.02041],
                           [0.00000, 0.42857, -0.02041], [0.00000, 0.28571, 0.02041],
                           [0.00000, 0.14286, -0.02041], [0.14286, 0.14286, -0.02041],
                           [0.71429, 0.14286, -0.02041], [0.14286, 0.71429, -0.02041],
                           [0.28571, 0.14286, -0.02041], [0.42857, 0.14286, -0.02041],
                           [0.57143, 0.14286, -0.02041], [0.57143, 0.28571, 0.02041],
                           [0.42857, 0.42857, -0.02041], [0.28571, 0.57143, 0.02041],
                           [0.14286, 0.57143, 0.02041], [0.14286, 0.42857, -0.02041],
                           [0.14286, 0.28571, 0.02041], [0.28571, 0.28571, 0.02041],
                           [0.42857, 0.28571, 0.02041], [0.28571, 0.42857, -0.02041]])
    # Higher order tests are too slow
    elif order == 8:
        points = np.array([[0.00000, 0.00000, 0.00000], [1.00000, 0.00000, 0.00000],
                           [0.00000, 1.00000, 0.00000], [0.12500, 0.00000, 0.00000],
                           [0.25000, 0.00000, 0.00000], [0.37500, 0.00000, 0.00000],
                           [0.50000, 0.00000, 0.00000], [0.62500, 0.00000, 0.00000],
                           [0.75000, 0.00000, 0.00000], [0.87500, 0.00000, 0.00000],
                           [0.87500, 0.12500, -0.00195], [0.75000, 0.25000, 0.00195],
                           [0.62500, 0.37500, -0.00195], [0.50000, 0.50000, 0.00195],
                           [0.37500, 0.62500, -0.00195], [0.25000, 0.75000, 0.00195],
                           [0.12500, 0.87500, -0.00195], [0.00000, 0.87500, -0.00195],
                           [0.00000, 0.75000, 0.00195], [0.00000, 0.62500, -0.00195],
                           [0.00000, 0.50000, 0.00195], [0.00000, 0.37500, -0.00195],
                           [0.00000, 0.25000, 0.00195], [0.00000, 0.12500, -0.00195],
                           [0.12500, 0.12500, -0.00195], [0.75000, 0.12500, -0.00195],
                           [0.12500, 0.75000, 0.00195], [0.25000, 0.12500, -0.00195],
                           [0.37500, 0.12500, -0.00195], [0.50000, 0.12500, -0.00195],
                           [0.62500, 0.12500, -0.00195], [0.62500, 0.25000, 0.00195],
                           [0.50000, 0.37500, -0.00195], [0.37500, 0.50000, 0.00195],
                           [0.25000, 0.62500, -0.00195], [0.12500, 0.62500, -0.00195],
                           [0.12500, 0.50000, 0.00195], [0.12500, 0.37500, -0.00195],
                           [0.12500, 0.25000, 0.00195], [0.25000, 0.25000, 0.00195],
                           [0.50000, 0.25000, 0.00195], [0.25000, 0.50000, 0.00195],
                           [0.37500, 0.25000, 0.00195], [0.37500, 0.37500, -0.00195],
                           [0.25000, 0.37500, -0.00195]])

    mesh = Mesh(MPI.comm_world, CellType.triangle, points, cells,
                [], GhostMode.none)

    # Find nodes corresponding to y axis
    nodes = []
    for j in range(points.shape[0]):
        if np.isclose(points[j][0], 0):
            nodes.append(j)

    def e2(x):
        values = np.empty((x.shape[0], 1))
        values[:, 0] = x[:, 2] + x[:, 0] * x[:, 1]
        return values

    # For solution to be in functionspace
    V = FunctionSpace(mesh, ("CG", max(2, order)))
    u = Function(V)
    cmap = fem.create_coordinate_map(mesh.ufl_domain())
    mesh.geometry.coord_mapping = cmap
    u.interpolate(e2)

    quad_order = 30
    intu = assemble_scalar(u * dx(metadata={"quadrature_degree": quad_order}))

    ref = scipy_one_cell(points, nodes)
    assert ref == pytest.approx(intu, rel=3e-3)
