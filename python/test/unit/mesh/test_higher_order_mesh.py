# Copyright (C) 2019 Jørgen Schartum Dokken and Matthew Scroggs
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
""" Unit-tests for higher order meshes """

import os
import random

import numpy as np
import pytest
import scipy.integrate
import sympy as sp
import ufl
from dolfinx import Function, FunctionSpace, cpp
from dolfinx.cpp.io import perm_gmsh, perm_vtk
from dolfinx.cpp.mesh import CellType
from dolfinx.fem import assemble_scalar
from dolfinx.io import XDMFFile, ufl_mesh_from_gmsh
from dolfinx.mesh import create_mesh
from dolfinx_utils.test.skips import skip_in_parallel
from mpi4py import MPI
from sympy.vector import CoordSys3D, matrix_to_vector
from ufl import dx


@skip_in_parallel
@pytest.mark.parametrize('order', [1, 2, 3, 4])
def test_quadrilateral_mesh(order):
    random.seed(13)

    points = []
    points += [[i / order, 0] for i in range(order + 1)]
    for j in range(1, order):
        points += [[i / order + 0.1, j / order] for i in range(order + 1)]

    points += [[j / order, 1] for j in range(order + 1)]

    cell = [0, order, order * (order + 1), order * (order + 2)]
    if order > 1:
        cell += list(range(1, order))
        cell += list(range(order + 1, order * (order + 1), order + 1))
        cell += list(range(2 * order + 1, order * (order + 1), order + 1))
        cell += list(range(order * (order + 1) + 1, order * (order + 2)))

        for i in range(1, order):
            cell += list(range(i * (order + 1) + 1, i * (order + 1) + order))

    domain = ufl.Mesh(ufl.VectorElement(
        "Q", ufl.Cell("quadrilateral", geometric_dimension=2), order))

    point_order = list(range((order + 1) ** 2))
    for repeat in range(5):
        random.shuffle(point_order)
        ordered_points = np.zeros(((order + 1) ** 2, 2))
        for i, j in enumerate(point_order):
            ordered_points[j] = points[i]
        ordered_cell = [point_order[i] for i in cell]

        mesh = create_mesh(MPI.COMM_WORLD, [ordered_cell], ordered_points, domain)
        area = assemble_scalar(1 * dx(mesh))

        assert np.isclose(area, 1)


@skip_in_parallel
@pytest.mark.parametrize('order', [1, 2, 3, 4])
def test_hexahedron_mesh(order):
    random.seed(13)

    points = []
    points += [[i / order, j / order, 0] for j in range(order + 1)
               for i in range(order + 1)]
    for k in range(1, order):
        points += [[i / order, j / order + 0.1, k / order] for j in range(order + 1)
                   for i in range(order + 1)]

    points += [[i / order, j / order, 1] for j in range(order + 1) for i in range(order + 1)]

    def coord_to_vertex(x, y, z):
        return (order + 1) ** 2 * z + (order + 1) * y + x

    cell = [coord_to_vertex(x, y, z) for x, y, z in [
        (0, 0, 0), (order, 0, 0), (0, order, 0), (order, order, 0),
        (0, 0, order), (order, 0, order), (0, order, order), (order, order, order)]]

    if order > 1:
        for i in range(1, order):
            cell.append(coord_to_vertex(i, 0, 0))
        for i in range(1, order):
            cell.append(coord_to_vertex(0, i, 0))
        for i in range(1, order):
            cell.append(coord_to_vertex(0, 0, i))
        for i in range(1, order):
            cell.append(coord_to_vertex(order, i, 0))
        for i in range(1, order):
            cell.append(coord_to_vertex(order, 0, i))
        for i in range(1, order):
            cell.append(coord_to_vertex(i, order, 0))
        for i in range(1, order):
            cell.append(coord_to_vertex(0, order, i))
        for i in range(1, order):
            cell.append(coord_to_vertex(order, order, i))
        for i in range(1, order):
            cell.append(coord_to_vertex(i, 0, order))
        for i in range(1, order):
            cell.append(coord_to_vertex(0, i, order))
        for i in range(1, order):
            cell.append(coord_to_vertex(order, i, order))
        for i in range(1, order):
            cell.append(coord_to_vertex(i, order, order))

        for j in range(1, order):
            for i in range(1, order):
                cell.append(coord_to_vertex(i, j, 0))
        for j in range(1, order):
            for i in range(1, order):
                cell.append(coord_to_vertex(i, 0, j))
        for j in range(1, order):
            for i in range(1, order):
                cell.append(coord_to_vertex(0, i, j))
        for j in range(1, order):
            for i in range(1, order):
                cell.append(coord_to_vertex(order, i, j))
        for j in range(1, order):
            for i in range(1, order):
                cell.append(coord_to_vertex(i, order, j))
        for j in range(1, order):
            for i in range(1, order):
                cell.append(coord_to_vertex(i, j, order))

        for k in range(1, order):
            for j in range(1, order):
                for i in range(1, order):
                    cell.append(coord_to_vertex(i, j, k))

    domain = ufl.Mesh(ufl.VectorElement(
        "Q", ufl.Cell("hexahedron", geometric_dimension=3), order))

    print(cell, len(cell))

    point_order = list(range((order + 1) ** 3))
    for repeat in range(5):
        random.shuffle(point_order)
        ordered_points = np.zeros(((order + 1) ** 3, 3))
        for i, j in enumerate(point_order):
            ordered_points[j] = points[i]
        ordered_cell = [point_order[i] for i in cell]

        mesh = create_mesh(MPI.COMM_WORLD, [ordered_cell], ordered_points, domain)
        volume = assemble_scalar(1 * dx(mesh))

        assert np.isclose(volume, 1)


def sympy_scipy(points, nodes, L, H):
    """Approximated integration of z + x*y over a surface where the z-coordinate
    is only dependent of the y-component of the box. x in [0,L], y in
    [0,H]

    Input: points: All points of defining the geometry nodes:  Points on
      one of the outer boundaries varying in the y-direction
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
@pytest.mark.parametrize("vtk,dolfin,cell_type", [
    ([0, 1, 2, 3, 4, 5], [0, 1, 2, 4, 5, 3], CellType.triangle),
    ([0, 1, 2, 3], [0, 1, 3, 2], CellType.quadrilateral),
    ([0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 3, 2, 4, 5, 7, 6], CellType.hexahedron)
])
def test_map_vtk_to_dolfin(vtk, dolfin, cell_type):
    p = perm_vtk(cell_type, len(vtk))
    cell_p = np.array(vtk)[p]
    assert (cell_p == dolfin).all()

    p = np.argsort(perm_vtk(cell_type, len(vtk)))
    cell_p = np.array(dolfin)[p]
    assert (cell_p == vtk).all()


@skip_in_parallel
def test_second_order_tri():
    # Test second order mesh by computing volume of two cells
    #  *-----*-----*   3----6-----2
    #  | \         |   | \        |
    #  |   \       |   |   \      |
    #  *     *     *   7     8    5
    #  |       \   |   |      \   |
    #  |         \ |   |        \ |
    #  *-----*-----*   0----4-----1
    for H in (1.0, 2.0):
        for Z in (0.0, 0.5):
            L = 1
            points = np.array([[0, 0, 0], [L, 0, 0], [L, H, Z], [0, H, Z],
                               [L / 2, 0, 0], [L, H / 2, 0], [L / 2, H, Z],
                               [0, H / 2, 0], [L / 2, H / 2, 0]])

            cells = np.array([[0, 1, 3, 4, 8, 7],
                              [1, 2, 3, 5, 6, 8]])
            cells = cells[:, perm_vtk(CellType.triangle, cells.shape[1])]

            cell = ufl.Cell("triangle", geometric_dimension=points.shape[1])
            domain = ufl.Mesh(ufl.VectorElement("Lagrange", cell, 2))
            mesh = create_mesh(MPI.COMM_WORLD, cells, points, domain)

            def e2(x):
                return x[2] + x[0] * x[1]
            # Interpolate function
            V = FunctionSpace(mesh, ("CG", 2))
            u = Function(V)
            u.interpolate(e2)

            intu = assemble_scalar(u * dx(mesh, metadata={"quadrature_degree": 20}))
            intu = mesh.mpi_comm().allreduce(intu, op=MPI.SUM)

            nodes = [0, 3, 7]
            ref = sympy_scipy(points, nodes, L, H)
            assert ref == pytest.approx(intu, rel=1e-6)


@skip_in_parallel
def test_third_order_tri():
    #  *---*---*---*   3--11--10--2
    #  | \         |   | \        |
    #  *   *   *   *   8   7  15  13
    #  |     \     |   |    \     |
    #  *  *    *   *   9  14  6   12
    #  |         \ |   |        \ |
    #  *---*---*---*   0--4---5---1
    for H in (1.0, 2.0):
        for Z in (0.0, 0.5):
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
            cells = cells[:, perm_vtk(CellType.triangle, cells.shape[1])]

            cell = ufl.Cell("triangle", geometric_dimension=points.shape[1])
            domain = ufl.Mesh(ufl.VectorElement("Lagrange", cell, 3))
            mesh = create_mesh(MPI.COMM_WORLD, cells, points, domain)

            def e2(x):
                return x[2] + x[0] * x[1]
            # Interpolate function
            V = FunctionSpace(mesh, ("Lagrange", 3))
            u = Function(V)
            u.interpolate(e2)

            intu = assemble_scalar(u * dx(metadata={"quadrature_degree": 40}))
            intu = mesh.mpi_comm().allreduce(intu, op=MPI.SUM)

            nodes = [0, 9, 8, 3]
            ref = sympy_scipy(points, nodes, L, H)
            assert ref == pytest.approx(intu, rel=1e-6)


@skip_in_parallel
def test_fourth_order_tri():
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
    for H in (1.0, 2.0):
        for Z in (0.0, 0.5):
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
            cells = cells[:, perm_vtk(CellType.triangle, cells.shape[1])]
            cell = ufl.Cell("triangle", geometric_dimension=points.shape[1])
            domain = ufl.Mesh(ufl.VectorElement("Lagrange", cell, 4))
            mesh = create_mesh(MPI.COMM_WORLD, cells, points, domain)

            def e2(x):
                return x[2] + x[0] * x[1]

            # Interpolate function
            V = FunctionSpace(mesh, ("Lagrange", 4))
            u = Function(V)
            u.interpolate(e2)

            intu = assemble_scalar(u * dx(metadata={"quadrature_degree": 50}))
            intu = mesh.mpi_comm().allreduce(intu, op=MPI.SUM)
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
    cells = cells[:, perm_vtk(CellType.triangle, cells.shape[1])]

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

    cell = ufl.Cell("triangle", geometric_dimension=points.shape[1])
    domain = ufl.Mesh(ufl.VectorElement("Lagrange", cell, order))
    mesh = create_mesh(MPI.COMM_WORLD, cells, points, domain)

    # Find nodes corresponding to y axis
    nodes = []
    for j in range(points.shape[0]):
        if np.isclose(points[j][0], 0):
            nodes.append(j)

    def e2(x):
        return x[2] + x[0] * x[1]

    # For solution to be in functionspace
    V = FunctionSpace(mesh, ("CG", max(2, order)))
    u = Function(V)
    u.interpolate(e2)

    quad_order = 30
    intu = assemble_scalar(u * dx(metadata={"quadrature_degree": quad_order}))
    intu = mesh.mpi_comm().allreduce(intu, op=MPI.SUM)

    ref = scipy_one_cell(points, nodes)
    assert ref == pytest.approx(intu, rel=3e-3)


@skip_in_parallel
def test_xdmf_input_tri(datadir):
    with XDMFFile(MPI.COMM_WORLD, os.path.join(datadir, "mesh.xdmf"), "r", encoding=XDMFFile.Encoding.ASCII) as xdmf:
        mesh = xdmf.read_mesh(name="Grid")
    surface = assemble_scalar(1 * dx(mesh))
    assert mesh.mpi_comm().allreduce(surface, op=MPI.SUM) == pytest.approx(4 * np.pi, rel=1e-4)


@skip_in_parallel
@pytest.mark.parametrize('order', [2, 3])
def test_gmsh_input_quad(order):
    gmsh = pytest.importorskip("gmsh")
    R = 1
    res = 0.2
    algorithm = 2 if order == 2 else 5
    gmsh.initialize()
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", res)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", res)
    gmsh.option.setNumber("Mesh.Algorithm", algorithm)

    gmsh.model.occ.addSphere(0, 0, 0, 1, tag=1)
    gmsh.model.occ.synchronize()

    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.recombine()
    gmsh.model.mesh.setOrder(order)
    idx, points, _ = gmsh.model.mesh.getNodes()
    points = points.reshape(-1, 3)
    idx -= 1
    srt = np.argsort(idx)
    assert np.all(idx[srt] == np.arange(len(idx)))
    x = points[srt]

    element_types, element_tags, node_tags = gmsh.model.mesh.getElements(dim=2)
    name, dim, order, num_nodes, local_coords, num_first_order_nodes = gmsh.model.mesh.getElementProperties(
        element_types[0])

    cells = node_tags[0].reshape(-1, num_nodes) - 1
    gmsh_cell_id = gmsh.model.mesh.getElementType("quadrangle", order)
    gmsh.finalize()

    gmsh_quad = perm_gmsh(cpp.mesh.CellType.quadrilateral, (order + 1)**2)
    cells = cells[:, gmsh_quad]
    mesh = create_mesh(MPI.COMM_WORLD, cells, x, ufl_mesh_from_gmsh(gmsh_cell_id, x.shape[1]))
    surface = assemble_scalar(1 * dx(mesh))

    assert mesh.mpi_comm().allreduce(surface, op=MPI.SUM) == pytest.approx(4 * np.pi * R * R, rel=1e-5)

    # Bug related to VTK output writing
    # def e2(x):
    #     values = np.empty((x.shape[0], 1))
    #     values[:, 0] = x[:, 0]
    #     return values
    # cmap = fem.create_coordinate_map(mesh.mpi_comm(), mesh.ufl_domain())
    # mesh.geometry.coord_mapping = cmap
    # V = FunctionSpace(mesh, ("CG", order))
    # u = Function(V)
    # u.interpolate(e2)
    # from dolfinx.io import VTKFile
    # VTKFile("u{0:d}.pvd".format(order)).write(u)
    # print(min(u.vector.array),max(u.vector.array))
    # print(assemble_scalar(u*dx(mesh)))
