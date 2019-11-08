# Copyright (C) 2019 Jørgen Schartum Dokken and Matthew Scroggs
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
""" Unit-tests for higher order meshes """

import meshio
import numpy as np
import pygmsh
import pytest
import scipy.integrate
import sympy as sp
from dolfin_utils.test.skips import skip_in_parallel
from sympy.vector import CoordSys3D, matrix_to_vector

from dolfin import MPI, Function, FunctionSpace, Mesh, fem
from dolfin.cpp.io import permute_cell_ordering, permutation_vtk_to_dolfin
from dolfin.cpp.mesh import CellType, GhostMode
from dolfin.fem import assemble_scalar
from dolfin.io import XDMFFile
from ufl import dx


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
@pytest.mark.parametrize('H', [1, 2])
@pytest.mark.parametrize('Z', [0, 0.5])
def test_second_order_tri(H, Z):
    # Test second order mesh by computing volume of two cells
    #  *-----*-----*   3----6-----2
    #  | \         |   | \        |
    #  |   \       |   |   \      |
    #  *     *     *   7     8    5
    #  |       \   |   |      \   |
    #  |         \ |   |        \ |
    #  *-----*-----*   0----4-----1

    L = 1
    points = np.array([[0, 0, 0], [L, 0, 0], [L, H, Z], [0, H, Z],
                       [L / 2, 0, 0], [L, H / 2, 0], [L / 2, H, Z],
                       [0, H / 2, 0], [L / 2, H / 2, 0]])

    cells = np.array([[0, 1, 3, 4, 8, 7],
                      [1, 2, 3, 5, 6, 8]])
    cells = permute_cell_ordering(cells, permutation_vtk_to_dolfin(CellType.triangle, cells.shape[1]))
    mesh = Mesh(MPI.comm_world, CellType.triangle, points, cells, [], GhostMode.none)

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

    intu = assemble_scalar(u * dx(mesh, metadata={"quadrature_degree": 20}))
    intu = MPI.sum(mesh.mpi_comm(), intu)

    nodes = [0, 3, 7]
    ref = sympy_scipy(points, nodes, L, H)
    assert ref == pytest.approx(intu, rel=1e-6)


@skip_in_parallel
@pytest.mark.parametrize('H', [1, 2])
@pytest.mark.parametrize('Z', [0, 0.5])
def test_third_order_tri(H, Z):
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
    cells = permute_cell_ordering(cells, permutation_vtk_to_dolfin(CellType.triangle, cells.shape[1]))
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
    intu = MPI.sum(mesh.mpi_comm(), intu)

    nodes = [0, 9, 8, 3]
    ref = sympy_scipy(points, nodes, L, H)
    assert ref == pytest.approx(intu, rel=1e-6)


@skip_in_parallel
@pytest.mark.parametrize('H', [1, 2])
@pytest.mark.parametrize('Z', [0, 0.5])
def test_fourth_order_tri(H, Z):
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
    cells = permute_cell_ordering(cells, permutation_vtk_to_dolfin(CellType.triangle, cells.shape[1]))

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
    intu = MPI.sum(mesh.mpi_comm(), intu)
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
    cells = permute_cell_ordering(cells, permutation_vtk_to_dolfin(CellType.triangle, cells.shape[1]))

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
    intu = MPI.sum(mesh.mpi_comm(), intu)

    ref = scipy_one_cell(points, nodes)
    assert ref == pytest.approx(intu, rel=3e-3)


def test_xdmf_input_tri():
    # Parameterize test if gmsh gets wider support
    order = 2
    R = 1
    res = R / 7
    geo = pygmsh.opencascade.Geometry()
    geo.add_raw_code("Mesh.ElementOrder={0:d};".format(order))
    geo.add_ball([0, 0, 0], R, char_length=res)
    element = "triangle{0:d}".format(int((order + 1) * (order + 2) / 2))
    if order == 1:
        element = element[:-1]
    msh = pygmsh.generate_mesh(geo, verbose=True, dim=2)
    meshio.write("mesh.xdmf", meshio.Mesh(points=msh.points, cells={element: msh.cells[element]}))
    with XDMFFile(MPI.comm_world, "mesh.xdmf") as xdmf:
        mesh = xdmf.read_mesh(GhostMode.none)

    surface = assemble_scalar(1 * dx(mesh))
    assert MPI.sum(mesh.mpi_comm(), surface) == pytest.approx(4 * np.pi * R * R, rel=1e-5)


@skip_in_parallel
@pytest.mark.parametrize('L', [1, 2])
@pytest.mark.parametrize('H', [1])
@pytest.mark.parametrize('Z', [0, 0.3])
def test_second_order_quad(L, H, Z):
    """ Test by comparing integration of z+x*y against sympy/scipy
    integration of a quad element. Z>0 implies curved element.

      *-----*   3--6--2
      |     |   |     |
      |     |   7  8  5
      |     |   |     |
      *-----*   0--4--1

    """

    points = np.array([[0, 0, 0], [L, 0, 0], [L, H, Z], [0, H, Z],
                       [L / 2, 0, 0], [L, H / 2, 0],
                       [L / 2, H, Z], [0, H / 2, 0],
                       [L / 2, H / 2, 0],
                       [2 * L, 0, 0], [2 * L, H, Z]])
    cells = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8]])
    cells = permute_cell_ordering(cells, permutation_vtk_to_dolfin(CellType.quadrilateral, cells.shape[1]))

    mesh = Mesh(MPI.comm_world, CellType.quadrilateral, points, cells,
                [], GhostMode.none)

    def e2(x):
        values = np.empty((x.shape[0], 1))
        values[:, 0] = x[:, 2] + x[:, 0] * x[:, 1]
        return values

    # Interpolate function
    V = FunctionSpace(mesh, ("CG", 2))
    u = Function(V)
    cmap = fem.create_coordinate_map(mesh.ufl_domain())

    mesh.geometry.coord_mapping = cmap

    u.interpolate(e2)

    intu = assemble_scalar(u * dx(mesh))
    intu = MPI.sum(mesh.mpi_comm(), intu)

    nodes = [0, 3, 7]
    ref = sympy_scipy(points, nodes, L, H)
    assert ref == pytest.approx(intu, rel=1e-6)


@skip_in_parallel
@pytest.mark.parametrize('L', [1, 2])
@pytest.mark.parametrize('H', [1])
@pytest.mark.parametrize('Z', [0, 0.3])
def test_third_order_quad(L, H, Z):
    """Test by comparing integration of z+x*y against sympy/scipy integration
    of a quad element. Z>0 implies curved element.

      *---------*   3--8--9--2-22-23-17
      |         |   |        |       |
      |         |   11 14 15 7 26 27 21
      |         |   |        |       |
      |         |   10 12 13 6 24 25 20
      |         |   |        |       |
      *---------*   0--4--5--1-18-19-16

    """
    points = np.array([[0, 0, 0], [L, 0, 0], [L, H, Z], [0, H, Z],        # 0  1 2 3
                       [L / 3, 0, 0], [2 * L / 3, 0, 0],                  # 4  5
                       [L, H / 3, 0], [L, 2 * H / 3, 0],                  # 6  7
                       [L / 3, H, Z], [2 * L / 3, H, Z],                  # 8  9
                       [0, H / 3, 0], [0, 2 * H / 3, 0],                  # 10 11
                       [L / 3, H / 3, 0], [2 * L / 3, H / 3, 0],          # 12 13
                       [L / 3, 2 * H / 3, 0], [2 * L / 3, 2 * H / 3, 0],  # 14 15
                       [2 * L, 0, 0], [2 * L, H, 0],                      # 16 17
                       [4 * L / 3, 0, 0], [5 * L / 3, 0, 0],              # 18 19
                       [2 * L, H / 3, 0], [2 * L, 2 * H / 3, 0],          # 20 21
                       [4 * L / 3, H, 0], [5 * L / 3, H, 0],              # 22 23
                       [4 * L / 3, H / 3, 0], [5 * L / 3, H / 3, 0],           # 24 25
                       [4 * L / 3, 2 * H / 3, 0], [5 * L / 3, 2 * H / 3, 0]])  # 26 27

    # Change to multiple cells when matthews dof-maps work for quads
    cells = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]])
    # ,[1, 16, 17, 2, 18, 19, 20, 21, 22, 23, 6, 7, 24, 25, 26, 27]])

    cells = permute_cell_ordering(cells, permutation_vtk_to_dolfin(CellType.quadrilateral, cells.shape[1]))
    mesh = Mesh(MPI.comm_world, CellType.quadrilateral, points, cells,
                [], GhostMode.none)

    def e2(x):
        values = np.empty((x.shape[0], 1))
        values[:, 0] = x[:, 2] + x[:, 0] * x[:, 1]
        return values

    # Interpolate function
    V = FunctionSpace(mesh, ("CG", 3))
    u = Function(V)
    cmap = fem.create_coordinate_map(mesh.ufl_domain())

    mesh.geometry.coord_mapping = cmap

    u.interpolate(e2)

    intu = assemble_scalar(u * dx(mesh))
    intu = MPI.sum(mesh.mpi_comm(), intu)

    nodes = [0, 3, 10, 11]
    ref = sympy_scipy(points, nodes, L, H)
    assert ref == pytest.approx(intu, rel=1e-6)


@skip_in_parallel
@pytest.mark.parametrize('L', [1, 2])
@pytest.mark.parametrize('H', [1])
@pytest.mark.parametrize('Z', [0, 0.3])
def test_fourth_order__quad(L, H, Z):
    """Test by comparing integration of z+x*y against sympy/scipy integration
    of a quad element. Z>0 implies curved element.

      *---------*   20-21-22-23-24-41--42--43--44
      |         |   |           |              |
      |         |   15 16 17 18 19 37  38  39  40
      |         |   |           |              |
      |         |   10 11 12 13 14 33  34  35  36
      |         |   |           |              |
      |         |   5  6  7  8  9  29  30  31  32
      |         |   |           |              |
      *---------*   0--1--2--3--4--25--26--27--28

    """
    points = np.array([[0, 0, 0], [L / 4, 0, 0], [L / 2, 0, 0],               # 0 1 2
                       [3 * L / 4, 0, 0], [L, 0, 0],                          # 3 4
                       [0, H / 4, -Z / 3], [L / 4, H / 4, -Z / 3], [L / 2, H / 4, -Z / 3],   # 5 6 7
                       [3 * L / 4, H / 4, -Z / 3], [L, H / 4, -Z / 3],                  # 8 9
                       [0, H / 2, 0], [L / 4, H / 2, 0], [L / 2, H / 2, 0],   # 10 11 12
                       [3 * L / 4, H / 2, 0], [L, H / 2, 0],                  # 13 14
                       [0, (3 / 4) * H, 0], [L / 4, (3 / 4) * H, 0],          # 15 16
                       [L / 2, (3 / 4) * H, 0], [3 * L / 4, (3 / 4) * H, 0],  # 17 18
                       [L, (3 / 4) * H, 0], [0, H, Z], [L / 4, H, Z],         # 19 20 21
                       [L / 2, H, Z], [3 * L / 4, H, Z], [L, H, Z],           # 22 23 24
                       [(5 / 4) * L, 0, 0], [(6 / 4) * L, 0, 0],              # 25 26
                       [(7 / 4) * L, 0, 0], [2 * L, 0, 0],                    # 27 28
                       [(5 / 4) * L, H / 4, -Z / 3], [(6 / 4) * L, H / 4, -Z / 3],      # 29 30
                       [(7 / 4) * L, H / 4, -Z / 3], [2 * L, H / 4, -Z / 3],            # 31 32
                       [(5 / 4) * L, H / 2, 0], [(6 / 4) * L, H / 2, 0],      # 33 34
                       [(7 / 4) * L, H / 2, 0], [2 * L, H / 2, 0],            # 35 36
                       [(5 / 4) * L, 3 / 4 * H, 0],                           # 37
                       [(6 / 4) * L, 3 / 4 * H, 0],                           # 38
                       [(7 / 4) * L, 3 / 4 * H, 0], [2 * L, 3 / 4 * H, 0],    # 39 40
                       [(5 / 4) * L, H, Z], [(6 / 4) * L, H, Z],              # 41 42
                       [(7 / 4) * L, H, Z], [2 * L, H, Z]])                   # 43 44

    # VTK ordering
    cells = np.array([[0, 4, 24, 20, 1, 2, 3, 9, 14, 19, 21, 22, 23, 5, 10, 15, 6, 7, 8, 11, 12, 13, 16, 17, 18]])
    #  , [4, 28, 44, 24, 25, 26, 27, 32, 36, 40, 41, 42, 43, 9, 14, 19,
    #     29, 30, 31, 33, 34, 35, 37, 38, 39]])

    cells = permute_cell_ordering(cells, permutation_vtk_to_dolfin(CellType.quadrilateral, cells.shape[1]))
    mesh = Mesh(MPI.comm_world, CellType.quadrilateral, points, cells,
                [], GhostMode.none)

    def e2(x):
        values = np.empty((x.shape[0], 1))
        values[:, 0] = x[:, 2] + x[:, 0] * x[:, 1]
        return values

    V = FunctionSpace(mesh, ("CG", 4))
    u = Function(V)
    cmap = fem.create_coordinate_map(mesh.ufl_domain())

    mesh.geometry.coord_mapping = cmap

    u.interpolate(e2)

    intu = assemble_scalar(u * dx(mesh))
    intu = MPI.sum(mesh.mpi_comm(), intu)

    nodes = [0, 5, 10, 15, 20]
    ref = sympy_scipy(points, nodes, L, H)
    assert ref == pytest.approx(intu, rel=1e-5)


@skip_in_parallel
@pytest.mark.parametrize('order', [2, 3])
def test_gmsh_input_quad(order):
    # Parameterize test if gmsh gets wider support
    R = 1
    res = 0.2 if order == 2 else 0.2
    algorithm = 2 if order == 2 else 5
    element = "quad{0:d}".format(int((order + 1)**2))

    geo = pygmsh.opencascade.Geometry()
    geo.add_raw_code("Mesh.ElementOrder={0:d};".format(order))
    geo.add_ball([0, 0, 0], R, char_length=res)
    geo.add_raw_code("Recombine Surface {1};")
    geo.add_raw_code("Mesh.Algorithm = {0:d};".format(algorithm))

    msh = pygmsh.generate_mesh(geo, verbose=True, dim=2)

    if order > 2:
        # Quads order > 3 have a gmsh specific ordering, and has to be permuted.
        msh_to_dolfin = np.array([0, 3, 11, 10, 1, 2, 6, 7, 4, 9, 12, 15, 5, 8, 13, 14])
        cells = np.zeros(msh.cells[element].shape)
        for i in range(len(cells)):
            for j in range(len(msh_to_dolfin)):
                cells[i, j] = msh.cells[element][i, msh_to_dolfin[j]]
    else:
        # XDMF does not support higher order quads
        cells = permute_cell_ordering(msh.cells[element], permutation_vtk_to_dolfin(
            CellType.quadrilateral, msh.cells[element].shape[1]))

    mesh = Mesh(MPI.comm_world, CellType.quadrilateral, msh.points, cells,
                [], GhostMode.none)
    surface = assemble_scalar(1 * dx(mesh))

    assert MPI.sum(mesh.mpi_comm(), surface) == pytest.approx(4 * np.pi * R * R, rel=1e-5)

    # Bug related to VTK output writing
    # def e2(x):
    #     values = np.empty((x.shape[0], 1))
    #     values[:, 0] = x[:, 0]
    #     return values
    # cmap = fem.create_coordinate_map(mesh.ufl_domain())
    # mesh.geometry.coord_mapping = cmap
    # V = FunctionSpace(mesh, ("CG", order))
    # u = Function(V)
    # u.interpolate(e2)
    # from dolfin.io import VTKFile
    # VTKFile("u{0:d}.pvd".format(order)).write(u)
    # print(min(u.vector.array),max(u.vector.array))
    # print(assemble_scalar(u*dx(mesh)))
