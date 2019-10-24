# Copyright (C) 2019 Jørgen Schartum Dokken & Matthew Scroggs
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
""" Unit-tests for higher order meshes """

from dolfin import Mesh, MPI, Constant, fem, Function, FunctionSpace
from dolfin.cpp.mesh import CellType, GhostMode
from dolfin.fem import assemble_scalar
from dolfin_utils.test.skips import skip_in_parallel
from ufl import dx, SpatialCoordinate, sin
from dolfin.cpp.io import vtk_to_dolfin_ordering

import numpy as np
import pytest

import sympy as sp
import scipy.integrate
from sympy.vector import CoordSys3D, matrix_to_vector
import pygmsh
import meshio

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

    expr = (transform + x + x * y) * cross
    approx = sp.lambdify((x, y), expr)
    print(transform)
    ref = scipy.integrate.nquad(approx, [[0, L], [0, H]])[0]
    # Slow and only works for simple integrals
    # integral = sp.integrate(expr, (y, 0, H))
    # integral = sp.integrate(integral, (x, 0, L))
    # ex = integral.evalf()

    return ref


@pytest.mark.parametrize('L', [1])
@pytest.mark.parametrize('H', [1])
@pytest.mark.parametrize('Z', [0.5])
def test_quad_dofs_order_2(L, H, Z):
    # Test second order mesh by computing volume of two cells
    #  *-----*-----*   3--6--2--13-10
    #  |     |     |   |     |     |
    #  |     |     |   7  8  5  14 12
    #  |     |     |   |     |     |
    #  *-----*-----*   0--4--1--11-9
    points = np.array([[0, 0, 0], [L, 0, 0], [L, H, Z], [0, H, Z],
                       [L / 2, 0, 0], [L, H / 2, 0], [L / 2, H, Z], [0, H / 2, 0],
                       [L / 2, H / 2, 0],
                       [2 * L, 0, 0], [2 * L, H, Z],
                       [3 * L / 2, 0, 0], [2 * L, H / 2, 0], [3 * L / 2, H, Z],
                       [3 * L / 2, H / 2, 0]])

    cells = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8]])#, [1, 9, 10, 2, 11, 12, 13, 5, 14]])

    cells = vtk_to_dolfin_ordering(cells, CellType.quadrilateral)
    print(cells)
    mesh = Mesh(MPI.comm_world, CellType.quadrilateral, points, cells,
                [], GhostMode.none)
    from dolfin.io import VTKFile
    VTKFile("mesh.pvd").write(mesh)
    def e2(x):
        values = np.empty((x.shape[0], 1))
        values[:, 0] = x[:, 2] + x[:, 0] * x[:, 1]
        return values
    degree = mesh.degree()
    # Interpolate function
    V = FunctionSpace(mesh, ("CG", degree))
    print(V.dofmap.dofs(mesh,0))


    u = Function(V)
    cmap = fem.create_coordinate_map(mesh.ufl_domain())
    mesh.geometry.coord_mapping = cmap
    coord = V.tabulate_dof_coordinates()
    u.interpolate(e2)
    for i in range(len(coord)):
        print(coord[i], u.vector.array[i])
    intu = assemble_scalar(u * dx(metadata={"quadrature_degree": 30}))
    intu = MPI.sum(mesh.mpi_comm(), intu)
    VTKFile("u.pvd").write(u)
    nodes = [0, 3, 7]
    ref = sympy_scipy(points, nodes, L, H)
    print(ref, intu)

    assert ref == pytest.approx(intu, rel=1e-6)




@skip_in_parallel
@pytest.mark.parametrize('L', [1])
@pytest.mark.parametrize('H', [1, 5])
@pytest.mark.parametrize('eps', [0, 0.01])
def test_quad_dofs_order_3(L, H, eps):
    # Test third order mesh by computing volume of two cells
    #  *---------*   3--8--9--2-22-23-17
    #  |         |   |        |       |
    #  |         |   11 14 15 7 26 27 21
    #  |         |   |        |       |
    #  |         |   10 12 13 6 24 25 20
    #  |         |   |        |       |
    #  *---------*   0--4--5--1-18-19-16
    points = np.array([[0, 0], [L, 0], [L, H], [0, H],          # 0  1 2 3
                       [L / 3, - eps], [2 * L / 3, eps],         # 4  5
                       [L + eps, H / 3], [L - eps, 2 * H / 3],  # 6  7
                       [L / 3, H - eps], [2 * L / 3, H + eps],      # 8  9
                       [eps, H / 3], [-eps, 2 * H / 3],         # 10 11
                       [L / 3 + eps, H / 3 - eps], [2 * L / 3 + eps, H / 3 + eps],          # 12 13
                       [L / 3 - eps, 2 * H / 3 - eps], [2 * L / 3 - eps, 2 * H / 3 + eps],  # 14,15
                       [2 * L, 0], [2 * L, H],                 # 16 17
                       [4 * L / 3, eps], [5 * L / 3, -eps],    # 18 19
                       [2 * L + eps, H / 3], [2 * L - eps, 2 * H / 3],  # 20 21
                       [4 * L / 3, H - eps], [5 * L / 3, H + eps],      # 22 23
                       [4 * L / 3 + eps, H / 3 + eps], [5 * L / 3 + eps, H / 3 - eps],           # 24 25
                       [4 * L / 3 - eps, 2 * H / 3 - eps], [5 * L / 3 - eps, 2 * H / 3 + eps]])  # 26 27
    cells = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                      [1, 16, 17, 2, 18, 19, 20, 21, 22, 23, 6, 7, 24, 25, 26, 27]])

    mesh = Mesh(MPI.comm_world, CellType.quadrilateral, points, cells,
                [], GhostMode.none)

    vol = assemble_scalar(Constant(mesh, 1) * dx)
    assert(vol == pytest.approx(L * H * 2, rel=1e-9))

    # Volume of cell 1
    cell_1 = np.array([cells[0]])
    mesh = Mesh(MPI.comm_world, CellType.quadrilateral, points, cell_1,
                [], GhostMode.none)
    vol = assemble_scalar(Constant(mesh, 1) * dx)
    assert(vol == pytest.approx(L * H, rel=1e-9))

    # Volume of cell 2
    cell_2 = np.array([cells[1]])
    mesh = Mesh(MPI.comm_world, CellType.quadrilateral, points, cell_2,
                [], GhostMode.none)
    vol = assemble_scalar(Constant(mesh, 1) * dx)
    assert(vol == pytest.approx(L * H, rel=1e-9))


@skip_in_parallel
@pytest.mark.parametrize('L', [0.5])
@pytest.mark.parametrize('H', [1])
def test_quad_dofs_order_4(L, H):
    # Test fourth order mesh by computing volume of one cell
    #  *---------*   20-21-22-23-24-41--42--43--44
    #  |         |   |           |              |
    #  |         |   15 16 17 18 19 37  38  39  40
    #  |         |   |           |              |
    #  |         |   10 11 12 13 14 33  34  35  36
    #  |         |   |           |              |
    #  |         |   5  6  7  8  9  29  30  31  32
    #  |         |   |           |              |
    #  *---------*   0--1--2--3--4--25--26--27--28
    points = np.array([[0, 0], [L / 4, 0], [L / 2, 0], [3 * L / 4, 0], [L, 0],  # 0 1 2 3 4
                       [0, H / 4], [L / 4, H / 4], [L / 2, H / 4], [3 * L / 4, H / 4], [L, H / 4],  # 5 6 7 8 9
                       [0, H / 2], [L / 4, H / 2], [L / 2, H / 2], [3 * L / 4, H / 2], [L, H / 2],  # 10 11 12 13 14
                       [0, (3 / 4) * H], [L / 4, (3 / 4) * H], [L / 2, (3 / 4) * H],  # 15 16 17
                       [3 * L / 4, (3 / 4) * H], [L, (3 / 4) * H],  # 18 19
                       [0, H], [L / 4, H], [L / 2, H], [3 * L / 4, H], [L, H],  # 20, 21, 22, 23, 24
                       [(5 / 4) * L, 0], [(6 / 4) * L, 0], [(7 / 4) * L, 0], [2 * L, 0],  # 25 26 27 28
                       [(5 / 4) * L, H / 4], [(6 / 4) * L, H / 4], [(7 / 4) * L, H / 4], [2 * L, H / 4],  # 29 30 31 32
                       [(5 / 4) * L, H / 2], [(6 / 4) * L, H / 2], [(7 / 4) * L, H / 2], [2 * L, H / 2],  # 33 34 35 36
                       [(5 / 4) * L, 3 / 4 * H], [(6 / 4) * L, 3 / 4 * H],  # 37 38
                       [(7 / 4) * L, 3 / 4 * H], [2 * L, 3 / 4 * H],  # 39 40
                       [(5 / 4) * L, H], [(6 / 4) * L, H], [(7 / 4) * L, H], [2 * L, H]])  # 41 42 43 44

    # Lexicographical ordering, does not work due to mesh::compute_local_to_global_point_map
    # cells = np.array([list(range(25)) ,
    #                   [4,25,26,27,28,9,29,30,31,32,14,33,34,35,36,19,37,38,39,40,24,41,42,43,44]])

    cells = np.array([[0, 4, 24, 20, 1, 2, 3, 9, 14, 19, 21, 22, 23, 5, 10, 15, 6, 7, 8, 11, 12, 13, 16, 17, 18],
                      [4, 28, 44, 24, 25, 26, 27, 32, 36, 40, 41, 42, 43, 9, 14, 19,
                       29, 30, 31, 33, 34, 35, 37, 38, 39]])

    # First order
    # cells = np.array([[0,4,20,24], [4,28,24,44]])

    # Second order
    # points = np.array([[0, 0], [L / 2, 0], [L, 0],  # 0 1 2
    #                    [0, H / 2], [L / 2, H / 2],  [L, H / 2],  # 3 4 5
    #                    [0, H], [L / 2, H], [L, H],  # 6, 7, 8
    #                    [(6 / 4) * L, 0],       [2 * L, 0], # 9 10
    #                    [(6 / 4) * L, H / 2],    [2 * L, H / 2], # 11 12
    #                    [(6 / 4) * L, H],        [2 * L, H]]) # 13 14
    # Lexicographical ordering, does not work due to mesh::compute_local_to_global_point_map
    # cells = np.array([list(range(9)),[2,9,10,5,11,12,8,13,14]])

    # cells = np.array([[0,2,8,6,1,5,7,3,4], [2,10,14,8,9,12,13,5,11]]) # VTK ordering

    mesh = Mesh(MPI.comm_world, CellType.quadrilateral, points, cells,
                [], GhostMode.none)

    cmap = fem.create_coordinate_map(mesh.ufl_domain())
    mesh.geometry.coord_mapping = cmap
    vol = assemble_scalar(Constant(mesh, 1) * dx)
    assert(vol == pytest.approx(2 * L * H, rel=1e-9))

    def e1(x):
        values = np.empty((x.shape[0], 1))
        values[:, 0] = x[:, 0] * x[:, 1]
        return values

    def e2(x):
        values = np.empty((x.shape[0], 1))
        values[:, 0] = x[:, 0] * x[:, 1] * np.sin(x[:, 0])
        return values

    def e3(x):
        values = np.ones((x.shape[0], 1))
        return values

    def quantities(mesh):

        # FIXME: Need to permute dofmap
        # V = FunctionSpace(mesh, ("CG", 4))
        # u = Function(V)
        # u.interpolate(e1)
        x = SpatialCoordinate(mesh)
        u = x[0] * x[1]

        q1 = assemble_scalar(u * dx)
        u = x[0] * x[1] * sin(x[0])

        # FIXME: Need to permute dofmap
        # u.interpolate(e2)
        q2 = assemble_scalar(u * dx)

        # FIXME: Need to permute dofmap
        # u.interpolate(e3)
        u = Constant(mesh, 1)
        q3 = assemble_scalar(u * dx)
        return q1, q2, q3

    q1, q2, q3 = quantities(mesh)
    assert q1 == pytest.approx(0.25, rel=1e-9)
    assert q2 == pytest.approx(0.5 * (np.sin(1) - np.cos(1)),
                               rel=1e-9)
    assert q3 == pytest.approx(2 * L * H)
