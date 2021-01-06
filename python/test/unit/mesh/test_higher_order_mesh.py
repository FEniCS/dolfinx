# Copyright (C) 2019 Jørgen Schartum Dokken and Matthew Scroggs
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
""" Unit-tests for higher order meshes """

import os

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


@skip_in_parallel
def test_xdmf_input_tri(datadir):
    with XDMFFile(MPI.COMM_WORLD, os.path.join(datadir, "mesh.xdmf"), "r", encoding=XDMFFile.Encoding.ASCII) as xdmf:
        mesh = xdmf.read_mesh(name="Grid")
    surface = assemble_scalar(1 * dx(mesh))
    assert mesh.mpi_comm().allreduce(surface, op=MPI.SUM) == pytest.approx(4 * np.pi, rel=1e-4)


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
    cells = cells[:, perm_vtk(CellType.quadrilateral, cells.shape[1])]
    cell = ufl.Cell("quadrilateral", geometric_dimension=points.shape[1])
    domain = ufl.Mesh(ufl.VectorElement("Lagrange", cell, 2))
    mesh = create_mesh(MPI.COMM_WORLD, cells, points, domain)

    def e2(x):
        return x[2] + x[0] * x[1]

    # Interpolate function
    V = FunctionSpace(mesh, ("CG", 2))
    u = Function(V)
    u.interpolate(e2)

    intu = assemble_scalar(u * dx(mesh))
    intu = mesh.mpi_comm().allreduce(intu, op=MPI.SUM)

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
                       [2 * L, 0, 0], [2 * L, H, Z],                      # 16 17
                       [4 * L / 3, 0, 0], [5 * L / 3, 0, 0],              # 18 19
                       [2 * L, H / 3, 0], [2 * L, 2 * H / 3, 0],          # 20 21
                       [4 * L / 3, H, Z], [5 * L / 3, H, Z],              # 22 23
                       [4 * L / 3, H / 3, 0], [5 * L / 3, H / 3, 0],           # 24 25
                       [4 * L / 3, 2 * H / 3, 0], [5 * L / 3, 2 * H / 3, 0]])  # 26 27

    # Change to multiple cells when matthews dof-maps work for quads
    cells = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                      [1, 16, 17, 2, 18, 19, 20, 21, 22, 23, 6, 7, 24, 25, 26, 27]])
    cells = cells[:, perm_vtk(CellType.quadrilateral, cells.shape[1])]

    assert (cells[0] == [0, 1, 3, 2, 4, 5, 10, 11, 6, 7, 8, 9, 12, 13, 14, 15]).all()
    assert (cells[1] == [1, 16, 2, 17, 18, 19, 6, 7, 20, 21, 22, 23, 24, 25, 26, 27]).all()

    cell = ufl.Cell("quadrilateral", geometric_dimension=points.shape[1])
    domain = ufl.Mesh(ufl.VectorElement("Lagrange", cell, 3))
    mesh = create_mesh(MPI.COMM_WORLD, cells, points, domain)

    def e2(x):
        return x[2] + x[0] * x[1]

    # Interpolate function
    V = FunctionSpace(mesh, ("CG", 3))
    u = Function(V)
    u.interpolate(e2)

    intu = assemble_scalar(u * dx(mesh))
    intu = mesh.mpi_comm().allreduce(intu, op=MPI.SUM)

    nodes = [0, 3, 10, 11]
    ref = sympy_scipy(points, nodes, 2 * L, H)
    assert ref == pytest.approx(intu, rel=1e-6)


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
    print(cells[0])
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
