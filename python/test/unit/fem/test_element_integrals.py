# Copyright (C) 2009-2020 Garth N. Wells, Matthew W. Scroggs and Jorgen S. Dokken
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for the fem interface"""

from random import shuffle
from itertools import combinations, product

import numpy as np
import pytest
from dolfinx_utils.test.skips import skip_in_parallel

from dolfinx import MPI, cpp, fem, Mesh, FunctionSpace, VectorFunctionSpace, FacetNormal, Function, MeshFunction
from ufl import inner, ds, dS, TestFunction, TrialFunction
from dolfinx.cpp.mesh import CellType


parametrize_cell_types = pytest.mark.parametrize(
    "cell_type",
    [CellType.interval, CellType.triangle, CellType.tetrahedron, CellType.quadrilateral, CellType.hexahedron])


def unit_cell(cell_type, random_order=True):
    if cell_type == CellType.interval:
        points = np.array([[0.], [1.]])
    if cell_type == CellType.triangle:
        # Define equilateral triangle with area 1
        root = 3 ** 0.25  # 4th root of 3
        points = np.array([[0., 0.], [2 / root, 0.],
                           [1 / root, root]])
    elif cell_type == CellType.tetrahedron:
        # Define regular tetrahedron with volume 1
        s = 2 ** 0.5 * 3 ** (1 / 3)  # side length
        points = np.array([[0., 0., 0.], [s, 0., 0.],
                           [s / 2, s * np.sqrt(3) / 2, 0.],
                           [s / 2, s / 2 / np.sqrt(3), s * np.sqrt(2 / 3)]])
    elif cell_type == CellType.quadrilateral:
        # Define unit quadrilateral (area 1)
        points = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
    elif cell_type == CellType.hexahedron:
        # Define unit hexahedron (volume 1)
        points = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
                           [1., 1., 0.], [0., 0., 1.], [1., 0., 1.],
                           [0., 1., 1.], [1., 1., 1.]])
    num_points = len(points)

    # Randomly number the points and create the mesh
    order = list(range(num_points))
    if random_order:
        shuffle(order)
    ordered_points = np.zeros(points.shape)
    for i, j in enumerate(order):
        ordered_points[j] = points[i]
    cells = np.array([order])
    mesh = Mesh(MPI.comm_world, cell_type, ordered_points, cells,
                [], cpp.mesh.GhostMode.none)
    mesh.geometry.coord_mapping = fem.create_coordinate_map(mesh)
    mesh.create_connectivity_all()
    return mesh


def two_unit_cells(cell_type, agree=False, random_order=True, return_order=False):
    if cell_type == CellType.interval:
        points = np.array([[0.], [1.], [-1.]])
        if agree:
            cells = [[0, 1], [2, 0]]
        else:
            cells = [[0, 1], [0, 2]]
    if cell_type == CellType.triangle:
        # Define equilateral triangles with area 1
        root = 3 ** 0.25  # 4th root of 3
        points = np.array([[0., 0.], [2 / root, 0.],
                           [1 / root, root], [1 / root, -root]])
        if agree:
            cells = [[0, 1, 2], [0, 3, 1]]
        else:
            cells = [[0, 1, 2], [1, 0, 3]]
    elif cell_type == CellType.tetrahedron:
        # Define regular tetrahedra with volume 1
        s = 2 ** 0.5 * 3 ** (1 / 3)  # side length
        points = np.array([[0., 0., 0.], [s, 0., 0.],
                           [s / 2, s * np.sqrt(3) / 2, 0.],
                           [s / 2, s / 2 / np.sqrt(3), s * np.sqrt(2 / 3)],
                           [s / 2, s / 2 / np.sqrt(3), -s * np.sqrt(2 / 3)]])
        if agree:
            cells = [[0, 1, 2, 3], [0, 1, 4, 2]]
        else:
            cells = [[0, 1, 2, 3], [0, 2, 1, 4]]
    elif cell_type == CellType.quadrilateral:
        # Define unit quadrilaterals (area 1)
        points = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.], [0., -1.], [1., -1.]])
        if agree:
            cells = [[0, 1, 2, 3], [4, 5, 0, 1]]
        else:
            cells = [[0, 1, 2, 3], [5, 1, 4, 0]]
    elif cell_type == CellType.hexahedron:
        # Define unit hexahedra (volume 1)
        points = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
                           [1., 1., 0.], [0., 0., 1.], [1., 0., 1.],
                           [0., 1., 1.], [1., 1., 1.], [0., 0., -1.],
                           [1., 0., -1.], [0., 1., -1.], [1., 1., -1.]])
        if agree:
            cells = [[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 0, 1, 2, 3]]
        else:
            cells = [[0, 1, 2, 3, 4, 5, 6, 7], [9, 11, 8, 10, 1, 3, 0, 2]]
    num_points = len(points)

    # Randomly number the points and create the mesh
    order = list(range(num_points))
    if random_order:
        shuffle(order)
    ordered_points = np.zeros(points.shape)
    for i, j in enumerate(order):
        ordered_points[j] = points[i]
    ordered_cells = np.array([[order[i] for i in c] for c in cells])
    mesh = Mesh(MPI.comm_world, cell_type, ordered_points, ordered_cells,
                [], cpp.mesh.GhostMode.none)
    mesh.geometry.coord_mapping = fem.create_coordinate_map(mesh)
    mesh.create_connectivity_all()
    if return_order:
        return mesh, order
    return mesh


@skip_in_parallel
@parametrize_cell_types
def test_facet_integral(cell_type):
    """Test that the integral of a function over a facet is correct"""
    for count in range(5):
        mesh = unit_cell(cell_type)

        V = FunctionSpace(mesh, ("Lagrange", 2))

        num_facets = mesh.num_entities(mesh.topology.dim - 1)

        v = Function(V)
        facet_function = MeshFunction("size_t", mesh, mesh.topology.dim - 1, 1)
        facet_function.values[:] = range(num_facets)

        # Functions that will have the same integral over each facet
        if cell_type == CellType.triangle:
            root = 3 ** 0.25  # 4th root of 3
            v.interpolate(lambda x: (x[0] - 1 / root) ** 2 + (x[1] - root / 3) ** 2)
        elif cell_type == CellType.quadrilateral:
            v.interpolate(lambda x: x[0] * (1 - x[0]) + x[1] * (1 - x[1]))
        elif cell_type == CellType.tetrahedron:
            s = 2 ** 0.5 * 3 ** (1 / 3)  # side length
            v.interpolate(lambda x: (x[0] - s / 2) ** 2 + (x[1] - s / 2 / np.sqrt(3)) ** 2
                          + (x[2] - s * np.sqrt(2 / 3) / 4) ** 2)
        elif cell_type == CellType.hexahedron:
            v.interpolate(lambda x: x[0] * (1 - x[0]) + x[1] * (1 - x[1]) + x[2] * (1 - x[2]))

        # assert that the integral of these functions over each face are equal
        out = []
        for j in range(num_facets):
            a = v * ds(subdomain_data=facet_function, subdomain_id=j)
            result = fem.assemble_scalar(a)
            out.append(result)
            assert np.isclose(result, out[0])


@skip_in_parallel
@parametrize_cell_types
def test_facet_normals(cell_type):
    """Test that FacetNormal is outward facing"""
    for count in range(5):
        mesh = unit_cell(cell_type)

        V = VectorFunctionSpace(mesh, ("Lagrange", 1))
        normal = FacetNormal(mesh)

        num_facets = mesh.num_entities(mesh.topology.dim - 1)

        v = Function(V)
        facet_function = MeshFunction("size_t", mesh, mesh.topology.dim - 1, 1)
        facet_function.values[:] = range(num_facets)

        # For each facet, check that the inner product of the normal and
        # the vector that has a positive normal component on only that facet
        # is positive
        for i in range(num_facets):
            if cell_type == CellType.interval:
                co = mesh.geometry.points[i]
                v.interpolate(lambda x: x[0] - co[0])
            if cell_type == CellType.triangle:
                co = mesh.geometry.points[i]
                # Vector function that is zero at `co` and points away from `co`
                # so that there is no normal component on two edges and the integral
                # over the other edge is 1
                v.interpolate(lambda x: ((x[0] - co[0]) / 2, (x[1] - co[1]) / 2))
            elif cell_type == CellType.tetrahedron:
                co = mesh.geometry.points[i]
                # Vector function that is zero at `co` and points away from `co`
                # so that there is no normal component on three faces and the integral
                # over the other edge is 1
                v.interpolate(lambda x: ((x[0] - co[0]) / 3, (x[1] - co[1]) / 3, (x[2] - co[2]) / 3))
            elif cell_type == CellType.quadrilateral:
                # function that is 0 on one edge and points away from that edge
                # so that there is no normal component on three edges
                v.interpolate(lambda x: tuple(x[j] - i % 2 if j == i // 2 else 0 * x[j] for j in range(2)))
            elif cell_type == CellType.hexahedron:
                # function that is 0 on one face and points away from that face
                # so that there is no normal component on five faces
                v.interpolate(lambda x: tuple(x[j] - i % 2 if j == i // 3 else 0 * x[j] for j in range(3)))

            # assert that the integrals these functions dotted with the normal over a face
            # is 1 on one face and 0 on the others
            ones = 0
            for j in range(num_facets):
                a = inner(v, normal) * ds(subdomain_data=facet_function, subdomain_id=j)
                result = fem.assemble_scalar(a)
                if np.isclose(result, 1):
                    ones += 1
                else:
                    assert np.isclose(result, 0)
            assert ones == 1


@skip_in_parallel
@pytest.mark.parametrize('space_type', ["CG", "DG"])
@parametrize_cell_types
def test_plus_minus(cell_type, space_type):
    """Test that ('+') and ('-') give the same value for continuous functions"""
    results = []
    for count in range(3):
        for agree in [True, False]:
            mesh = two_unit_cells(cell_type, agree)

            V = FunctionSpace(mesh, (space_type, 1))
            v = Function(V)
            v.interpolate(lambda x: x[0] - 2 * x[1])
            # Check that these two integrals are equal
            for pm1, pm2 in product(["+", "-"], repeat=2):
                a = v(pm1) * v(pm2) * dS
                results.append(fem.assemble_scalar(a))
    for i, j in combinations(results, 2):
        assert np.isclose(i, j)


@skip_in_parallel
@pytest.mark.parametrize('pm', ["+", "-"])
@parametrize_cell_types
def test_plus_minus_simple_vector(cell_type, pm):
    """Test that ('+') and ('-') match up with the correct DOFs for DG functions"""
    results = []
    orders = []
    spaces = []
    for count in range(3):
        for agree in [True, False]:
            mesh, order = two_unit_cells(cell_type, agree, return_order=True)

            if cell_type in [CellType.interval, CellType.triangle, CellType.tetrahedron]:
                V = FunctionSpace(mesh, ("DG", 1))
            else:
                V = FunctionSpace(mesh, ("DQ", 1))
            v = TestFunction(V)
            a = inner(1, v(pm)) * dS
            result = fem.assemble_vector(a)
            result.assemble()
            spaces.append(V)
            results.append(result)
            orders.append(order)

    for i, j in combinations(zip(results, spaces, orders), 2):
        dof_order = []
        for cell in range(2):
            for point in range(len(mesh.geometry.points)):
                point_n = j[2][point]
                cell_points = list(j[1].mesh.cells()[cell])
                if point_n in cell_points:
                    point_n_in_cell = cell_points.index(point_n)
                    dofmap = j[1].dofmap.cell_dofs(cell)
                    j_dof_n = dofmap[point_n_in_cell]
                else:
                    j_dof_n = None

                point_n = i[2][point]
                cell_points = list(i[1].mesh.cells()[cell])
                if point_n in cell_points:
                    point_n_in_cell = cell_points.index(point_n)
                    dofmap = i[1].dofmap.cell_dofs(cell)
                    i_dof_n = dofmap[point_n_in_cell]
                else:
                    i_dof_n = None

                if i_dof_n is None:
                    assert j_dof_n is None
                else:
                    dof_order.append((i_dof_n, j_dof_n))

        for a, b in dof_order:
            assert np.isclose(i[0][a], j[0][b])


@skip_in_parallel
@pytest.mark.parametrize('pm1', ["+", "-"])
@pytest.mark.parametrize('pm2', ["+", "-"])
@parametrize_cell_types
def test_plus_minus_vector(cell_type, pm1, pm2):
    """Test that ('+') and ('-') match up with the correct DOFs for DG functions"""
    results = []
    orders = []
    spaces = []
    for count in range(3):
        for agree in [True, False]:
            mesh, order = two_unit_cells(cell_type, agree, return_order=True)

            if cell_type in [CellType.interval, CellType.triangle, CellType.tetrahedron]:
                V = FunctionSpace(mesh, ("DG", 1))
            else:
                V = FunctionSpace(mesh, ("DQ", 1))
            f = Function(V)
            f.interpolate(lambda x: x[0] - 2 * x[1])
            v = TestFunction(V)
            a = inner(f(pm1), v(pm2)) * dS
            result = fem.assemble_vector(a)
            result.assemble()
            spaces.append(V)
            results.append(result)
            orders.append(order)

    for i, j in combinations(zip(results, spaces, orders), 2):
        dof_order = []
        for cell in range(2):
            for point in range(len(mesh.geometry.points)):
                point_n = j[2][point]
                cell_points = list(j[1].mesh.cells()[cell])
                if point_n in cell_points:
                    point_n_in_cell = cell_points.index(point_n)
                    dofmap = j[1].dofmap.cell_dofs(cell)
                    j_dof_n = dofmap[point_n_in_cell]
                else:
                    j_dof_n = None

                point_n = i[2][point]
                cell_points = list(i[1].mesh.cells()[cell])
                if point_n in cell_points:
                    point_n_in_cell = cell_points.index(point_n)
                    dofmap = i[1].dofmap.cell_dofs(cell)
                    i_dof_n = dofmap[point_n_in_cell]
                else:
                    i_dof_n = None

                if i_dof_n is None:
                    assert j_dof_n is None
                else:
                    dof_order.append((i_dof_n, j_dof_n))

        for a, b in dof_order:
            assert np.isclose(i[0][a], j[0][b])


@skip_in_parallel
@pytest.mark.parametrize('pm1', ["+", "-"])
@pytest.mark.parametrize('pm2', ["+", "-"])
@parametrize_cell_types
def test_plus_minus_matrix(cell_type, pm1, pm2):
    """Test that ('+') and ('-') match up with the correct DOFs for DG functions"""
    results = []
    spaces = []
    orders = []
    for count in range(3):
        for agree in [True, False]:
            mesh, order = two_unit_cells(cell_type, agree, return_order=True)

            V = FunctionSpace(mesh, ("DG", 1))
            u, v = TrialFunction(V), TestFunction(V)
            a = inner(u(pm1), v(pm2)) * dS
            result = fem.assemble_matrix(a, [])
            result.assemble()
            spaces.append(V)
            results.append(result)
            orders.append(order)

    for i, j in combinations(zip(results, spaces, orders), 2):
        dof_order = []
        for cell in range(2):
            for point in range(len(mesh.geometry.points)):
                point_n = j[2][point]
                cell_points = list(j[1].mesh.cells()[cell])
                if point_n in cell_points:
                    point_n_in_cell = cell_points.index(point_n)
                    dofmap = j[1].dofmap.cell_dofs(cell)
                    j_dof_n = dofmap[point_n_in_cell]
                else:
                    j_dof_n = None

                point_n = i[2][point]
                cell_points = list(i[1].mesh.cells()[cell])
                if point_n in cell_points:
                    point_n_in_cell = cell_points.index(point_n)
                    dofmap = i[1].dofmap.cell_dofs(cell)
                    i_dof_n = dofmap[point_n_in_cell]
                else:
                    i_dof_n = None

                if i_dof_n is None:
                    assert j_dof_n is None
                else:
                    dof_order.append((i_dof_n, j_dof_n))

        for a, b in dof_order:
            for c, d in dof_order:
                assert np.isclose(i[0][a, c], j[0][b, d])
