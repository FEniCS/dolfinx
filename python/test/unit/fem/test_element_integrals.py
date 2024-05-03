# Copyright (C) 2009-2020 Garth N. Wells, Matthew W. Scroggs and Jorgen S. Dokken
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for the fem interface"""

import random
from itertools import combinations, product

from mpi4py import MPI

import numpy as np
import pytest

import dolfinx
import ufl
from basix.ufl import element
from dolfinx.fem import (
    Constant,
    Function,
    assemble_matrix,
    assemble_scalar,
    assemble_vector,
    form,
    functionspace,
)
from dolfinx.mesh import CellType, create_mesh, meshtags

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


def unit_cell_points(cell_type, dtype):
    if cell_type == CellType.interval:
        return np.array([[0.0], [1.0]], dtype=dtype)
    if cell_type == CellType.triangle:
        # Define equilateral triangle with area 1
        root = 3**0.25  # 4th root of 3
        return np.array([[0.0, 0.0], [2 / root, 0.0], [1 / root, root]], dtype=dtype)
    if cell_type == CellType.tetrahedron:
        # Define regular tetrahedron with volume 1
        s = 2**0.5 * 3 ** (1 / 3)  # side length
        return np.array(
            [
                [0.0, 0.0, 0.0],
                [s, 0.0, 0.0],
                [s / 2, s * np.sqrt(3) / 2, 0.0],
                [s / 2, s / 2 / np.sqrt(3), s * np.sqrt(2 / 3)],
            ],
            dtype=dtype,
        )
    elif cell_type == CellType.quadrilateral:
        # Define unit quadrilateral (area 1)
        return np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=dtype)
    elif cell_type == CellType.hexahedron:
        # Define unit hexahedron (volume 1)
        return np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
            dtype=dtype,
        )


def unit_cell(cell_type, dtype, random_order=True):
    points = unit_cell_points(cell_type, dtype)
    num_points = len(points)

    # Randomly number the points and create the mesh
    order = list(range(num_points))
    if random_order:
        random.shuffle(order)
    ordered_points = np.zeros(points.shape, dtype=dtype)
    for i, j in enumerate(order):
        ordered_points[j] = points[i]
    cells = np.array([order])

    domain = ufl.Mesh(
        element("Lagrange", cell_type.name, 1, shape=(ordered_points.shape[1],), dtype=dtype)
    )
    mesh = create_mesh(MPI.COMM_WORLD, cells, ordered_points, domain)
    return mesh


def two_unit_cells(cell_type, dtype, agree=False, random_order=True, return_order=False):
    if cell_type == CellType.interval:
        points = np.array([[0.0], [1.0], [-1.0]], dtype=dtype)
        if agree:
            cells = [[0, 1], [2, 0]]
        else:
            cells = [[0, 1], [0, 2]]
    if cell_type == CellType.triangle:
        # Define equilateral triangles with area 1
        root = 3**0.25  # 4th root of 3
        points = np.array(
            [[0.0, 0.0], [2 / root, 0.0], [1 / root, root], [1 / root, -root]], dtype=dtype
        )
        if agree:
            cells = [[0, 1, 2], [0, 3, 1]]
        else:
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
            dtype=dtype,
        )
        if agree:
            cells = [[0, 1, 2, 3], [0, 1, 4, 2]]
        else:
            cells = [[0, 1, 2, 3], [0, 2, 1, 4]]
    elif cell_type == CellType.quadrilateral:
        # Define unit quadrilaterals (area 1)
        points = np.array(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, -1.0], [1.0, -1.0]], dtype=dtype
        )
        if agree:
            cells = [[0, 1, 2, 3], [4, 5, 0, 1]]
        else:
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
            dtype=dtype,
        )
        if agree:
            cells = [[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 0, 1, 2, 3]]
        else:
            cells = [[0, 1, 2, 3, 4, 5, 6, 7], [9, 11, 8, 10, 1, 3, 0, 2]]
    num_points = len(points)

    # Randomly number the points and create the mesh
    order = list(range(num_points))
    if random_order:
        random.shuffle(order)
    ordered_points = np.zeros(points.shape, dtype=dtype)
    for i, j in enumerate(order):
        ordered_points[j] = points[i]
    ordered_cells = np.array([[order[i] for i in c] for c in cells])

    domain = ufl.Mesh(
        element("Lagrange", cell_type.name, 1, shape=(ordered_points.shape[1],), dtype=dtype)
    )
    mesh = create_mesh(MPI.COMM_WORLD, ordered_cells, ordered_points, domain)
    if return_order:
        return mesh, order
    return mesh


@pytest.mark.skip_in_parallel
@parametrize_cell_types
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_facet_integral(cell_type, dtype):
    """Test that the integral of a function over a facet is correct"""
    xtype = np.real(dtype(0)).dtype
    for count in range(5):
        mesh = unit_cell(cell_type, xtype)
        tdim = mesh.topology.dim

        V = functionspace(mesh, ("Lagrange", 2))
        v = Function(V, dtype=dtype)

        mesh.topology.create_entities(tdim - 1)
        map_f = mesh.topology.index_map(tdim - 1)
        num_facets = map_f.size_local + map_f.num_ghosts
        indices = np.arange(0, num_facets)
        values = np.arange(0, num_facets, dtype=np.intc)
        marker = meshtags(mesh, tdim - 1, indices, values)

        # Functions that will have the same integral over each facet
        if cell_type == CellType.triangle:
            root = 3**0.25  # 4th root of 3
            v.interpolate(lambda x: (x[0] - 1 / root) ** 2 + (x[1] - root / 3) ** 2)
        elif cell_type == CellType.quadrilateral:
            v.interpolate(lambda x: x[0] * (1 - x[0]) + x[1] * (1 - x[1]))
        elif cell_type == CellType.tetrahedron:
            s = 2**0.5 * 3 ** (1 / 3)  # side length
            v.interpolate(
                lambda x: (x[0] - s / 2) ** 2
                + (x[1] - s / 2 / np.sqrt(3)) ** 2
                + (x[2] - s * np.sqrt(2 / 3) / 4) ** 2
            )
        elif cell_type == CellType.hexahedron:
            v.interpolate(lambda x: x[0] * (1 - x[0]) + x[1] * (1 - x[1]) + x[2] * (1 - x[2]))

        # Check that integral of these functions over each face are
        # equal
        mesh.topology.create_connectivity(tdim - 1, tdim)
        mesh.topology.create_connectivity(tdim, tdim - 1)
        out = []
        for j in range(num_facets):
            a = form(v * ufl.ds(subdomain_data=marker, subdomain_id=j), dtype=dtype)
            result = assemble_scalar(a)
            out.append(result)
            assert np.isclose(result, out[0], atol=np.finfo(dtype).eps)


@pytest.mark.skip_in_parallel
@parametrize_cell_types
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_facet_normals(cell_type, dtype):
    """Test that FacetNormal is outward facing"""
    xtype = np.real(dtype(0)).dtype
    for count in range(5):
        mesh = unit_cell(cell_type, xtype)
        tdim = mesh.topology.dim
        mesh.topology.create_entities(tdim - 1)

        gdim = mesh.geometry.dim
        V = functionspace(mesh, ("Lagrange", 1, (gdim,)))
        normal = ufl.FacetNormal(mesh)
        v = Function(V, dtype=dtype)

        mesh.topology.create_entities(tdim - 1)
        map_f = mesh.topology.index_map(tdim - 1)
        num_facets = map_f.size_local + map_f.num_ghosts
        indices = np.arange(0, num_facets)
        values = np.arange(0, num_facets, dtype=np.intc)
        marker = meshtags(mesh, tdim - 1, indices, values)

        # For each facet, check that the inner product of the normal and
        # the vector that has a positive normal component on only that
        # facet is positive
        for i in range(num_facets):
            if cell_type == CellType.interval:
                co = mesh.geometry.x[i]
                v.interpolate(lambda x: x[0] - co[0])
            if cell_type == CellType.triangle:
                co = mesh.geometry.x[i]
                # Vector function that is zero at `co` and points away
                # from `co` so that there is no normal component on two
                # edges and the integral over the other edge is 1
                v.interpolate(lambda x: ((x[0] - co[0]) / 2, (x[1] - co[1]) / 2))
            elif cell_type == CellType.tetrahedron:
                co = mesh.geometry.x[i]
                # Vector function that is zero at `co` and points away
                # from `co` so that there is no normal component on
                # three faces and the integral over the other edge is 1
                v.interpolate(
                    lambda x: ((x[0] - co[0]) / 3, (x[1] - co[1]) / 3, (x[2] - co[2]) / 3)
                )
            elif cell_type == CellType.quadrilateral:
                # function that is 0 on one edge and points away from
                # that edge so that there is no normal component on
                # three edges
                v.interpolate(
                    lambda x: tuple(x[j] - i % 2 if j == i // 2 else 0 * x[j] for j in range(2))
                )
            elif cell_type == CellType.hexahedron:
                # function that is 0 on one face and points away from
                # that face so that there is no normal component on five
                # faces
                v.interpolate(
                    lambda x: tuple(x[j] - i % 2 if j == i // 3 else 0 * x[j] for j in range(3))
                )

            # Check that integrals these functions dotted with the
            # normal over a face is 1 on one face and 0 on the others
            ones = 0
            for j in range(num_facets):
                a = form(
                    ufl.inner(v, normal) * ufl.ds(subdomain_data=marker, subdomain_id=j),
                    dtype=dtype,
                )
                result = assemble_scalar(a)
                if np.isclose(result, 1, atol=np.finfo(dtype).eps):
                    ones += 1
                else:
                    assert np.isclose(result, 0, atol=1.0e-6)
            assert ones == 1


@pytest.mark.skip_in_parallel
@pytest.mark.parametrize("space_type", ["Lagrange", "DG"])
@parametrize_cell_types
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_plus_minus(cell_type, space_type, dtype):
    """Test that ('+') and ('-') give the same value for continuous functions"""
    xtype = np.real(dtype(0)).dtype
    results = []
    for count in range(3):
        for agree in [True, False]:
            mesh = two_unit_cells(cell_type, xtype, agree)
            V = functionspace(mesh, (space_type, 1))
            v = Function(V, dtype=dtype)
            v.interpolate(lambda x: x[0] - 2 * x[1])
            # Check that these two integrals are equal
            for pm1, pm2 in product(["+", "-"], repeat=2):
                a = form(v(pm1) * v(pm2) * ufl.dS, dtype=dtype)
                results.append(assemble_scalar(a))
    for i, j in combinations(results, 2):
        assert np.isclose(i, j, atol=np.finfo(dtype).eps)


@pytest.mark.skip_in_parallel
@pytest.mark.parametrize("pm", ["+", "-"])
@parametrize_cell_types
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_plus_minus_simple_vector(cell_type, pm, dtype):
    """Test that ('+') and ('-') match up with the correct DOFs for DG functions"""
    xtype = np.real(dtype(0)).dtype
    results = []
    orders = []
    spaces = []
    for count in range(3):
        for agree in [True, False]:
            # Two cell mesh with randomly numbered points
            mesh, order = two_unit_cells(cell_type, xtype, agree, return_order=True)
            if cell_type in [CellType.interval, CellType.triangle, CellType.tetrahedron]:
                V = functionspace(mesh, ("DG", 1))
            else:
                V = functionspace(mesh, ("DQ", 1))

            # Assemble vectors v['+'] * dS and v['-'] * dS for a few
            # different numberings
            v = ufl.TestFunction(V)
            a = form(ufl.inner(1.0, v(pm)) * ufl.dS, dtype=dtype)
            result = assemble_vector(a)
            spaces.append(V)
            results.append(result.array)
            orders.append(order)

    # Check that the above vectors all have the same values as the first
    # one, but permuted due to differently ordered dofs
    dofmap0 = spaces[0].mesh.geometry.dofmap
    for result, space in zip(results[1:], spaces[1:]):
        # Get the data relating to two results
        dofmap1 = space.mesh.geometry.dofmap

        # For each cell
        for cell in range(2):
            # For each point in cell 0 in the first mesh
            for dof0, point0 in zip(spaces[0].dofmap.cell_dofs(cell), dofmap0[cell]):
                # Find the point in the cell 0 in the second mesh
                for dof1, point1 in zip(space.dofmap.cell_dofs(cell), dofmap1[cell]):
                    if np.allclose(
                        spaces[0].mesh.geometry.x[point0], space.mesh.geometry.x[point1]
                    ):
                        break
                else:
                    # If no matching point found, fail
                    assert False

                assert np.isclose(results[0][dof0], result[dof1], atol=np.finfo(dtype).eps)


@pytest.mark.skip_in_parallel
@pytest.mark.parametrize("pm1", ["+", "-"])
@pytest.mark.parametrize("pm2", ["+", "-"])
@parametrize_cell_types
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_plus_minus_vector(cell_type, pm1, pm2, dtype):
    """Test that ('+') and ('-') match up with the correct DOFs for DG functions"""
    xtype = np.real(dtype(0)).dtype
    results = []
    orders = []
    spaces = []
    for count in range(3):
        for agree in [True, False]:
            # Two cell mesh with randomly numbered points
            mesh, order = two_unit_cells(cell_type, xtype, agree, return_order=True)
            if cell_type in [CellType.interval, CellType.triangle, CellType.tetrahedron]:
                V = functionspace(mesh, ("DG", 1))
            else:
                V = functionspace(mesh, ("DQ", 1))

            # Assemble vectors with combinations of + and - for a few
            # different numberings
            f = Function(V, dtype=dtype)
            f.interpolate(lambda x: x[0] - 2 * x[1])
            v = ufl.TestFunction(V)
            a = form(ufl.inner(f(pm1), v(pm2)) * ufl.dS, dtype=dtype)
            result = assemble_vector(a)
            spaces.append(V)
            results.append(result.array)
            orders.append(order)

    # Check that the above vectors all have the same values as the first
    # one, but permuted due to differently ordered dofs
    dofmap0 = spaces[0].mesh.geometry.dofmap
    for result, space in zip(results[1:], spaces[1:]):
        # Get the data relating to two results
        dofmap1 = space.mesh.geometry.dofmap

        # For each cell
        for cell in range(2):
            # For each point in cell 0 in the first mesh
            for dof0, point0 in zip(spaces[0].dofmap.cell_dofs(cell), dofmap0[cell]):
                # Find the point in the cell 0 in the second mesh
                for dof1, point1 in zip(space.dofmap.cell_dofs(cell), dofmap1[cell]):
                    if np.allclose(
                        spaces[0].mesh.geometry.x[point0], space.mesh.geometry.x[point1]
                    ):
                        break
                else:
                    # If no matching point found, fail
                    assert False

                assert np.isclose(results[0][dof0], result[dof1], atol=1.0e-6)


@pytest.mark.skip_in_parallel
@pytest.mark.parametrize("pm1", ["+", "-"])
@pytest.mark.parametrize("pm2", ["+", "-"])
@parametrize_cell_types
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_plus_minus_matrix(cell_type, pm1, pm2, dtype):
    """Test that ('+') and ('-') match up with the correct DOFs for DG functions"""
    xtype = np.real(dtype(0)).dtype
    results = []
    spaces = []
    orders = []
    for count in range(3):
        for agree in [True, False]:
            # Two cell mesh with randomly numbered points
            mesh, order = two_unit_cells(cell_type, xtype, agree, return_order=True)
            V = functionspace(mesh, ("DG", 1))
            u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

            # Assemble matrices with combinations of + and - for a few
            # different numberings
            a = form(ufl.inner(u(pm1), v(pm2)) * ufl.dS, dtype=dtype)
            result = assemble_matrix(a, [])
            result.scatter_reverse()
            spaces.append(V)
            results.append(result.to_dense())
            orders.append(order)

    # Check that the above matrices all have the same values, but
    # permuted due to differently ordered dofs
    dofmap0 = spaces[0].mesh.geometry.dofmap
    for result, space in zip(results[1:], spaces[1:]):
        # Get the data relating to two results
        dofmap1 = space.mesh.geometry.dofmap
        dof_order = []

        # For each cell
        for cell in range(2):
            # For each point in cell 0 in the first mesh
            for dof0, point0 in zip(spaces[0].dofmap.cell_dofs(cell), dofmap0[cell]):
                # Find the point in the cell 0 in the second mesh
                for dof1, point1 in zip(space.dofmap.cell_dofs(cell), dofmap1[cell]):
                    if np.allclose(
                        spaces[0].mesh.geometry.x[point0], space.mesh.geometry.x[point1]
                    ):
                        break
                else:
                    # If no matching point found, fail
                    assert False

                dof_order.append((dof0, dof1))

        # For all dof pairs, check that entries in the matrix agree
        for a, b in dof_order:
            for c, d in dof_order:
                assert np.isclose(results[0][a, c], result[b, d], atol=np.finfo(dtype).eps)


@pytest.mark.skip(
    reason="Test needs replacing because it assumes the mesh constructor doesn't re-order points."
)
@pytest.mark.skip_in_parallel
@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("space_type", ["N1curl", "N2curl"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_curl(space_type, order, dtype):
    """Test that curl is consistent for different cell permutations of a tetrahedron."""
    xtype = np.real(dtype(0)).dtype
    tdim = dolfinx.mesh.cell_dim(CellType.tetrahedron)
    points = unit_cell_points(CellType.tetrahedron, xtype)

    spaces = []
    results = []
    cell = list(range(len(points)))
    random.seed(2)

    # Assemble vector on 5 randomly numbered cells
    for i in range(5):
        random.shuffle(cell)
        domain = ufl.Mesh(element("Lagrange", "tetrahedron", 1, shape=(3,), dtype=dtype))
        mesh = create_mesh(MPI.COMM_WORLD, [cell], points, domain)
        V = functionspace(mesh, (space_type, order))
        v = ufl.TestFunction(V)
        f = ufl.as_vector(tuple(1 if i == 0 else 0 for i in range(tdim)))
        L = form(ufl.inner(f, ufl.curl(v)) * ufl.dx)
        result = assemble_vector(L)
        spaces.append(V)
        results.append(result.array)

    # Set data for first space
    V0 = spaces[0]
    c10_0 = V.mesh.topology.connectivity(1, 0)

    # Check that all DOFs on edges agree

    # Loop over cell edges
    for i, edge in enumerate(V0.mesh.topology.connectivity(tdim, 1).links(0)):
        # Get the edge vertices
        vertices0 = c10_0.links(edge)  # Need to map back

        # Get assembled values on edge
        values0 = sorted(
            [result[V0.dofmap.cell_dofs(0)[a]] for a in V0.dofmap.dof_layout.entity_dofs(1, i)]
        )

        for V, result in zip(spaces[1:], results[1:]):
            # Get edge->vertex connectivity
            c10 = V.mesh.topology.connectivity(1, 0)

            # Loop over cell edges
            for j, e in enumerate(V.mesh.topology.connectivity(tdim, 1).links(0)):
                if sorted(c10.links(e)) == sorted(vertices0):  # need to map back c.links(e)
                    values = sorted(
                        [
                            result[V.dofmap.cell_dofs(0)[a]]
                            for a in V.dofmap.dof_layout.entity_dofs(1, j)
                        ]
                    )
                    assert np.allclose(values0, values)
                    break
            else:
                continue
            break


def create_quad_mesh(offset, dtype):
    """Creates a mesh of a single square element if offset = 0, or a
    trapezium element if |offset| > 0."""
    x = np.array([[0, 0], [1, 0], [0, 0.5 + offset], [1, 0.5 - offset]], dtype=dtype)
    cells = np.array([[0, 1, 2, 3]])
    ufl_mesh = ufl.Mesh(element("Lagrange", "quadrilateral", 1, shape=(2,), dtype=dtype))
    mesh = create_mesh(MPI.COMM_WORLD, cells, x, ufl_mesh)
    return mesh


@pytest.mark.skip_in_parallel
@pytest.mark.parametrize("k", [0, 1, 2])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_div_general_quads_mat(k, dtype):
    """Tests that assembling inner(u, div(w)) * dx, where u is from a
    "DQ" space and w is from an "RTCF" space, gives the same matrix for
    square and trapezoidal elements. This should be the case due to the
    properties of the Piola transform."""
    # Assemble matrix on a mesh of square elements and on a mesh of
    # trapezium elements
    xtype = np.real(dtype(0)).dtype

    def assemble_div_matrix(k, offset):
        mesh = create_quad_mesh(offset, dtype=xtype)
        V = functionspace(mesh, ("DQ", k))
        W = functionspace(mesh, ("RTCF", k + 1))
        u, w = ufl.TrialFunction(V), ufl.TestFunction(W)
        a = form(ufl.inner(u, ufl.div(w)) * ufl.dx, dtype=dtype)
        A = assemble_matrix(a)
        return A.to_dense()

    A_square = assemble_div_matrix(k, 0)
    A_trap = assemble_div_matrix(k, 0.25)

    # Due to the properties of the Piola transform, A_square and A_trap
    # should be equal
    assert np.allclose(A_square, A_trap, atol=1e-6)


@pytest.mark.skip_in_parallel
@pytest.mark.parametrize("k", [0, 1, 2])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_div_general_quads_vec(k, dtype):
    """Tests that assembling inner(1, div(w)) * dx, where w is from an
    "RTCF" space, gives the same matrix for square and trapezoidal
    elements. This should be the case due to the properties of the Piola
    transform."""
    # Assemble vector on a mesh of square elements and on a mesh of
    # trapezium elements
    xtype = np.real(dtype(0)).dtype

    def assemble_div_vector(k, offset):
        mesh = create_quad_mesh(offset, dtype=xtype)
        V = functionspace(mesh, ("RTCF", k + 1))
        v = ufl.TestFunction(V)
        L = form(ufl.inner(Constant(mesh, dtype(1)), ufl.div(v)) * ufl.dx, dtype=dtype)
        b = assemble_vector(L)
        return b.array

    L_square = assemble_div_vector(k, 0)
    L_trap = assemble_div_vector(k, 0.25)

    # Due to the properties of the Piola transform, L_square and L_trap
    # should be equal
    assert np.allclose(L_square, L_trap, atol=1e-5)
