# Copyright (C) 2019-2021 Jørgen Schartum Dokken and Matthew Scroggs
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit-tests for higher order meshes"""

import random
from pathlib import Path

from mpi4py import MPI

import numpy as np
import pytest

import basix
import ufl
from basix.ufl import element
from dolfinx import default_real_type
from dolfinx.cpp.io import perm_vtk
from dolfinx.fem import assemble_scalar, form
from dolfinx.io import XDMFFile
from dolfinx.io.gmshio import cell_perm_array, ufl_mesh
from dolfinx.mesh import CellType, create_mesh, create_submesh
from ufl import dx


def check_cell_volume(points, cell, domain, volume, dtype):
    random.seed(13)

    point_order = [i for i, _ in enumerate(points)]
    for repeat in range(5):
        # Shuffle the cell to check that permutations of
        # CoordinateElement are correct
        random.shuffle(point_order)
        ordered_points = np.zeros((len(points), len(points[0])))
        for i, j in enumerate(point_order):
            ordered_points[j] = points[i]
        ordered_cell = [point_order[i] for i in cell]

        ordered_points = np.array(ordered_points, dtype=dtype)
        mesh = create_mesh(MPI.COMM_WORLD, [ordered_cell], ordered_points, domain)
        area = assemble_scalar(form(1 * dx(mesh), dtype=dtype))
        assert np.isclose(area, volume)


@pytest.mark.skip_in_parallel
@pytest.mark.parametrize("order", range(1, 5))
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_submesh(order, dtype):
    # Generate a single cell higher order mesh
    points = []
    points += [[i / order, j / order, 0] for j in range(order + 1) for i in range(order + 1 - j)]
    for k in range(1, order):
        points += [
            [i / order, j / order + 0.1, k / order]
            for j in range(order + 1 - k)
            for i in range(order + 1 - k - j)
        ]

    points += [[0, 0, 1]]

    def coord_to_vertex(x, y, z):
        return (
            z * (3 * order**2 - 3 * order * z + 12 * order + z**2 - 6 * z + 11) // 6
            + y * (2 * (order - z) + 3 - y) // 2
            + x
        )

    # Define a cell using DOLFINx ordering
    cell = [
        coord_to_vertex(x, y, z)
        for x, y, z in [(0, 0, 0), (order, 0, 0), (0, order, 0), (0, 0, order)]
    ]

    if order > 1:
        for i in range(1, order):
            cell.append(coord_to_vertex(0, order - i, i))
        for i in range(1, order):
            cell.append(coord_to_vertex(order - i, 0, i))
        for i in range(1, order):
            cell.append(coord_to_vertex(order - i, i, 0))
        for i in range(1, order):
            cell.append(coord_to_vertex(0, 0, i))
        for i in range(1, order):
            cell.append(coord_to_vertex(0, i, 0))
        for i in range(1, order):
            cell.append(coord_to_vertex(i, 0, 0))

        for j in range(1, order):
            for i in range(1, order - j):
                cell.append(coord_to_vertex(order - i - j, i, j))
        for j in range(1, order):
            for i in range(1, order - j):
                cell.append(coord_to_vertex(0, i, j))
        for j in range(1, order):
            for i in range(1, order - j):
                cell.append(coord_to_vertex(i, 0, j))
        for j in range(1, order):
            for i in range(1, order - j):
                cell.append(coord_to_vertex(i, j, 0))

        for k in range(1, order):
            for j in range(1, order - k):
                for i in range(1, order - j - k):
                    cell.append(coord_to_vertex(i, j, k))

    domain = ufl.Mesh(
        element(
            "Lagrange",
            "tetrahedron",
            order,
            lagrange_variant=basix.LagrangeVariant.equispaced,
            shape=(3,),
            dtype=dtype,
        )
    )
    points = np.array(points, dtype=dtype)
    mesh = create_mesh(MPI.COMM_WORLD, [cell], points, domain)
    for i in range(mesh.topology.dim):
        mesh.topology.create_entities(i)
    md = {"quadrature_degree": 10}
    measures = (ufl.ds(mesh, metadata=md), ufl.dx(mesh, metadata=md))
    dimensions = (mesh.topology.dim - 1, mesh.topology.dim)
    # Check that creating a submesh of single cell mesh, consisting of:
    # 1. The cell
    # 2. The facets of the cell
    # Gives the correct computation of: volume (case 1) or surface area (case 2)
    for dim, dC in zip(dimensions, measures):
        # Integrate on original mesh
        value = assemble_scalar(form(1 * dC, dtype=dtype))
        num_local_entities = mesh.topology.index_map(dim).size_local
        submesh, _, _, _ = create_submesh(mesh, dim, np.arange(num_local_entities, dtype=np.int32))
        submesh_area = assemble_scalar(form(1 * ufl.dx(submesh, metadata=md), dtype=dtype))
        assert np.isclose(value, submesh_area)


@pytest.mark.skip_in_parallel
@pytest.mark.parametrize("order", range(1, 5))
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_triangle_mesh(order, dtype):
    points = []
    points += [[i / order, 0] for i in range(order + 1)]
    for j in range(1, order):
        points += [[i / order + 0.1, j / order] for i in range(order + 1 - j)]
    points += [[0, 1]]

    def coord_to_vertex(x, y):
        return y * (2 * order + 3 - y) // 2 + x

    # Define a cell using DOLFINx ordering
    cell = [coord_to_vertex(i, j) for i, j in [(0, 0), (order, 0), (0, order)]]
    if order > 1:
        for i in range(1, order):
            cell.append(coord_to_vertex(order - i, i))
        for i in range(1, order):
            cell.append(coord_to_vertex(0, i))
        for i in range(1, order):
            cell.append(coord_to_vertex(i, 0))

        for j in range(1, order):
            for i in range(1, order - j):
                cell.append(coord_to_vertex(i, j))

    domain = ufl.Mesh(
        element(
            "Lagrange",
            "triangle",
            order,
            lagrange_variant=basix.LagrangeVariant.equispaced,
            shape=(2,),
            dtype=dtype,
        )
    )
    check_cell_volume(points, cell, domain, 0.5, dtype=dtype)


@pytest.mark.skip_in_parallel
@pytest.mark.parametrize("order", range(1, 5))
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_tetrahedron_mesh(order, dtype):
    points = []
    points += [[i / order, j / order, 0] for j in range(order + 1) for i in range(order + 1 - j)]
    for k in range(1, order):
        points += [
            [i / order, j / order + 0.1, k / order]
            for j in range(order + 1 - k)
            for i in range(order + 1 - k - j)
        ]

    points += [[0, 0, 1]]

    def coord_to_vertex(x, y, z):
        return (
            z * (3 * order**2 - 3 * order * z + 12 * order + z**2 - 6 * z + 11) // 6
            + y * (2 * (order - z) + 3 - y) // 2
            + x
        )

    # Define a cell using DOLFINx ordering
    cell = [
        coord_to_vertex(x, y, z)
        for x, y, z in [(0, 0, 0), (order, 0, 0), (0, order, 0), (0, 0, order)]
    ]

    if order > 1:
        for i in range(1, order):
            cell.append(coord_to_vertex(0, order - i, i))
        for i in range(1, order):
            cell.append(coord_to_vertex(order - i, 0, i))
        for i in range(1, order):
            cell.append(coord_to_vertex(order - i, i, 0))
        for i in range(1, order):
            cell.append(coord_to_vertex(0, 0, i))
        for i in range(1, order):
            cell.append(coord_to_vertex(0, i, 0))
        for i in range(1, order):
            cell.append(coord_to_vertex(i, 0, 0))

        for j in range(1, order):
            for i in range(1, order - j):
                cell.append(coord_to_vertex(order - i - j, i, j))
        for j in range(1, order):
            for i in range(1, order - j):
                cell.append(coord_to_vertex(0, i, j))
        for j in range(1, order):
            for i in range(1, order - j):
                cell.append(coord_to_vertex(i, 0, j))
        for j in range(1, order):
            for i in range(1, order - j):
                cell.append(coord_to_vertex(i, j, 0))

        for k in range(1, order):
            for j in range(1, order - k):
                for i in range(1, order - j - k):
                    cell.append(coord_to_vertex(i, j, k))

    domain = ufl.Mesh(
        element(
            "Lagrange",
            "tetrahedron",
            order,
            lagrange_variant=basix.LagrangeVariant.equispaced,
            shape=(3,),
            dtype=dtype,
        )
    )
    check_cell_volume(points, cell, domain, 1 / 6, dtype=dtype)


@pytest.mark.skip_in_parallel
@pytest.mark.parametrize("order", [1, 2, 3, 4])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_quadrilateral_mesh(order, dtype):
    random.seed(13)

    points = []
    points += [[i / order, 0] for i in range(order + 1)]
    for j in range(1, order):
        points += [[i / order + 0.1, j / order] for i in range(order + 1)]
    points += [[j / order, 1] for j in range(order + 1)]

    def coord_to_vertex(x, y):
        return (order + 1) * y + x

    # Define a cell using DOLFINx ordering
    cell = [coord_to_vertex(i, j) for i, j in [(0, 0), (order, 0), (0, order), (order, order)]]
    if order > 1:
        for i in range(1, order):
            cell.append(coord_to_vertex(i, 0))
        for i in range(1, order):
            cell.append(coord_to_vertex(0, i))
        for i in range(1, order):
            cell.append(coord_to_vertex(order, i))
        for i in range(1, order):
            cell.append(coord_to_vertex(i, order))

        for j in range(1, order):
            for i in range(1, order):
                cell.append(coord_to_vertex(i, j))

    domain = ufl.Mesh(
        element(
            "Q",
            "quadrilateral",
            order,
            lagrange_variant=basix.LagrangeVariant.equispaced,
            shape=(2,),
            dtype=dtype,
        )
    )
    check_cell_volume(points, cell, domain, 1, dtype=dtype)


@pytest.mark.skip_in_parallel
@pytest.mark.parametrize("order", [1, 2, 3, 4])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_hexahedron_mesh(order, dtype):
    random.seed(13)
    points = []
    points += [[i / order, j / order, 0] for j in range(order + 1) for i in range(order + 1)]
    for k in range(1, order):
        points += [
            [i / order, j / order + 0.1, k / order]
            for j in range(order + 1)
            for i in range(order + 1)
        ]

    points += [[i / order, j / order, 1] for j in range(order + 1) for i in range(order + 1)]

    def coord_to_vertex(x, y, z):
        return (order + 1) ** 2 * z + (order + 1) * y + x

    # Define a cell using DOLFINx ordering
    cell = [
        coord_to_vertex(x, y, z)
        for x, y, z in [
            (0, 0, 0),
            (order, 0, 0),
            (0, order, 0),
            (order, order, 0),
            (0, 0, order),
            (order, 0, order),
            (0, order, order),
            (order, order, order),
        ]
    ]

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

    domain = ufl.Mesh(
        element(
            "Q",
            "hexahedron",
            order,
            lagrange_variant=basix.LagrangeVariant.equispaced,
            shape=(3,),
            dtype=dtype,
        )
    )
    check_cell_volume(points, cell, domain, 1, dtype=dtype)


@pytest.mark.skip_in_parallel
@pytest.mark.parametrize("order", range(1, 5))
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_triangle_mesh_vtk(order, dtype):
    points = []
    points += [[i / order, 0] for i in range(order + 1)]
    for j in range(1, order):
        points += [[i / order + 0.1, j / order] for i in range(order + 1 - j)]
    points += [[0, 1]]

    def coord_to_vertex(x, y):
        return y * (2 * order + 3 - y) // 2 + x

    # Make the cell, following
    # https://blog.kitware.com/modeling-arbitrary-order-lagrange-finite-elements-in-the-visualization-toolkit/
    cell = [coord_to_vertex(i, j) for i, j in [(0, 0), (order, 0), (0, order)]]
    if order > 1:
        for i in range(1, order):
            cell.append(coord_to_vertex(i, 0))
        for i in range(1, order):
            cell.append(coord_to_vertex(order - i, i))
        for i in range(1, order):
            cell.append(coord_to_vertex(0, order - i))

    if order == 3:
        cell.append(coord_to_vertex(1, 1))
    elif order > 3:
        cell.append(coord_to_vertex(1, 1))
        cell.append(coord_to_vertex(order - 2, 1))
        cell.append(coord_to_vertex(1, order - 2))
        if order > 4:
            raise NotImplementedError

    cell = np.array(cell)[perm_vtk(CellType.triangle, len(cell))]
    domain = ufl.Mesh(
        element(
            "Lagrange",
            "triangle",
            order,
            lagrange_variant=basix.LagrangeVariant.equispaced,
            shape=(2,),
            dtype=dtype,
        )
    )
    check_cell_volume(points, cell, domain, 0.5, dtype=dtype)


@pytest.mark.skip_in_parallel
@pytest.mark.parametrize("order", range(1, 5))
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_tetrahedron_mesh_vtk(order, dtype):
    if order > 3:
        pytest.xfail("VTK permutation for order > 3 tetrahedra not implemented in DOLFINx.")
    points = []
    points += [[i / order, j / order, 0] for j in range(order + 1) for i in range(order + 1 - j)]
    for k in range(1, order):
        points += [
            [i / order, j / order + 0.1, k / order]
            for j in range(order + 1 - k)
            for i in range(order + 1 - k - j)
        ]

    points += [[0, 0, 1]]

    def coord_to_vertex(x, y, z):
        return (
            z * (3 * order**2 - 3 * order * z + 12 * order + z**2 - 6 * z + 11) // 6
            + y * (2 * (order - z) + 3 - y) // 2
            + x
        )

    # Make the cell, following
    # https://blog.kitware.com/modeling-arbitrary-order-lagrange-finite-elements-in-the-visualization-toolkit/
    cell = [
        coord_to_vertex(x, y, z)
        for x, y, z in [(0, 0, 0), (order, 0, 0), (0, order, 0), (0, 0, order)]
    ]

    if order > 1:
        for i in range(1, order):
            cell.append(coord_to_vertex(i, 0, 0))
        for i in range(1, order):
            cell.append(coord_to_vertex(order - i, i, 0))
        for i in range(1, order):
            cell.append(coord_to_vertex(0, order - i, 0))
        for i in range(1, order):
            cell.append(coord_to_vertex(0, 0, i))
        for i in range(1, order):
            cell.append(coord_to_vertex(order - i, 0, i))
        for i in range(1, order):
            cell.append(coord_to_vertex(0, order - i, i))

        if order == 3:
            # The ordering of faces does not match documentation. See
            # https://gitlab.kitware.com/vtk/vtk/uploads/a0dc0173a41d3cf6b03a9266c0e23688/image.png
            cell.append(coord_to_vertex(1, 0, 1))
            cell.append(coord_to_vertex(1, 1, 1))
            cell.append(coord_to_vertex(0, 1, 1))
            cell.append(coord_to_vertex(1, 1, 0))
        elif order == 4:
            # The ordering of faces does not match documentation.
            # See https://gitlab.kitware.com/vtk/vtk/uploads/a0dc0173a41d3cf6b03a9266c0e23688/image.png
            cell.append(coord_to_vertex(1, 0, 1))
            cell.append(coord_to_vertex(2, 0, 1))
            cell.append(coord_to_vertex(1, 0, 2))

            cell.append(coord_to_vertex(1, 2, 1))
            cell.append(coord_to_vertex(1, 1, 2))
            cell.append(coord_to_vertex(2, 1, 1))

            cell.append(coord_to_vertex(0, 1, 1))
            cell.append(coord_to_vertex(0, 1, 2))
            cell.append(coord_to_vertex(0, 2, 1))

            cell.append(coord_to_vertex(1, 1, 0))
            cell.append(coord_to_vertex(1, 2, 0))
            cell.append(coord_to_vertex(2, 1, 0))

            cell.append(coord_to_vertex(1, 1, 1))

        elif order > 4:
            raise NotImplementedError
        if False:
            for j in range(1, order):
                for i in range(1, order - j):
                    cell.append(coord_to_vertex(i, 0, j))
            for j in range(1, order):
                for i in range(1, order - j):
                    cell.append(coord_to_vertex(0, i, j))
            for j in range(1, order):
                for i in range(1, order - j):
                    cell.append(coord_to_vertex(i, j, 0))
            for j in range(1, order):
                for i in range(1, order - j):
                    cell.append(coord_to_vertex(order - i - j, i, j))

            for k in range(1, order):
                for j in range(1, order - k):
                    for i in range(1, order - j - k):
                        cell.append(coord_to_vertex(i, j, k))

    cell = np.array(cell)[perm_vtk(CellType.tetrahedron, len(cell))]
    domain = ufl.Mesh(
        element(
            "Lagrange",
            "tetrahedron",
            order,
            lagrange_variant=basix.LagrangeVariant.equispaced,
            shape=(3,),
            dtype=dtype,
        )
    )
    check_cell_volume(points, cell, domain, 1 / 6, dtype=dtype)


@pytest.mark.skip_in_parallel
@pytest.mark.parametrize("order", [1, 2, 3, 4])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_quadrilateral_mesh_vtk(order, dtype):
    random.seed(13)

    points = []
    points += [[i / order, 0] for i in range(order + 1)]
    for j in range(1, order):
        points += [[i / order + 0.1, j / order] for i in range(order + 1)]
    points += [[j / order, 1] for j in range(order + 1)]

    def coord_to_vertex(x, y):
        return (order + 1) * y + x

    # Make the cell, following
    # https://blog.kitware.com/modeling-arbitrary-order-lagrange-finite-elements-in-the-visualization-toolkit/
    cell = [coord_to_vertex(i, j) for i, j in [(0, 0), (order, 0), (order, order), (0, order)]]
    if order > 1:
        for i in range(1, order):
            cell.append(coord_to_vertex(i, 0))
        for i in range(1, order):
            cell.append(coord_to_vertex(order, i))
        for i in range(1, order):
            cell.append(coord_to_vertex(i, order))
        for i in range(1, order):
            cell.append(coord_to_vertex(0, i))

        for j in range(1, order):
            for i in range(1, order):
                cell.append(coord_to_vertex(i, j))

    cell = np.array(cell)[perm_vtk(CellType.quadrilateral, len(cell))]
    domain = ufl.Mesh(
        element(
            "Q",
            "quadrilateral",
            order,
            lagrange_variant=basix.LagrangeVariant.equispaced,
            shape=(2,),
            dtype=dtype,
        )
    )
    check_cell_volume(points, cell, domain, 1, dtype=dtype)


@pytest.mark.skip_in_parallel
@pytest.mark.parametrize("order", [1, 2, 3, 4])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_hexahedron_mesh_vtk(order, dtype):
    if order > 2:
        pytest.xfail("VTK permutation for order > 2 hexahedra not implemented in DOLFINx.")
    random.seed(13)

    points = []
    points += [[i / order, j / order, 0] for j in range(order + 1) for i in range(order + 1)]
    for k in range(1, order):
        points += [
            [i / order, j / order + 0.1, k / order]
            for j in range(order + 1)
            for i in range(order + 1)
        ]

    points += [[i / order, j / order, 1] for j in range(order + 1) for i in range(order + 1)]

    def coord_to_vertex(x, y, z):
        return (order + 1) ** 2 * z + (order + 1) * y + x

    # Make the cell, following
    # https://blog.kitware.com/modeling-arbitrary-order-lagrange-finite-elements-in-the-visualization-toolkit/
    cell = [
        coord_to_vertex(x, y, z)
        for x, y, z in [
            (0, 0, 0),
            (order, 0, 0),
            (order, order, 0),
            (0, order, 0),
            (0, 0, order),
            (order, 0, order),
            (order, order, order),
            (0, order, order),
        ]
    ]

    if order > 1:
        for i in range(1, order):
            cell.append(coord_to_vertex(i, 0, 0))
        for i in range(1, order):
            cell.append(coord_to_vertex(order, i, 0))
        for i in range(1, order):
            cell.append(coord_to_vertex(i, order, 0))
        for i in range(1, order):
            cell.append(coord_to_vertex(0, i, 0))
        for i in range(1, order):
            cell.append(coord_to_vertex(i, 0, order))
        for i in range(1, order):
            cell.append(coord_to_vertex(order, i, order))
        for i in range(1, order):
            cell.append(coord_to_vertex(i, order, order))
        for i in range(1, order):
            cell.append(coord_to_vertex(0, i, order))
        for i in range(1, order):
            cell.append(coord_to_vertex(0, 0, i))
        for i in range(1, order):
            cell.append(coord_to_vertex(order, 0, i))
        for i in range(1, order):
            cell.append(coord_to_vertex(order, order, i))
        for i in range(1, order):
            cell.append(coord_to_vertex(0, order, i))

        # The ordering of faces does not match documentation. See
        # https://gitlab.kitware.com/vtk/vtk/uploads/a0dc0173a41d3cf6b03a9266c0e23688/image.png
        # The edge flip in this like however has been fixed in VTK so we
        # follow the main documentation link for edges
        for j in range(1, order):
            for i in range(1, order):
                cell.append(coord_to_vertex(0, i, j))
        for j in range(1, order):
            for i in range(1, order):
                cell.append(coord_to_vertex(order, i, j))
        for j in range(1, order):
            for i in range(1, order):
                cell.append(coord_to_vertex(i, 0, j))
        for j in range(1, order):
            for i in range(1, order):
                cell.append(coord_to_vertex(i, order, j))
        for j in range(1, order):
            for i in range(1, order):
                cell.append(coord_to_vertex(i, j, 0))
        for j in range(1, order):
            for i in range(1, order):
                cell.append(coord_to_vertex(i, j, order))

        for k in range(1, order):
            for j in range(1, order):
                for i in range(1, order):
                    cell.append(coord_to_vertex(i, j, k))

    cell = np.array(cell)[perm_vtk(CellType.hexahedron, len(cell))]
    domain = ufl.Mesh(
        element(
            "Q",
            "hexahedron",
            order,
            lagrange_variant=basix.LagrangeVariant.equispaced,
            shape=(3,),
            dtype=dtype,
        )
    )
    check_cell_volume(points, cell, domain, 1, dtype=dtype)


@pytest.mark.skip_in_parallel
@pytest.mark.parametrize(
    "vtk,dolfin,cell_type",
    [
        ([0, 1, 2, 3, 4, 5], [0, 1, 2, 4, 5, 3], CellType.triangle),
        ([0, 1, 2, 3], [0, 1, 3, 2], CellType.quadrilateral),
        ([0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 3, 2, 4, 5, 7, 6], CellType.hexahedron),
    ],
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_map_vtk_to_dolfin(vtk, dolfin, cell_type, dtype):
    p = perm_vtk(cell_type, len(vtk))
    cell_p = np.array(vtk)[p]
    assert (cell_p == dolfin).all()

    p = np.argsort(perm_vtk(cell_type, len(vtk)))
    cell_p = np.array(dolfin)[p]
    assert (cell_p == vtk).all()


@pytest.mark.skipif(default_real_type != np.float64, reason="float32 not supported yet")
@pytest.mark.skip_in_parallel
@pytest.mark.parametrize("dtype", [np.float64])
def test_xdmf_input_tri(datadir, dtype):
    with XDMFFile(
        MPI.COMM_WORLD, Path(datadir, "mesh.xdmf"), "r", encoding=XDMFFile.Encoding.ASCII
    ) as xdmf:
        mesh = xdmf.read_mesh(name="Grid")
    surface = assemble_scalar(form(1 * dx(mesh), dtype=dtype))
    assert mesh.comm.allreduce(surface, op=MPI.SUM) == pytest.approx(4 * np.pi, rel=1e-4)


@pytest.mark.skip_in_parallel
@pytest.mark.parametrize("order", range(1, 4))
@pytest.mark.parametrize("cell_type", [CellType.triangle, CellType.quadrilateral])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_gmsh_input_2d(order, cell_type, dtype):
    try:
        import gmsh
    except ImportError:
        pytest.skip()
    res = 0.2
    gmsh.initialize()
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", res)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", res)

    if cell_type == CellType.quadrilateral:
        gmsh.option.setNumber("Mesh.Algorithm", 2)
        # Force mesh to have no triangles
        gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 3)

    gmsh.model.occ.addSphere(0, 0, 0, 1, tag=1)
    gmsh.model.occ.synchronize()

    gmsh.model.mesh.generate(2)
    if cell_type == CellType.quadrilateral:
        gmsh.model.mesh.recombine()
    gmsh.model.mesh.setOrder(order)
    idx, points, _ = gmsh.model.mesh.getNodes()
    points = points.reshape(-1, 3)
    idx -= 1
    srt = np.argsort(idx)
    assert np.all(idx[srt] == np.arange(len(idx)))
    x = points[srt]

    element_types, element_tags, node_tags = gmsh.model.mesh.getElements(dim=2)
    (
        name,
        dim,
        order,
        num_nodes,
        local_coords,
        num_first_order_nodes,
    ) = gmsh.model.mesh.getElementProperties(element_types[0])

    cells = node_tags[0].reshape(-1, num_nodes) - 1
    if cell_type == CellType.triangle:
        gmsh_cell_id = gmsh.model.mesh.getElementType("triangle", order)
    elif cell_type == CellType.quadrilateral:
        gmsh_cell_id = gmsh.model.mesh.getElementType("quadrangle", order)
    gmsh.finalize()

    cells = cells[:, cell_perm_array(cell_type, cells.shape[1])].copy()
    x = x.astype(dtype)
    mesh = create_mesh(MPI.COMM_WORLD, cells, x, ufl_mesh(gmsh_cell_id, x.shape[1], dtype=dtype))
    surface = assemble_scalar(form(1 * dx(mesh), dtype=dtype))

    assert mesh.comm.allreduce(surface, op=MPI.SUM) == pytest.approx(
        4 * np.pi, rel=10 ** (-1 - order)
    )


@pytest.mark.skip_in_parallel
@pytest.mark.parametrize("order", range(1, 4))
@pytest.mark.parametrize("cell_type", [CellType.tetrahedron, CellType.hexahedron])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_gmsh_input_3d(order, cell_type, dtype):
    try:
        import gmsh
    except ImportError:
        pytest.skip()
    if cell_type == CellType.hexahedron and order > 2:
        pytest.xfail("Gmsh permutation for order > 2 hexahedra not implemented in DOLFINx.")

    res = 0.2

    gmsh.initialize()
    if cell_type == CellType.hexahedron:
        gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
        gmsh.option.setNumber("Mesh.RecombineAll", 2)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", res)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", res)

    circle = gmsh.model.occ.addDisk(0, 0, 0, 1, 1)

    if cell_type == CellType.hexahedron:
        gmsh.model.occ.extrude([(2, circle)], 0, 0, 1, numElements=[5], recombine=True)
    else:
        gmsh.model.occ.extrude([(2, circle)], 0, 0, 1, numElements=[5])
    gmsh.model.occ.synchronize()

    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.setOrder(order)

    idx, points, _ = gmsh.model.mesh.getNodes()
    points = points.reshape(-1, 3)
    idx -= 1
    srt = np.argsort(idx)
    assert np.all(idx[srt] == np.arange(len(idx)))
    x = points[srt]

    element_types, element_tags, node_tags = gmsh.model.mesh.getElements(dim=3)
    (
        name,
        dim,
        order,
        num_nodes,
        local_coords,
        num_first_order_nodes,
    ) = gmsh.model.mesh.getElementProperties(element_types[0])

    cells = node_tags[0].reshape(-1, num_nodes) - 1
    if cell_type == CellType.tetrahedron:
        gmsh_cell_id = MPI.COMM_WORLD.bcast(
            gmsh.model.mesh.getElementType("tetrahedron", order), root=0
        )
    elif cell_type == CellType.hexahedron:
        gmsh_cell_id = MPI.COMM_WORLD.bcast(
            gmsh.model.mesh.getElementType("hexahedron", order), root=0
        )
    gmsh.finalize()

    # Permute the mesh topology from Gmsh ordering to DOLFINx ordering
    domain = ufl_mesh(gmsh_cell_id, 3, dtype=dtype)
    cells = cells[:, cell_perm_array(cell_type, cells.shape[1])].copy()

    x = x.astype(dtype)
    mesh = create_mesh(MPI.COMM_WORLD, cells, x, domain)
    volume = assemble_scalar(form(1 * dx(mesh), dtype=dtype))
    assert mesh.comm.allreduce(volume, op=MPI.SUM) == pytest.approx(np.pi, rel=10 ** (-1 - order))


@pytest.mark.skip_in_parallel
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_quadrilateral_cell_order_3(dtype):
    points = [
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [1 / 3, 2 / 9],
        [2 / 3, 2 / 9],
        [0.0, 1 / 3],
        [0.0, 2 / 3],
        [1.0, 1 / 3],
        [1.0, 2 / 3],
        [1 / 3, 1.0],
        [2 / 3, 1.0],
        [1 / 3, 13 / 27],
        [2 / 3, 13 / 27],
        [1 / 3, 20 / 27],
        [2 / 3, 20 / 27],
    ]
    cell = list(range(16))
    domain = ufl.Mesh(
        element(
            "Q",
            "quadrilateral",
            3,
            lagrange_variant=basix.LagrangeVariant.equispaced,
            shape=(2,),
            dtype=dtype,
        )
    )
    check_cell_volume(points, cell, domain, 5 / 6, dtype=dtype)


@pytest.mark.parametrize("order", range(1, 11))
def test_vtk_perm_tetrahedron(order):
    size = (order + 1) * (order + 2) * (order + 3) // 6
    p = perm_vtk(CellType.tetrahedron, size)

    if order == 1:
        q = [0, 1, 2, 3]
    if order == 2:
        q = [0, 1, 2, 3, 9, 6, 8, 7, 5, 4]
    if order == 3:
        q = [0, 1, 2, 3, 14, 15, 8, 9, 13, 12, 10, 11, 6, 7, 4, 5, 18, 16, 17, 19]
    if order == 4:
        q = [
            0,
            1,
            2,
            3,
            19,
            20,
            21,
            10,
            11,
            12,
            18,
            17,
            16,
            13,
            14,
            15,
            7,
            8,
            9,
            4,
            5,
            6,
            28,
            29,
            30,
            23,
            24,
            22,
            25,
            27,
            26,
            31,
            33,
            32,
            34,
        ]
    if order == 5:
        q = [
            0,
            1,
            2,
            3,
            24,
            25,
            26,
            27,
            12,
            13,
            14,
            15,
            23,
            22,
            21,
            20,
            16,
            17,
            18,
            19,
            8,
            9,
            10,
            11,
            4,
            5,
            6,
            7,
            40,
            42,
            45,
            41,
            44,
            43,
            30,
            33,
            28,
            32,
            31,
            29,
            34,
            39,
            36,
            37,
            38,
            35,
            46,
            51,
            48,
            49,
            50,
            47,
            52,
            53,
            54,
            55,
        ]
    if order == 6:
        q = [
            0,
            1,
            2,
            3,
            29,
            30,
            31,
            32,
            33,
            14,
            15,
            16,
            17,
            18,
            28,
            27,
            26,
            25,
            24,
            19,
            20,
            21,
            22,
            23,
            9,
            10,
            11,
            12,
            13,
            4,
            5,
            6,
            7,
            8,
            54,
            57,
            63,
            55,
            56,
            60,
            62,
            61,
            58,
            59,
            37,
            43,
            34,
            40,
            42,
            41,
            38,
            35,
            36,
            39,
            44,
            53,
            47,
            48,
            51,
            52,
            50,
            46,
            45,
            49,
            64,
            73,
            67,
            68,
            71,
            72,
            70,
            66,
            65,
            69,
            74,
            76,
            79,
            83,
            75,
            78,
            77,
            80,
            81,
            82,
        ]
    if order == 7:
        q = [
            0,
            1,
            2,
            3,
            34,
            35,
            36,
            37,
            38,
            39,
            16,
            17,
            18,
            19,
            20,
            21,
            33,
            32,
            31,
            30,
            29,
            28,
            22,
            23,
            24,
            25,
            26,
            27,
            10,
            11,
            12,
            13,
            14,
            15,
            4,
            5,
            6,
            7,
            8,
            9,
            70,
            74,
            84,
            71,
            72,
            73,
            78,
            81,
            83,
            82,
            79,
            75,
            76,
            77,
            80,
            44,
            54,
            40,
            48,
            51,
            53,
            52,
            49,
            45,
            41,
            42,
            43,
            47,
            50,
            46,
            55,
            69,
            59,
            60,
            64,
            67,
            68,
            66,
            63,
            58,
            57,
            56,
            61,
            65,
            62,
            85,
            99,
            89,
            90,
            94,
            97,
            98,
            96,
            93,
            88,
            87,
            86,
            91,
            95,
            92,
            100,
            103,
            109,
            119,
            101,
            102,
            106,
            108,
            107,
            104,
            110,
            116,
            112,
            117,
            115,
            118,
            111,
            114,
            113,
            105,
        ]
    if order == 8:
        q = [
            0,
            1,
            2,
            3,
            39,
            40,
            41,
            42,
            43,
            44,
            45,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            38,
            37,
            36,
            35,
            34,
            33,
            32,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            88,
            93,
            108,
            89,
            90,
            91,
            92,
            98,
            102,
            105,
            107,
            106,
            103,
            99,
            94,
            95,
            97,
            104,
            96,
            101,
            100,
            51,
            66,
            46,
            56,
            60,
            63,
            65,
            64,
            61,
            57,
            52,
            47,
            48,
            49,
            50,
            55,
            62,
            53,
            59,
            58,
            54,
            67,
            87,
            72,
            73,
            78,
            82,
            85,
            86,
            84,
            81,
            77,
            71,
            70,
            69,
            68,
            74,
            83,
            76,
            79,
            80,
            75,
            109,
            129,
            114,
            115,
            120,
            124,
            127,
            128,
            126,
            123,
            119,
            113,
            112,
            111,
            110,
            116,
            125,
            118,
            121,
            122,
            117,
            130,
            134,
            144,
            164,
            131,
            132,
            133,
            138,
            141,
            143,
            142,
            139,
            135,
            145,
            155,
            161,
            148,
            157,
            162,
            154,
            160,
            163,
            146,
            147,
            156,
            153,
            159,
            151,
            149,
            158,
            152,
            136,
            140,
            137,
            150,
        ]
    if order == 9:
        q = [
            0,
            1,
            2,
            3,
            44,
            45,
            46,
            47,
            48,
            49,
            50,
            51,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            43,
            42,
            41,
            40,
            39,
            38,
            37,
            36,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            35,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            108,
            114,
            135,
            109,
            110,
            111,
            112,
            113,
            120,
            125,
            129,
            132,
            134,
            133,
            130,
            126,
            121,
            115,
            116,
            119,
            131,
            117,
            118,
            124,
            128,
            127,
            122,
            123,
            58,
            79,
            52,
            64,
            69,
            73,
            76,
            78,
            77,
            74,
            70,
            65,
            59,
            53,
            54,
            55,
            56,
            57,
            63,
            75,
            60,
            68,
            72,
            71,
            66,
            61,
            62,
            67,
            80,
            107,
            86,
            87,
            93,
            98,
            102,
            105,
            106,
            104,
            101,
            97,
            92,
            85,
            84,
            83,
            82,
            81,
            88,
            103,
            91,
            94,
            99,
            100,
            96,
            90,
            89,
            95,
            136,
            163,
            142,
            143,
            149,
            154,
            158,
            161,
            162,
            160,
            157,
            153,
            148,
            141,
            140,
            139,
            138,
            137,
            144,
            159,
            147,
            150,
            155,
            156,
            152,
            146,
            145,
            151,
            164,
            169,
            184,
            219,
            165,
            166,
            167,
            168,
            174,
            178,
            181,
            183,
            182,
            179,
            175,
            170,
            185,
            200,
            210,
            216,
            189,
            203,
            212,
            217,
            199,
            209,
            215,
            218,
            186,
            188,
            211,
            187,
            202,
            201,
            198,
            214,
            193,
            208,
            206,
            196,
            190,
            213,
            197,
            204,
            207,
            194,
            171,
            180,
            173,
            176,
            177,
            172,
            191,
            192,
            195,
            205,
        ]
    if order == 10:
        q = [
            0,
            1,
            2,
            3,
            49,
            50,
            51,
            52,
            53,
            54,
            55,
            56,
            57,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            48,
            47,
            46,
            45,
            44,
            43,
            42,
            41,
            40,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            130,
            137,
            165,
            131,
            132,
            133,
            134,
            135,
            136,
            144,
            150,
            155,
            159,
            162,
            164,
            163,
            160,
            156,
            151,
            145,
            138,
            139,
            143,
            161,
            140,
            141,
            142,
            149,
            154,
            158,
            157,
            152,
            146,
            147,
            148,
            153,
            65,
            93,
            58,
            72,
            78,
            83,
            87,
            90,
            92,
            91,
            88,
            84,
            79,
            73,
            66,
            59,
            60,
            61,
            62,
            63,
            64,
            71,
            89,
            67,
            77,
            82,
            86,
            85,
            80,
            74,
            68,
            69,
            70,
            76,
            81,
            75,
            94,
            129,
            101,
            102,
            109,
            115,
            120,
            124,
            127,
            128,
            126,
            123,
            119,
            114,
            108,
            100,
            99,
            98,
            97,
            96,
            95,
            103,
            125,
            107,
            110,
            116,
            121,
            122,
            118,
            113,
            106,
            105,
            104,
            111,
            117,
            112,
            166,
            201,
            173,
            174,
            181,
            187,
            192,
            196,
            199,
            200,
            198,
            195,
            191,
            186,
            180,
            172,
            171,
            170,
            169,
            168,
            167,
            175,
            197,
            179,
            182,
            188,
            193,
            194,
            190,
            185,
            178,
            177,
            176,
            183,
            189,
            184,
            202,
            208,
            229,
            285,
            203,
            204,
            205,
            206,
            207,
            214,
            219,
            223,
            226,
            228,
            227,
            224,
            220,
            215,
            209,
            230,
            251,
            266,
            276,
            282,
            235,
            255,
            269,
            278,
            283,
            250,
            265,
            275,
            281,
            284,
            231,
            234,
            277,
            232,
            233,
            254,
            268,
            267,
            252,
            253,
            249,
            280,
            240,
            264,
            274,
            272,
            259,
            244,
            247,
            262,
            236,
            279,
            248,
            256,
            270,
            273,
            263,
            245,
            241,
            260,
            210,
            225,
            213,
            216,
            221,
            222,
            218,
            212,
            211,
            217,
            237,
            239,
            246,
            271,
            238,
            243,
            242,
            257,
            258,
            261,
        ]

    pt = [0 for i in p]
    for i, j in enumerate(p):
        pt[j] = i
    print(" ".join([f"{i}" for i in pt]))
    print(" ".join([f"{i}" for i in q]))

    for i, j in enumerate(p):
        assert q[j] == i


@pytest.mark.parametrize("order", range(1, 7))
def test_vtk_perm_hexahedron(order):
    size = (order + 1) ** 3
    p = perm_vtk(CellType.hexahedron, size)

    if order == 1:
        q = [0, 1, 3, 2, 4, 5, 7, 6]
    if order == 2:
        q = [
            0,
            1,
            3,
            2,
            4,
            5,
            7,
            6,
            8,
            11,
            13,
            9,
            16,
            18,
            19,
            17,
            10,
            12,
            15,
            14,
            22,
            23,
            21,
            24,
            20,
            25,
            26,
        ]
    if order == 3:
        q = [
            0,
            1,
            3,
            2,
            4,
            5,
            7,
            6,
            8,
            9,
            14,
            15,
            18,
            19,
            10,
            11,
            24,
            25,
            28,
            29,
            30,
            31,
            26,
            27,
            12,
            13,
            16,
            17,
            22,
            23,
            20,
            21,
            40,
            41,
            42,
            43,
            44,
            45,
            46,
            47,
            36,
            37,
            38,
            39,
            48,
            49,
            50,
            51,
            32,
            33,
            34,
            35,
            52,
            53,
            54,
            55,
            56,
            57,
            58,
            59,
            60,
            61,
            62,
            63,
        ]
    if order == 4:
        q = [
            0,
            1,
            3,
            2,
            4,
            5,
            7,
            6,
            8,
            9,
            10,
            17,
            18,
            19,
            23,
            24,
            25,
            11,
            12,
            13,
            32,
            33,
            34,
            38,
            39,
            40,
            41,
            42,
            43,
            35,
            36,
            37,
            14,
            15,
            16,
            20,
            21,
            22,
            29,
            30,
            31,
            26,
            27,
            28,
            62,
            63,
            64,
            65,
            66,
            67,
            68,
            69,
            70,
            71,
            72,
            73,
            74,
            75,
            76,
            77,
            78,
            79,
            53,
            54,
            55,
            56,
            57,
            58,
            59,
            60,
            61,
            80,
            81,
            82,
            83,
            84,
            85,
            86,
            87,
            88,
            44,
            45,
            46,
            47,
            48,
            49,
            50,
            51,
            52,
            89,
            90,
            91,
            92,
            93,
            94,
            95,
            96,
            97,
            98,
            99,
            100,
            101,
            102,
            103,
            104,
            105,
            106,
            107,
            108,
            109,
            110,
            111,
            112,
            113,
            114,
            115,
            116,
            117,
            118,
            119,
            120,
            121,
            122,
            123,
            124,
        ]
    if order == 5:
        q = [
            0,
            1,
            3,
            2,
            4,
            5,
            7,
            6,
            8,
            9,
            10,
            11,
            20,
            21,
            22,
            23,
            28,
            29,
            30,
            31,
            12,
            13,
            14,
            15,
            40,
            41,
            42,
            43,
            48,
            49,
            50,
            51,
            52,
            53,
            54,
            55,
            44,
            45,
            46,
            47,
            16,
            17,
            18,
            19,
            24,
            25,
            26,
            27,
            36,
            37,
            38,
            39,
            32,
            33,
            34,
            35,
            88,
            89,
            90,
            91,
            92,
            93,
            94,
            95,
            96,
            97,
            98,
            99,
            100,
            101,
            102,
            103,
            104,
            105,
            106,
            107,
            108,
            109,
            110,
            111,
            112,
            113,
            114,
            115,
            116,
            117,
            118,
            119,
            72,
            73,
            74,
            75,
            76,
            77,
            78,
            79,
            80,
            81,
            82,
            83,
            84,
            85,
            86,
            87,
            120,
            121,
            122,
            123,
            124,
            125,
            126,
            127,
            128,
            129,
            130,
            131,
            132,
            133,
            134,
            135,
            56,
            57,
            58,
            59,
            60,
            61,
            62,
            63,
            64,
            65,
            66,
            67,
            68,
            69,
            70,
            71,
            136,
            137,
            138,
            139,
            140,
            141,
            142,
            143,
            144,
            145,
            146,
            147,
            148,
            149,
            150,
            151,
            152,
            153,
            154,
            155,
            156,
            157,
            158,
            159,
            160,
            161,
            162,
            163,
            164,
            165,
            166,
            167,
            168,
            169,
            170,
            171,
            172,
            173,
            174,
            175,
            176,
            177,
            178,
            179,
            180,
            181,
            182,
            183,
            184,
            185,
            186,
            187,
            188,
            189,
            190,
            191,
            192,
            193,
            194,
            195,
            196,
            197,
            198,
            199,
            200,
            201,
            202,
            203,
            204,
            205,
            206,
            207,
            208,
            209,
            210,
            211,
            212,
            213,
            214,
            215,
        ]
    if order == 6:
        q = [
            0,
            1,
            3,
            2,
            4,
            5,
            7,
            6,
            8,
            9,
            10,
            11,
            12,
            23,
            24,
            25,
            26,
            27,
            33,
            34,
            35,
            36,
            37,
            13,
            14,
            15,
            16,
            17,
            48,
            49,
            50,
            51,
            52,
            58,
            59,
            60,
            61,
            62,
            63,
            64,
            65,
            66,
            67,
            53,
            54,
            55,
            56,
            57,
            18,
            19,
            20,
            21,
            22,
            28,
            29,
            30,
            31,
            32,
            43,
            44,
            45,
            46,
            47,
            38,
            39,
            40,
            41,
            42,
            118,
            119,
            120,
            121,
            122,
            123,
            124,
            125,
            126,
            127,
            128,
            129,
            130,
            131,
            132,
            133,
            134,
            135,
            136,
            137,
            138,
            139,
            140,
            141,
            142,
            143,
            144,
            145,
            146,
            147,
            148,
            149,
            150,
            151,
            152,
            153,
            154,
            155,
            156,
            157,
            158,
            159,
            160,
            161,
            162,
            163,
            164,
            165,
            166,
            167,
            93,
            94,
            95,
            96,
            97,
            98,
            99,
            100,
            101,
            102,
            103,
            104,
            105,
            106,
            107,
            108,
            109,
            110,
            111,
            112,
            113,
            114,
            115,
            116,
            117,
            168,
            169,
            170,
            171,
            172,
            173,
            174,
            175,
            176,
            177,
            178,
            179,
            180,
            181,
            182,
            183,
            184,
            185,
            186,
            187,
            188,
            189,
            190,
            191,
            192,
            68,
            69,
            70,
            71,
            72,
            73,
            74,
            75,
            76,
            77,
            78,
            79,
            80,
            81,
            82,
            83,
            84,
            85,
            86,
            87,
            88,
            89,
            90,
            91,
            92,
            193,
            194,
            195,
            196,
            197,
            198,
            199,
            200,
            201,
            202,
            203,
            204,
            205,
            206,
            207,
            208,
            209,
            210,
            211,
            212,
            213,
            214,
            215,
            216,
            217,
            218,
            219,
            220,
            221,
            222,
            223,
            224,
            225,
            226,
            227,
            228,
            229,
            230,
            231,
            232,
            233,
            234,
            235,
            236,
            237,
            238,
            239,
            240,
            241,
            242,
            243,
            244,
            245,
            246,
            247,
            248,
            249,
            250,
            251,
            252,
            253,
            254,
            255,
            256,
            257,
            258,
            259,
            260,
            261,
            262,
            263,
            264,
            265,
            266,
            267,
            268,
            269,
            270,
            271,
            272,
            273,
            274,
            275,
            276,
            277,
            278,
            279,
            280,
            281,
            282,
            283,
            284,
            285,
            286,
            287,
            288,
            289,
            290,
            291,
            292,
            293,
            294,
            295,
            296,
            297,
            298,
            299,
            300,
            301,
            302,
            303,
            304,
            305,
            306,
            307,
            308,
            309,
            310,
            311,
            312,
            313,
            314,
            315,
            316,
            317,
            318,
            319,
            320,
            321,
            322,
            323,
            324,
            325,
            326,
            327,
            328,
            329,
            330,
            331,
            332,
            333,
            334,
            335,
            336,
            337,
            338,
            339,
            340,
            341,
            342,
        ]

    for i, j in enumerate(p):
        assert q[j] == i
