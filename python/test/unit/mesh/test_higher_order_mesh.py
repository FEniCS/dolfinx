# Copyright (C) 2019-2026 Jørgen Schartum Dokken and Matthew Scroggs
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit-tests for higher order meshes."""

import json
import random
from pathlib import Path

from mpi4py import MPI

import numpy as np
import pytest

import basix
import ufl
from basix.ufl import element
from dolfinx.cpp.io import perm_vtk
from dolfinx.fem import assemble_scalar, form, mixed_topology_form
from dolfinx.io import XDMFFile
from dolfinx.io.gmsh import model_to_mesh
from dolfinx.mesh import CellType, Mesh, create_mesh, create_submesh
from ufl import dx


@pytest.fixture(scope="module")
def vtk_permutations(datadir):
    """Reference VTK node orderings, keyed by cell type then by cell order.

    Each table maps a VTK node index to the corresponding DOLFINx node index,
    and is the expected inverse of `dolfinx.cpp.io.perm_vtk`. Taken from
    https://blog.kitware.com/modeling-arbitrary-order-lagrange-finite-elements-in-the-visualization-toolkit/
    """
    with open(Path(datadir, "vtk_permutations.json")) as file:
        return json.load(file)


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
        mesh = create_mesh(MPI.COMM_WORLD, [ordered_cell], domain, ordered_points)
        area = assemble_scalar(form(1 * dx(mesh), dtype=dtype))
        assert np.isclose(area, volume)


@pytest.mark.skip_in_parallel
@pytest.mark.parametrize("order", range(1, 5))
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
    mesh = create_mesh(MPI.COMM_WORLD, [cell], domain, points)
    for i in range(mesh.topology.dim):
        mesh.topology.create_entities(i)
    md = {"quadrature_degree": 10}
    measures = (ufl.ds(mesh, metadata=md), ufl.dx(mesh, metadata=md))
    dimensions = (mesh.topology.dim - 1, mesh.topology.dim)
    # Check that creating a submesh of single cell mesh, consisting of:
    # 1. The cell
    # 2. The facets of the cell
    # Gives the correct computation of: volume (case 1) or surface area (case 2)
    for dim, dC in zip(dimensions, measures, strict=True):
        # Integrate on original mesh
        value = assemble_scalar(form(1 * dC, dtype=dtype))
        num_local_entities = mesh.topology.index_map(dim).size_local
        submesh, _, _, _ = create_submesh(mesh, dim, np.arange(num_local_entities, dtype=np.int32))
        submesh_area = assemble_scalar(form(1 * ufl.dx(submesh, metadata=md), dtype=dtype))
        assert np.isclose(value, submesh_area)


@pytest.mark.skip_in_parallel
@pytest.mark.parametrize("order", range(1, 5))
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
def test_map_vtk_to_dolfin(vtk, dolfin, cell_type, dtype):
    p = perm_vtk(cell_type, len(vtk))
    cell_p = np.array(vtk)[p]
    assert (cell_p == dolfin).all()

    p = np.argsort(perm_vtk(cell_type, len(vtk)))
    cell_p = np.array(dolfin)[p]
    assert (cell_p == vtk).all()


@pytest.mark.skip_float32
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
@pytest.mark.parametrize("order", [1, 2])
def test_gmsh_mixed_mesh_2d(order, dtype):
    try:
        import gmsh
    except ImportError:
        pytest.skip()
    res = 0.2
    gmsh.initialize()
    model_name = f"mixed_2D_{dtype}_{order}"
    gmsh.model.add(model_name)
    gmsh.model.setCurrent(model_name)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", res)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", res)

    gmsh.option.setNumber("Mesh.Algorithm", 2)
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
    tag = gmsh.model.occ.addSphere(0, 0, 0, 1)
    gmsh.model.occ.synchronize()
    gmsh.model.addPhysicalGroup(2, [tag], tag=1)

    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.recombine()
    gmsh.model.mesh.setOrder(order)

    mesh_data = model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=3, dtype=dtype)
    gmsh.finalize()
    mesh = mesh_data.mesh
    if len(mesh.topology._cpp_object.cell_types) == 1:
        surface = assemble_scalar(form(1 * dx(mesh), dtype=dtype))
    else:
        Js = []
        for i, cell_type in enumerate(mesh._cpp_object.topology.cell_types):
            cell_name = cell_type.name
            cmap = mesh._cpp_object.geometry.cmaps[i]
            domain = ufl.Mesh(
                basix.ufl.element(
                    "Lagrange",
                    cell_name,
                    cmap.degree,
                    lagrange_variant=cmap.variant,
                    shape=(mesh.geometry.dim,),
                    dtype=dtype,
                )
            )
            mesh_i = Mesh(mesh._cpp_object, domain)
            Js += [1 * ufl.dx(mesh_i)]
        surface = assemble_scalar(mixed_topology_form(Js, dtype=dtype))

    assert mesh.comm.allreduce(surface, op=MPI.SUM) == pytest.approx(
        4 * np.pi, rel=10 ** (-1 - order)
    )


@pytest.mark.skip_in_parallel
@pytest.mark.parametrize("order", [1, 2, 3])
def test_gmsh_input_2d(order, dtype):
    try:
        import gmsh
    except ImportError:
        pytest.skip()
    res = 0.2
    gmsh.initialize()
    model_name = f"triangle_2D_{dtype}_{order}"
    gmsh.model.add(model_name)
    gmsh.model.setCurrent(model_name)

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", res)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", res)

    tag = gmsh.model.occ.addSphere(0, 0, 0, 1)
    gmsh.model.occ.synchronize()
    gmsh.model.addPhysicalGroup(2, [tag], tag=1)

    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(order)

    mesh = model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=3, dtype=dtype).mesh
    gmsh.finalize()
    surface = assemble_scalar(form(1 * dx(mesh), dtype=dtype))

    assert mesh.comm.allreduce(surface, op=MPI.SUM) == pytest.approx(
        4 * np.pi, rel=10 ** (-1 - order)
    )


@pytest.mark.skip_in_parallel
@pytest.mark.parametrize("order", [1, 2])
def test_gmsh_mixed_mesh_3d(order, dtype):
    try:
        import gmsh
    except ImportError:
        pytest.skip()
    gmsh.initialize()
    model_name = f"mixed_3D_{dtype}_{order}"
    gmsh.model.add(model_name)
    gmsh.model.setCurrent(model_name)

    res = 0.1
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", res)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", res)

    # Inner square core
    core = gmsh.model.occ.addRectangle(-0.5, -0.5, 0, 1, 1)
    # Outer disk
    disk = gmsh.model.occ.addDisk(0, 0, 0, 1, 1)

    # Fragment the disk with the core to create a separate inner square and outer ring
    out, _ = gmsh.model.occ.fragment([(2, disk)], [(2, core)])
    base_surfaces = [tag for dim, tag in out]

    # Extrude the base surfaces to create the bottom 3D volumes
    # (Recombine=True will turn a quad-meshed surface into Hexahedrons,
    # and a tri-meshed surface into Prisms)
    gmsh.model.occ.extrude(
        [(2, s) for s in base_surfaces], 0, 0, 1, numElements=[5], recombine=True
    )

    # Add a top cylinder for unstructured tetrahedral meshing
    gmsh.model.occ.addCylinder(0, 0, 1, 0, 0, 1, 1)

    # Fragment all volumes to ensure conformal interfaces (shared nodes at boundaries)
    vols = gmsh.model.occ.getEntities(3)
    gmsh.model.occ.fragment(vols, [])
    gmsh.model.occ.synchronize()

    # Find the square base surface (at z=0) and set it to recombine into quads
    for dim, tag in gmsh.model.occ.getEntities(2):
        com = gmsh.model.occ.getCenterOfMass(dim, tag)
        if abs(com[2]) < 1e-4:  # If it's on the bottom plane
            bb = gmsh.model.occ.getBoundingBox(dim, tag)
            width = bb[3] - bb[0]
            if abs(width - 1.0) < 1e-4:  # Identify the square core by its 1x1 dimension
                gmsh.model.mesh.setRecombine(2, tag)

    volumes = gmsh.model.getEntities(3)
    for phys_tag, (_, vol) in enumerate(volumes, start=1):
        gmsh.model.addPhysicalGroup(3, [vol], tag=phys_tag)

    # Generate the 3D mesh
    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.setOrder(order)

    mesh_data = model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=3, dtype=dtype)
    gmsh.finalize()

    mesh = mesh_data.mesh

    # Permute the mesh topology from Gmsh ordering to DOLFINx ordering
    cell_types = mesh._cpp_object.topology.cell_types
    if len(cell_types) == 1:
        volume = assemble_scalar(form(1 * dx(mesh), dtype=dtype))

    else:
        Js = []
        for i, cell_type in enumerate(cell_types):
            cell_name = cell_type.name
            cmap = mesh._cpp_object.geometry.cmaps[i]
            domain = ufl.Mesh(
                basix.ufl.element(
                    "Lagrange",
                    cell_name,
                    cmap.degree,
                    lagrange_variant=cmap.variant,
                    shape=(mesh.geometry.dim,),
                    dtype=dtype,
                )
            )
            mesh_i = Mesh(mesh._cpp_object, domain)
            Js += [1 * ufl.dx(mesh_i)]
        volume = assemble_scalar(mixed_topology_form(Js, dtype=dtype))

    assert mesh.comm.allreduce(volume, op=MPI.SUM) == pytest.approx(
        2 * np.pi, rel=10 ** (-1 - order)
    )


@pytest.mark.skip_in_parallel
@pytest.mark.parametrize("order", [1, 2, 3])
def test_gmsh_tetra(order, dtype):
    try:
        import gmsh
    except ImportError:
        pytest.skip()

    res = 0.2

    gmsh.initialize()
    model_name = f"tetra_{dtype}_{order}"
    gmsh.model.add(model_name)
    gmsh.model.setCurrent(model_name)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", res)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", res)

    circle = gmsh.model.occ.addDisk(0, 0, 0, 1, 1)
    gmsh.model.occ.extrude([(2, circle)], 0, 0, 1, numElements=[5])
    gmsh.model.occ.synchronize()

    # Tag only 3D volumes, and use unique physical tags
    for phys_tag, (_, vol) in enumerate(gmsh.model.getEntities(3), start=1):
        gmsh.model.addPhysicalGroup(3, [vol], tag=phys_tag)

    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.setOrder(order)
    mesh_data = model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=3, dtype=dtype)
    mesh = mesh_data.mesh
    gmsh.finalize()

    volume = assemble_scalar(form(1 * dx(mesh), dtype=dtype))
    assert mesh.comm.allreduce(volume, op=MPI.SUM) == pytest.approx(np.pi, rel=10 ** (-1 - order))


@pytest.mark.skip_in_parallel
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
def test_vtk_perm_tetrahedron(order, vtk_permutations):
    size = (order + 1) * (order + 2) * (order + 3) // 6
    p = perm_vtk(CellType.tetrahedron, size)
    q = vtk_permutations["tetrahedron"][str(order)]
    for i, j in enumerate(p):
        assert q[j] == i


@pytest.mark.parametrize("order", range(1, 7))
def test_vtk_perm_hexahedron(order, vtk_permutations):
    size = (order + 1) ** 3
    p = perm_vtk(CellType.hexahedron, size)
    q = vtk_permutations["hexahedron"][str(order)]
    for i, j in enumerate(p):
        assert q[j] == i
