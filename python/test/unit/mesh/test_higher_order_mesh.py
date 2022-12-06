# Copyright (C) 2019-2021 Jørgen Schartum Dokken and Matthew Scroggs
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
""" Unit-tests for higher order meshes """

import random
from pathlib import Path

import basix
import numpy as np
import pytest
import ufl
from basix.ufl_wrapper import create_vector_element
from dolfinx.cpp.io import perm_vtk
from dolfinx.fem import assemble_scalar, form
from dolfinx.io import XDMFFile
from dolfinx.io.gmshio import cell_perm_array, ufl_mesh
from dolfinx.mesh import CellType, create_mesh, create_submesh
from mpi4py import MPI
from ufl import dx


def check_cell_volume(points, cell, domain, volume):
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

        mesh = create_mesh(MPI.COMM_WORLD, [ordered_cell], ordered_points, domain)
        area = assemble_scalar(form(1 * dx(mesh)))
        assert np.isclose(area, volume)


@pytest.mark.skip_in_parallel
@pytest.mark.parametrize('order', range(1, 5))
def test_submesh(order):
    # Generate a single cell higher order mesh
    points = []
    points += [[i / order, j / order, 0] for j in range(order + 1) for i in range(order + 1 - j)]
    for k in range(1, order):
        points += [[i / order, j / order + 0.1, k / order]
                   for j in range(order + 1 - k) for i in range(order + 1 - k - j)]

    points += [[0, 0, 1]]

    def coord_to_vertex(x, y, z):
        return z * (
            3 * order ** 2 - 3 * order * z + 12 * order + z ** 2 - 6 * z + 11
        ) // 6 + y * (2 * (order - z) + 3 - y) // 2 + x

    # Define a cell using DOLFINx ordering
    cell = [coord_to_vertex(x, y, z) for x, y, z in [(0, 0, 0), (order, 0, 0), (0, order, 0), (0, 0, order)]]

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

    domain = ufl.Mesh(create_vector_element(
        "Lagrange", "tetrahedron", order, gdim=3, lagrange_variant=basix.LagrangeVariant.equispaced))

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
        value = assemble_scalar(form(1 * dC))
        num_local_entities = mesh.topology.index_map(dim).size_local
        submesh, _, _, _ = create_submesh(mesh, dim, np.arange(num_local_entities, dtype=np.int32))
        submesh_area = assemble_scalar(form(1 * ufl.dx(submesh, metadata=md)))
        assert np.isclose(value, submesh_area)


@pytest.mark.skip_in_parallel
@pytest.mark.parametrize('order', range(1, 5))
def test_triangle_mesh(order):
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

    domain = ufl.Mesh(create_vector_element(
        "Lagrange", "triangle", order, gdim=2, lagrange_variant=basix.LagrangeVariant.equispaced))

    check_cell_volume(points, cell, domain, 0.5)


@pytest.mark.skip_in_parallel
@pytest.mark.parametrize('order', range(1, 5))
def test_tetrahedron_mesh(order):
    points = []
    points += [[i / order, j / order, 0] for j in range(order + 1) for i in range(order + 1 - j)]
    for k in range(1, order):
        points += [[i / order, j / order + 0.1, k / order] for j in range(order + 1 - k)
                   for i in range(order + 1 - k - j)]

    points += [[0, 0, 1]]

    def coord_to_vertex(x, y, z):
        return z * (
            3 * order ** 2 - 3 * order * z + 12 * order + z ** 2 - 6 * z + 11
        ) // 6 + y * (2 * (order - z) + 3 - y) // 2 + x

    # Define a cell using DOLFINx ordering
    cell = [coord_to_vertex(x, y, z) for x, y, z in [(0, 0, 0), (order, 0, 0), (0, order, 0), (0, 0, order)]]

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

    domain = ufl.Mesh(create_vector_element(
        "Lagrange", "tetrahedron", order, gdim=3, lagrange_variant=basix.LagrangeVariant.equispaced))
    check_cell_volume(points, cell, domain, 1 / 6)


@pytest.mark.skip_in_parallel
@pytest.mark.parametrize('order', [1, 2, 3, 4])
def test_quadrilateral_mesh(order):
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

    domain = ufl.Mesh(create_vector_element(
        "Q", "quadrilateral", order, gdim=2, lagrange_variant=basix.LagrangeVariant.equispaced))
    check_cell_volume(points, cell, domain, 1)


@pytest.mark.skip_in_parallel
@pytest.mark.parametrize('order', [1, 2, 3, 4])
def test_hexahedron_mesh(order):
    random.seed(13)

    points = []
    points += [[i / order, j / order, 0] for j in range(order + 1) for i in range(order + 1)]
    for k in range(1, order):
        points += [[i / order, j / order + 0.1, k / order] for j in range(order + 1)
                   for i in range(order + 1)]

    points += [[i / order, j / order, 1] for j in range(order + 1) for i in range(order + 1)]

    def coord_to_vertex(x, y, z):
        return (order + 1) ** 2 * z + (order + 1) * y + x

    # Define a cell using DOLFINx ordering
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

    domain = ufl.Mesh(create_vector_element(
        "Q", "hexahedron", order, gdim=3, lagrange_variant=basix.LagrangeVariant.equispaced))
    check_cell_volume(points, cell, domain, 1)


@pytest.mark.skip_in_parallel
@pytest.mark.parametrize('order', range(1, 5))
def test_triangle_mesh_vtk(order):
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
    domain = ufl.Mesh(create_vector_element(
        "Lagrange", "triangle", order, gdim=2, lagrange_variant=basix.LagrangeVariant.equispaced))
    check_cell_volume(points, cell, domain, 0.5)


@pytest.mark.skip_in_parallel
@pytest.mark.parametrize('order', range(1, 5))
def test_tetrahedron_mesh_vtk(order):
    if order > 3:
        pytest.xfail("VTK permutation for order > 3 tetrahedra not implemented in DOLFINx.")
    points = []
    points += [[i / order, j / order, 0] for j in range(order + 1) for i in range(order + 1 - j)]
    for k in range(1, order):
        points += [[i / order, j / order + 0.1, k / order] for j in range(order + 1 - k)
                   for i in range(order + 1 - k - j)]

    points += [[0, 0, 1]]

    def coord_to_vertex(x, y, z):
        return z * (
            3 * order ** 2 - 3 * order * z + 12 * order + z ** 2 - 6 * z + 11
        ) // 6 + y * (2 * (order - z) + 3 - y) // 2 + x

    # Make the cell, following
    # https://blog.kitware.com/modeling-arbitrary-order-lagrange-finite-elements-in-the-visualization-toolkit/
    cell = [coord_to_vertex(x, y, z) for x, y, z in [
        (0, 0, 0), (order, 0, 0), (0, order, 0), (0, 0, order)]]

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
    domain = ufl.Mesh(create_vector_element(
        "Lagrange", "tetrahedron", order, gdim=3, lagrange_variant=basix.LagrangeVariant.equispaced))
    check_cell_volume(points, cell, domain, 1 / 6)


@pytest.mark.skip_in_parallel
@pytest.mark.parametrize('order', [1, 2, 3, 4])
def test_quadrilateral_mesh_vtk(order):
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
    cell = [coord_to_vertex(i, j)
            for i, j in [(0, 0), (order, 0), (order, order), (0, order)]]
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
    domain = ufl.Mesh(create_vector_element(
        "Q", "quadrilateral", order, gdim=2, lagrange_variant=basix.LagrangeVariant.equispaced))
    check_cell_volume(points, cell, domain, 1)


@pytest.mark.skip_in_parallel
@pytest.mark.parametrize('order', [1, 2, 3, 4])
def test_hexahedron_mesh_vtk(order):
    if order > 2:
        pytest.xfail("VTK permutation for order > 2 hexahedra not implemented in DOLFINx.")
    random.seed(13)

    points = []
    points += [[i / order, j / order, 0] for j in range(order + 1) for i in range(order + 1)]
    for k in range(1, order):
        points += [[i / order, j / order + 0.1, k / order] for j in range(order + 1) for i in range(order + 1)]

    points += [[i / order, j / order, 1] for j in range(order + 1) for i in range(order + 1)]

    def coord_to_vertex(x, y, z):
        return (order + 1) ** 2 * z + (order + 1) * y + x

    # Make the cell, following
    # https://blog.kitware.com/modeling-arbitrary-order-lagrange-finite-elements-in-the-visualization-toolkit/
    cell = [coord_to_vertex(x, y, z) for x, y, z in [
        (0, 0, 0), (order, 0, 0), (order, order, 0), (0, order, 0),
        (0, 0, order), (order, 0, order), (order, order, order), (0, order, order)]]

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
    domain = ufl.Mesh(create_vector_element(
        "Q", "hexahedron", order, gdim=3, lagrange_variant=basix.LagrangeVariant.equispaced))
    check_cell_volume(points, cell, domain, 1)


@pytest.mark.skip_in_parallel
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


@pytest.mark.skip_in_parallel
def test_xdmf_input_tri(datadir):
    with XDMFFile(MPI.COMM_WORLD, Path(datadir, "mesh.xdmf"), "r", encoding=XDMFFile.Encoding.ASCII) as xdmf:
        mesh = xdmf.read_mesh(name="Grid")
    surface = assemble_scalar(form(1 * dx(mesh)))
    assert mesh.comm.allreduce(surface, op=MPI.SUM) == pytest.approx(4 * np.pi, rel=1e-4)


@pytest.mark.skip_in_parallel
@pytest.mark.parametrize('order', range(1, 4))
@pytest.mark.parametrize('cell_type', [CellType.triangle, CellType.quadrilateral])
def test_gmsh_input_2d(order, cell_type):
    try:
        import gmsh
    except ImportError:
        pytest.skip()
    res = 0.2
    gmsh.initialize()
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", res)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", res)

    if cell_type == CellType.quadrilateral:
        gmsh.option.setNumber("Mesh.Algorithm", 2 if order == 2 else 5)

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
    name, dim, order, num_nodes, local_coords, num_first_order_nodes = gmsh.model.mesh.getElementProperties(
        element_types[0])

    cells = node_tags[0].reshape(-1, num_nodes) - 1
    if cell_type == CellType.triangle:
        gmsh_cell_id = gmsh.model.mesh.getElementType("triangle", order)
    elif cell_type == CellType.quadrilateral:
        gmsh_cell_id = gmsh.model.mesh.getElementType("quadrangle", order)
    gmsh.finalize()

    cells = cells[:, cell_perm_array(cell_type, cells.shape[1])]
    mesh = create_mesh(MPI.COMM_WORLD, cells, x, ufl_mesh(gmsh_cell_id, x.shape[1]))
    surface = assemble_scalar(form(1 * dx(mesh)))

    assert mesh.comm.allreduce(surface, op=MPI.SUM) == pytest.approx(4 * np.pi, rel=10 ** (-1 - order))


@pytest.mark.skip_in_parallel
@pytest.mark.parametrize('order', range(1, 4))
@pytest.mark.parametrize('cell_type', [CellType.tetrahedron, CellType.hexahedron])
def test_gmsh_input_3d(order, cell_type):
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
    name, dim, order, num_nodes, local_coords, num_first_order_nodes = gmsh.model.mesh.getElementProperties(
        element_types[0])

    cells = node_tags[0].reshape(-1, num_nodes) - 1
    if cell_type == CellType.tetrahedron:
        gmsh_cell_id = MPI.COMM_WORLD.bcast(gmsh.model.mesh.getElementType("tetrahedron", order), root=0)
    elif cell_type == CellType.hexahedron:
        gmsh_cell_id = MPI.COMM_WORLD.bcast(gmsh.model.mesh.getElementType("hexahedron", order), root=0)
    gmsh.finalize()

    # Permute the mesh topology from Gmsh ordering to DOLFINx ordering
    domain = ufl_mesh(gmsh_cell_id, 3)
    cells = cells[:, cell_perm_array(cell_type, cells.shape[1])]

    mesh = create_mesh(MPI.COMM_WORLD, cells, x, domain)
    volume = assemble_scalar(form(1 * dx(mesh)))
    assert mesh.comm.allreduce(volume, op=MPI.SUM) == pytest.approx(np.pi, rel=10 ** (-1 - order))


@pytest.mark.skip_in_parallel
def test_quadrilateral_cell_order_3():
    points = [
        [0., 0.], [1., 0.], [0., 1.], [1., 1.],
        [1 / 3, 2 / 9], [2 / 3, 2 / 9],
        [0., 1 / 3], [0., 2 / 3],
        [1., 1 / 3], [1., 2 / 3],
        [1 / 3, 1.], [2 / 3, 1.],
        [1 / 3, 13 / 27], [2 / 3, 13 / 27],
        [1 / 3, 20 / 27], [2 / 3, 20 / 27]
    ]

    cell = list(range(16))
    domain = ufl.Mesh(create_vector_element(
        "Q", "quadrilateral", 3, gdim=2, lagrange_variant=basix.LagrangeVariant.equispaced))
    check_cell_volume(points, cell, domain, 5 / 6)
