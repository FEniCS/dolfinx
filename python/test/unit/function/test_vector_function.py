# Copyright (C) 2019 Matthew Scroggs
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Test that the vectors in vector spaces are correctly oriented"""

import numpy as np
import pytest
from dolfinx_utils.test.skips import skip_in_parallel

from dolfinx import MPI, Function, FunctionSpace, cpp, fem
from dolfinx.cpp.mesh import CellType


@skip_in_parallel
@pytest.mark.parametrize('space_type', ["RT"])
@pytest.mark.parametrize('order', [1, 2, 3, 4, 5])
def test_div_conforming_triangle(space_type, order):
    """Checks that the vectors in div conforming spaces on a triangle are correctly oriented"""
    # Create simple triangle mesh
    def perform_test(points, cells, ordered):
        mesh = cpp.mesh.Mesh(MPI.comm_world, CellType.triangle, points,
                             np.array(cells), [], cpp.mesh.GhostMode.none)
        mesh.geometry.coord_mapping = fem.create_coordinate_map(mesh)
        if ordered:
            cpp.mesh.Ordering.order_simplex(mesh)
        V = FunctionSpace(mesh, (space_type, order))
        f = Function(V)
        output = []
        for dof in range(len(f.vector[:])):
            f.vector[:] = np.zeros(len(f.vector[:]))
            f.vector[dof] = 1
            points = np.array([[.5, .5, 0], [.5, .5, 0]])
            cells = np.array([0, 1])
            result = f.eval(points, cells)
            normal = np.array([-1., 1.])
            output.append(result.dot(normal))
        return output

    points = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    cells = np.array([[0, 1, 2], [2, 3, 0]])

    # Direction is incorrect if mesh is not ordered
    result = perform_test(points, cells, False)
    for i, j in result:
        assert np.allclose(i, -j)

    # Direction is correct if mesh is ordered
    result = perform_test(points, cells, True)
    for i, j in result:
        assert np.allclose(i, j)


@skip_in_parallel
@pytest.mark.parametrize('space_type', ["RT"])
@pytest.mark.parametrize('order', [1, 2, 3, 4, 5])
def test_div_conforming_tetrahedron(space_type, order):
    """Checks that the vectors in div conforming spaces on a tetrahedron are correctly oriented"""
    # Create simple tetrahedron mesh
    def perform_test(points, cells, ordered):
        mesh = cpp.mesh.Mesh(MPI.comm_world, CellType.tetrahedron, points,
                             np.array(cells), [], cpp.mesh.GhostMode.none)
        if ordered:
            cpp.mesh.Ordering.order_simplex(mesh)
        mesh.geometry.coord_mapping = fem.create_coordinate_map(mesh)
        V = FunctionSpace(mesh, (space_type, order))
        f = Function(V)
        output = []
        for dof in range(len(f.vector[:])):
            f.vector[:] = np.zeros(len(f.vector[:]))
            f.vector[dof] = 1
            points = np.array([[1 / 3, 1 / 3, 1 / 3], [1 / 3, 1 / 3, 1 / 3]])
            cells = np.array([0, 1])
            result = f.eval(points, cells)
            normal = np.array([1., 1., 1.])
            output.append(result.dot(normal))
        return output

    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])
    cells = np.array([[0, 1, 2, 3], [1, 3, 2, 4]])

    # Direction is incorrect if mesh is not ordered
    result = perform_test(points, cells, False)
    for i, j in result:
        assert np.allclose(i, -j)

    # Direction is correct if mesh is ordered
    result = perform_test(points, cells, True)
    for i, j in result:
        assert np.allclose(i, j)
