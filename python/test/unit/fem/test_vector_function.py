# Copyright (C) 2019 Matthew Scroggs
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Test that the vectors in vector spaces are correctly oriented"""

from mpi4py import MPI

import numpy as np
import pytest

import ufl
from basix.ufl import element
from dolfinx import default_real_type
from dolfinx.fem import Function, functionspace
from dolfinx.mesh import create_mesh


@pytest.mark.skip_in_parallel
@pytest.mark.parametrize("space_type", ["RT"])
@pytest.mark.parametrize("order", [1, 2, 3, 4, 5])
def test_div_conforming_triangle(space_type, order):
    """Checks that the vectors in div conforming spaces on a triangle are correctly oriented"""

    # Create simple triangle mesh
    def perform_test(points, cells):
        domain = ufl.Mesh(element("Lagrange", "triangle", 1, shape=(2,), dtype=default_real_type))
        mesh = create_mesh(MPI.COMM_WORLD, cells, points, domain)
        V = functionspace(mesh, (space_type, order))
        f = Function(V)
        x = f.x.array
        output = []
        for dof in range(len(x)):
            x[:] = 0.0
            x[dof] = 1
            points = np.array([[0.5, 0.5, 0], [0.5, 0.5, 0]])
            cells = np.array([0, 1])
            result = f.eval(points, cells)
            normal = np.array([-1.0, 1.0])
            output.append(result.dot(normal))
        return output

    points = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=default_real_type)
    cells = np.array([[0, 1, 2], [2, 3, 0]])
    result = perform_test(points, cells)
    for i, j in result:
        assert i == pytest.approx(j, abs=1.0e-5)


@pytest.mark.skip_in_parallel
@pytest.mark.parametrize("space_type", ["RT"])
@pytest.mark.parametrize("order", [1, 2, 3, 4, 5])
def test_div_conforming_tetrahedron(space_type, order):
    """Checks that the vectors in div conforming spaces on a tetrahedron are correctly oriented"""

    # Create simple tetrahedron cell mesh
    def perform_test(points, cells):
        domain = ufl.Mesh(
            element("Lagrange", "tetrahedron", 1, shape=(3,), dtype=default_real_type)
        )
        mesh = create_mesh(MPI.COMM_WORLD, cells, points, domain)
        V = functionspace(mesh, (space_type, order))
        f = Function(V)
        output = []
        x = f.x.array
        for dof in range(len(x)):
            x[:] = 0.0
            x[dof] = 1
            points = np.array([[1 / 3, 1 / 3, 1 / 3], [1 / 3, 1 / 3, 1 / 3]])
            cells = np.array([0, 1])
            result = f.eval(points, cells)
            normal = np.array([1.0, 1.0, 1.0])
            output.append(result.dot(normal))
        return output

    points = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=default_real_type
    )
    cells = np.array([[0, 1, 2, 3], [1, 3, 2, 4]])
    result = perform_test(points, cells)
    for i, j in result:
        assert i == pytest.approx(j, rel=1.0e-6, abs=1.0e-4)
