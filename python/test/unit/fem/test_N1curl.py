# Copyright (C) 2009-2020 Garth N. Wells, Matthew W. Scroggs and Jorgen S. Dokken
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for the fem interface"""

from itertools import permutations
import numpy as np
import pytest
from dolfinx_utils.test.skips import skip_in_parallel

from dolfinx import MPI, cpp, fem, Mesh, FunctionSpace
from ufl import inner, dx, TestFunction, TrialFunction
from dolfinx.cpp.mesh import CellType, Ordering


@skip_in_parallel
@pytest.mark.parametrize('space_order', [1, 2, 3])
def test__(space_order):
    results = []
    for cell in permutations(range(4)):
        points = np.array([[0., 0., 0.], [2., 0., 0.],
                           [1., np.sqrt(3), 0.],
                           [1., 1. / np.sqrt(3), 2. * np.sqrt(2 / 3)]])

        mesh = Mesh(MPI.comm_world, CellType.tetrahedron, points, [cell],
                    [], cpp.mesh.GhostMode.none)
        mesh.geometry.coord_mapping = fem.create_coordinate_map(mesh)
        mesh.create_connectivity_all()
        V = FunctionSpace(mesh, ("N1curl", space_order))
        v = TestFunction(V)
        a = v[0] * dx
        result = fem.assemble_vector(a)
        result.assemble()
        results.append(result)
        print(*[i for i in result[:]])

    e_ = space_order
    f_ = space_order * (space_order - 1)
    i_ = space_order * (space_order - 1) * (space_order - 2) // 2
    n_dofs = 6 * e_ + 4 * f_ + i_
    if space_order == 3:
        assert n_dofs == 45

    for r in results:
        for i in range(6 * e_):
            assert np.isclose(abs(results[0][i]), abs(r[i]))
        for i in range(6 * e_, 6 * e_ + 4 * f_):
            assert np.isclose(results[0][i], r[i])



    return

    # Matrix test
    results = []
    spaces = []
    for i in range(2):
        mesh = two_unit_cells()
        if i == 1:
            Ordering.order_simplex(mesh)
        V = FunctionSpace(mesh, ("N1curl", space_order))
        spaces.append(V)
        u, v = TrialFunction(V), TestFunction(V)
        a = inner(u, v) * dx
        result = fem.assemble_matrix(a, [])
        result.assemble()
        results.append(result)

    e_ = space_order
    f_ = space_order * (space_order - 1)
    i_ = space_order * (space_order - 1) * (space_order - 2) // 2
    n_dofs = 6 * e_ + 4 * f_ + i_
    total_dofs = n_dofs * 2 - f_ - 3 * e_
    if space_order == 3:
        assert n_dofs == 45
        assert total_dofs == 75

    for i in range(total_dofs):
        for j in range(total_dofs):
            assert np.isclose(results[0][i, j], results[1][i, j])
