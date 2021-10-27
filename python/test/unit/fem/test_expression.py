# Copyright (C) 2021 Jorgen S. Dokken
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for dolfinx.cpp.fem.CoordinateMap.pull_back and dolfinx.Expression"""

import dolfinx
import dolfinx.geometry
import numpy as np
import pytest
import ufl
from mpi4py import MPI
from petsc4py import PETSc


@pytest.mark.skipif(np.issubdtype(PETSc.ScalarType, np.complexfloating),
                    reason="Complex expression not implemented in ufc")
def test_expression():
    """Test UFL expression evaluation"""
    mesh = dolfinx.UnitSquareMesh(MPI.COMM_WORLD, 10, 10)
    V = dolfinx.FunctionSpace(mesh, ("Lagrange", 2))

    def f(x):
        return 2 * x[0]**2 + x[1]**2

    def gradf(x):
        return np.asarray([4 * x[0], 2 * x[1]])

    u = dolfinx.Function(V)
    u.interpolate(f)
    u.x.scatter_forward()

    grad_u = ufl.grad(u)
    points = np.array([[0.15, 0.3, 0], [0.953, 0.81, 0]])
    gdim = mesh.geometry.dim
    tdim = mesh.topology.dim
    bb = dolfinx.geometry.BoundingBoxTree(mesh, tdim)

    # Find colliding cells on proc
    closest_cell = []
    local_map = []
    for i, p in enumerate(points):
        cells = dolfinx.geometry.compute_collisions_point(bb, p)
        if len(cells) > 0:
            actual_cells = dolfinx.geometry.select_colliding_cells(mesh, cells, p, 1)
            if len(actual_cells) > 0:
                local_map.append(i)
                closest_cell.append(actual_cells[0])

    num_dofs_x = mesh.geometry.dofmap.links(0).size  # NOTE: Assumes same cell geometry in whole mesh
    t_imap = mesh.topology.index_map(tdim)
    num_cells = t_imap.size_local + t_imap.num_ghosts
    x = mesh.geometry.x
    x_dofs = mesh.geometry.dofmap.array.reshape(num_cells, num_dofs_x)
    cell_geometry = np.zeros((num_dofs_x, gdim), dtype=np.float64)
    points_ref = np.zeros((len(local_map), tdim))

    # Map cells on process back to reference element
    for i, cell in enumerate(closest_cell):
        cell_geometry[:] = x[x_dofs[cell], :gdim]
        point_ref = mesh.geometry.cmap.pull_back(points[local_map[i]][:gdim].reshape(1, -1), cell_geometry)
        points_ref[i] = point_ref

    # Eval using Expression
    expr = dolfinx.Expression(grad_u, points_ref)
    grad_u_at_x = expr.eval(closest_cell).reshape(len(closest_cell), points_ref.shape[0], gdim)

    # Compare solutions
    for i, cell in enumerate(closest_cell):
        point = points[local_map[i]]
        assert np.allclose(grad_u_at_x[i, i], gradf(point))
