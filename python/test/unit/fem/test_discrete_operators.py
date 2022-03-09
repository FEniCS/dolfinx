# Copyright (C) 2015-2022 Garth N. Wells, Jørgen S. Dokken
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for the DiscreteOperator class"""

import numpy as np
import pytest

import ufl
from dolfinx.cpp.fem.petsc import create_discrete_gradient
from dolfinx.fem import Expression, Function, FunctionSpace
from dolfinx.mesh import GhostMode, create_unit_cube, create_unit_square, CellType

from mpi4py import MPI
from petsc4py import PETSc


@pytest.mark.skip_in_parallel
@pytest.mark.parametrize("mesh", [
    create_unit_square(MPI.COMM_WORLD, 11, 6, ghost_mode=GhostMode.none),
    create_unit_square(MPI.COMM_WORLD, 11, 6, ghost_mode=GhostMode.shared_facet),
    create_unit_cube(MPI.COMM_WORLD, 4, 3, 7, ghost_mode=GhostMode.none),
    create_unit_cube(MPI.COMM_WORLD, 4, 3, 7, ghost_mode=GhostMode.shared_facet)
])
def test_gradient(mesh):
    """Test discrete gradient computation (typically used for curl-curl
    AMG preconditioners"""

    V = FunctionSpace(mesh, ("Lagrange", 1))
    W = FunctionSpace(mesh, ("Nedelec 1st kind H(curl)", 1))
    G = create_discrete_gradient(W._cpp_object, V._cpp_object)
    assert G.getRefCount() == 1
    num_edges = mesh.topology.index_map(1).size_global
    m, n = G.getSize()
    assert m == num_edges
    assert n == mesh.topology.index_map(0).size_global
    assert np.isclose(G.norm(PETSc.NormType.FROBENIUS), np.sqrt(2.0 * num_edges))


@pytest.mark.parametrize("p", range(1, 5))
@pytest.mark.parametrize("q", range(1, 5))
@pytest.mark.parametrize("cell_type", [
    CellType.quadrilateral, CellType.triangle, CellType.tetrahedron, CellType.hexahedron])
def test_interpolation_matrix(cell_type, p, q):
    """Test discrete gradient computation (typically used for curl-curl
    AMG preconditioners"""

    if cell_type == CellType.triangle:
        mesh = create_unit_square(MPI.COMM_WORLD, 11, 6, ghost_mode=GhostMode.none, cell_type=cell_type)
        l_fam = "Lagrange"
        n_fam = "Nedelec 1st kind H(curl)"
    elif cell_type == CellType.quadrilateral:
        mesh = create_unit_square(MPI.COMM_WORLD, 11, 6, ghost_mode=GhostMode.none, cell_type=cell_type)
        l_fam = "Q"
        n_fam = "RTCE"
    elif cell_type == CellType.hexahedron:
        mesh = create_unit_cube(MPI.COMM_WORLD, 3, 3, 2, ghost_mode=GhostMode.none, cell_type=cell_type)
        l_fam = "Q"
        n_fam = "NCE"
    elif cell_type == CellType.tetrahedron:
        mesh = create_unit_cube(MPI.COMM_WORLD, 3, 2, 2, ghost_mode=GhostMode.none, cell_type=cell_type)
        l_fam = "Lagrange"
        n_fam = "Nedelec 1st kind H(curl)"

    V = FunctionSpace(mesh, (l_fam, p))
    W = FunctionSpace(mesh, (n_fam, q))
    G = create_discrete_gradient(W._cpp_object, V._cpp_object)

    u = Function(V)
    u.interpolate(lambda x: 2 * x[0]**p + 3 * x[1]**p)
    u.x.scatter_forward()

    grad_u = Expression(ufl.grad(u), W.element.interpolation_points)
    w_expr = Function(W)
    w_expr.interpolate(grad_u)

    # Compute global matrix vector product
    w = Function(W)
    G.mult(u.vector, w.vector)
    w.x.scatter_forward()

    assert np.allclose(w_expr.x.array, w.x.array)
