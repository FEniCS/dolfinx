# Copyright (C) 2015-2022 Garth N. Wells, Jørgen S. Dokken
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for the DiscreteOperator class"""

import numpy as np
import pytest

import ufl
from dolfinx.cpp.fem.petsc import discrete_gradient, interpolation_matrix
from dolfinx.fem import Expression, Function, FunctionSpace
from dolfinx.mesh import (CellType, GhostMode, create_unit_cube,
                          create_unit_square)

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
    """Test discrete gradient computation for lowest order elements."""

    V = FunctionSpace(mesh, ("Lagrange", 1))
    W = FunctionSpace(mesh, ("Nedelec 1st kind H(curl)", 1))
    G = discrete_gradient(V._cpp_object, W._cpp_object)
    assert G.getRefCount() == 1

    num_edges = mesh.topology.index_map(1).size_global
    m, n = G.getSize()
    assert m == num_edges
    assert n == mesh.topology.index_map(0).size_global

    G.assemble()
    assert np.isclose(G.norm(PETSc.NormType.FROBENIUS), np.sqrt(2.0 * num_edges))


@pytest.mark.parametrize("p", range(1, 4))
@pytest.mark.parametrize("q", range(1, 4))
@pytest.mark.parametrize("cell_type", [CellType.quadrilateral,
                                       CellType.triangle,
                                       CellType.tetrahedron,
                                       CellType.hexahedron])
def test_gradient_interpolation(cell_type, p, q):
    """Test discrete gradient computation with verification using Expression."""

    comm = MPI.COMM_WORLD
    if cell_type == CellType.triangle:
        mesh = create_unit_square(comm, 11, 6, ghost_mode=GhostMode.none, cell_type=cell_type)
        family0 = "Lagrange"
        family1 = "Nedelec 1st kind H(curl)"
    elif cell_type == CellType.quadrilateral:
        mesh = create_unit_square(comm, 11, 6, ghost_mode=GhostMode.none, cell_type=cell_type)
        family0 = "Q"
        family1 = "RTCE"
    elif cell_type == CellType.hexahedron:
        mesh = create_unit_cube(comm, 3, 3, 2, ghost_mode=GhostMode.none, cell_type=cell_type)
        family0 = "Q"
        family1 = "NCE"
    elif cell_type == CellType.tetrahedron:
        mesh = create_unit_cube(comm, 3, 2, 2, ghost_mode=GhostMode.none, cell_type=cell_type)
        family0 = "Lagrange"
        family1 = "Nedelec 1st kind H(curl)"

    V = FunctionSpace(mesh, (family0, p))
    W = FunctionSpace(mesh, (family1, q))
    G = discrete_gradient(V._cpp_object, W._cpp_object)
    G.assemble()

    u = Function(V)
    u.interpolate(lambda x: 2 * x[0]**p + 3 * x[1]**p)

    grad_u = Expression(ufl.grad(u), W.element.interpolation_points)
    w_expr = Function(W)
    w_expr.interpolate(grad_u)

    # Compute global matrix vector product
    w = Function(W)
    G.mult(u.vector, w.vector)
    w.x.scatter_forward()

    assert np.allclose(w_expr.x.array, w.x.array)


@pytest.mark.parametrize("p", range(1, 4))
@pytest.mark.parametrize("q", range(1, 4))
@pytest.mark.parametrize("from_lagrange", [True, False])
@pytest.mark.parametrize("cell_type", [CellType.quadrilateral,
                                       CellType.triangle,
                                       CellType.tetrahedron,
                                       CellType.hexahedron])
def test_interpolation_matrix(cell_type, p, q, from_lagrange):
    """Test that discrete interpolation matrix yields the same result as interpolation."""

    comm = MPI.COMM_WORLD
    if cell_type == CellType.triangle:
        mesh = create_unit_square(comm, 7, 5, ghost_mode=GhostMode.none, cell_type=cell_type)
        lagrange = "Lagrange" if from_lagrange else "DG"
        nedelec = "Nedelec 1st kind H(curl)"
    elif cell_type == CellType.quadrilateral:
        mesh = create_unit_square(comm, 11, 6, ghost_mode=GhostMode.none, cell_type=cell_type)
        lagrange = "Q" if from_lagrange else "DQ"
        nedelec = "RTCE"
    elif cell_type == CellType.hexahedron:
        mesh = create_unit_cube(comm, 3, 2, 1, ghost_mode=GhostMode.none, cell_type=cell_type)
        lagrange = "Q" if from_lagrange else "DQ"
        nedelec = "NCE"
    elif cell_type == CellType.tetrahedron:
        mesh = create_unit_cube(comm, 3, 2, 2, ghost_mode=GhostMode.none, cell_type=cell_type)
        lagrange = "Lagrange" if from_lagrange else "DG"
        nedelec = "Nedelec 1st kind H(curl)"
    v_el = ufl.VectorElement(lagrange, mesh.ufl_cell(), p)
    s_el = ufl.FiniteElement(nedelec, mesh.ufl_cell(), q)
    if from_lagrange:
        el0 = v_el
        el1 = s_el
    else:
        el0 = s_el
        el1 = v_el

    V = FunctionSpace(mesh, el0)
    W = FunctionSpace(mesh, el1)
    G = interpolation_matrix(V._cpp_object, W._cpp_object)
    G.assemble()

    u = Function(V)

    def f(x):
        if mesh.geometry.dim == 2:
            return (x[1]**p, x[0]**p)
        else:
            return (x[0]**p, x[2]**p, x[1]**p)
    u.interpolate(f)
    w_vec = Function(W)
    w_vec.interpolate(u)

    # Compute global matrix vector product
    w = Function(W)
    G.mult(u.vector, w.vector)
    w.x.scatter_forward()

    assert np.allclose(w_vec.x.array, w.x.array)
