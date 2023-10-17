# Copyright (C) 2015-2022 Garth N. Wells, Jørgen S. Dokken
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for the DiscreteOperator class"""

import numpy as np
import pytest
import scipy

import dolfinx.la
import ufl
from dolfinx.cpp.fem import discrete_gradient
from dolfinx.fem import Expression, Function, functionspace
from dolfinx.mesh import (CellType, GhostMode, create_unit_cube,
                          create_unit_square)

from mpi4py import MPI


@pytest.mark.parametrize("mesh", [create_unit_square(MPI.COMM_WORLD, 11, 6,
                                                     ghost_mode=GhostMode.none, dtype=np.float32),
                                  create_unit_square(MPI.COMM_WORLD, 11, 6,
                                                     ghost_mode=GhostMode.shared_facet, dtype=np.float64),
                                  create_unit_cube(MPI.COMM_WORLD, 4, 3, 7,
                                                   ghost_mode=GhostMode.none, dtype=np.float64),
                                  create_unit_cube(MPI.COMM_WORLD, 4, 3, 7,
                                                   ghost_mode=GhostMode.shared_facet, dtype=np.float32)])
def test_gradient(mesh):
    """Test discrete gradient computation for lowest order elements."""
    V = functionspace(mesh, ("Lagrange", 1))
    W = functionspace(mesh, ("Nedelec 1st kind H(curl)", 1))
    G = discrete_gradient(V._cpp_object, W._cpp_object)
    # N.B. do not scatter_rev G - doing so would transfer rows to other processes
    # where they will be summed to give an incorrect matrix

    num_edges = mesh.topology.index_map(1).size_global
    m, n = G.index_map(0).size_global, G.index_map(1).size_global
    assert m == num_edges
    assert n == mesh.topology.index_map(0).size_global
    assert np.isclose(G.squared_norm(), 2.0 * num_edges)


@pytest.mark.parametrize("p", range(1, 4))
@pytest.mark.parametrize("q", range(1, 4))
@pytest.mark.parametrize("cell_type", [CellType.quadrilateral,
                                       CellType.triangle,
                                       CellType.tetrahedron,
                                       CellType.hexahedron])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_gradient_interpolation(cell_type, p, q, dtype):
    """Test discrete gradient computation with verification using Expression."""
    comm = MPI.COMM_WORLD
    if cell_type == CellType.triangle:
        mesh = create_unit_square(comm, 11, 6, ghost_mode=GhostMode.none, cell_type=cell_type, dtype=dtype)
        family0 = "Lagrange"
        family1 = "Nedelec 1st kind H(curl)"
    elif cell_type == CellType.quadrilateral:
        mesh = create_unit_square(comm, 11, 6, ghost_mode=GhostMode.none, cell_type=cell_type, dtype=dtype)
        family0 = "Q"
        family1 = "RTCE"
    elif cell_type == CellType.hexahedron:
        mesh = create_unit_cube(comm, 3, 3, 2, ghost_mode=GhostMode.none, cell_type=cell_type, dtype=dtype)
        family0 = "Q"
        family1 = "NCE"
    elif cell_type == CellType.tetrahedron:
        mesh = create_unit_cube(comm, 3, 2, 2, ghost_mode=GhostMode.none, cell_type=cell_type, dtype=dtype)
        family0 = "Lagrange"
        family1 = "Nedelec 1st kind H(curl)"

    V = functionspace(mesh, (family0, p))
    W = functionspace(mesh, (family1, q))
    G = discrete_gradient(V._cpp_object, W._cpp_object)
    # N.B. do not scatter_rev G - doing so would transfer rows to other processes
    # where they will be summed to give an incorrect matrix

    # Vector for 'u' needs additional ghosts defined in columns of G
    uvec = dolfinx.la.vector(G.index_map(1), dtype=dtype)
    u = Function(V, uvec, dtype=dtype)
    u.interpolate(lambda x: 2 * x[0]**p + 3 * x[1]**p)

    grad_u = Expression(ufl.grad(u), W.element.interpolation_points(), dtype=dtype)
    w_expr = Function(W, dtype=dtype)
    w_expr.interpolate(grad_u)

    # Compute global matrix vector product
    w = Function(W, dtype=dtype)

    # Get the local part of G (no ghost rows)
    nrlocal = G.index_map(0).size_local
    nnzlocal = G.indptr[nrlocal]
    Glocal = scipy.sparse.csr_matrix((G.data[:nnzlocal], G.indices[:nnzlocal], G.indptr[:nrlocal + 1]))

    # MatVec
    w.x.array[:nrlocal] = Glocal @ u.x.array
    w.x.scatter_forward()

    atol = 100 * np.finfo(dtype).resolution
    assert np.allclose(w_expr.x.array, w.x.array, atol=atol)
