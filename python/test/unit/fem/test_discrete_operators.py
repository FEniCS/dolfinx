# Copyright (C) 2015-2022 Garth N. Wells, Jørgen S. Dokken
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for the DiscreteOperator class"""

from mpi4py import MPI

import numpy as np
import pytest
import scipy

import dolfinx.la
import ufl
from basix.ufl import element
from dolfinx.fem import Expression, Function, discrete_gradient, functionspace
from dolfinx.mesh import CellType, GhostMode, create_unit_cube, create_unit_square


@pytest.mark.parametrize(
    "mesh",
    [
        create_unit_square(MPI.COMM_WORLD, 11, 6, ghost_mode=GhostMode.none, dtype=np.float32),
        create_unit_square(
            MPI.COMM_WORLD, 11, 6, ghost_mode=GhostMode.shared_facet, dtype=np.float64
        ),
        create_unit_cube(MPI.COMM_WORLD, 4, 3, 7, ghost_mode=GhostMode.none, dtype=np.float64),
        create_unit_cube(
            MPI.COMM_WORLD, 4, 3, 7, ghost_mode=GhostMode.shared_facet, dtype=np.float32
        ),
    ],
)
def test_gradient(mesh):
    """Test discrete gradient computation for lowest order elements."""
    V = functionspace(mesh, ("Lagrange", 1))
    W = functionspace(mesh, ("Nedelec 1st kind H(curl)", 1))
    G = discrete_gradient(V, W)
    # N.B. do not scatter_rev G - doing so would transfer rows to other processes
    # where they will be summed to give an incorrect matrix

    num_edges = mesh.topology.index_map(1).size_global
    m, n = G.index_map(0).size_global, G.index_map(1).size_global
    assert m == num_edges
    assert n == mesh.topology.index_map(0).size_global
    assert np.isclose(G.squared_norm(), 2.0 * num_edges)


@pytest.mark.parametrize("p", range(1, 4))
@pytest.mark.parametrize("q", range(1, 4))
@pytest.mark.parametrize(
    "cell_type",
    [
        (
            create_unit_square(
                MPI.COMM_WORLD,
                11,
                6,
                ghost_mode=GhostMode.none,
                cell_type=CellType.triangle,
                dtype=np.float32,
            ),
            "Lagrange",
            "Nedelec 1st kind H(curl)",
        ),
        (
            create_unit_square(
                MPI.COMM_WORLD,
                11,
                6,
                ghost_mode=GhostMode.none,
                cell_type=CellType.triangle,
                dtype=np.float64,
            ),
            "Lagrange",
            "Nedelec 1st kind H(curl)",
        ),
        (
            create_unit_square(
                MPI.COMM_WORLD,
                11,
                6,
                ghost_mode=GhostMode.none,
                cell_type=CellType.quadrilateral,
                dtype=np.float32,
            ),
            "Q",
            "RTCE",
        ),
        (
            create_unit_square(
                MPI.COMM_WORLD,
                11,
                6,
                ghost_mode=GhostMode.none,
                cell_type=CellType.quadrilateral,
                dtype=np.float64,
            ),
            "Q",
            "RTCE",
        ),
        (
            create_unit_cube(
                MPI.COMM_WORLD,
                3,
                3,
                2,
                ghost_mode=GhostMode.none,
                cell_type=CellType.tetrahedron,
                dtype=np.float32,
            ),
            "Lagrange",
            "Nedelec 1st kind H(curl)",
        ),
        (
            create_unit_cube(
                MPI.COMM_WORLD,
                3,
                3,
                2,
                ghost_mode=GhostMode.none,
                cell_type=CellType.tetrahedron,
                dtype=np.float64,
            ),
            "Lagrange",
            "Nedelec 1st kind H(curl)",
        ),
        (
            create_unit_cube(
                MPI.COMM_WORLD,
                3,
                3,
                2,
                ghost_mode=GhostMode.none,
                cell_type=CellType.hexahedron,
                dtype=np.float32,
            ),
            "Q",
            "NCE",
        ),
        (
            create_unit_cube(
                MPI.COMM_WORLD,
                3,
                2,
                2,
                ghost_mode=GhostMode.none,
                cell_type=CellType.hexahedron,
                dtype=np.float64,
            ),
            "Q",
            "NCE",
        ),
    ],
)
def test_gradient_interpolation(cell_type, p, q):
    """Test discrete gradient computation with verification using Expression."""
    mesh, family0, family1 = cell_type
    dtype = mesh.geometry.x.dtype

    V = functionspace(mesh, (family0, p))
    W = functionspace(mesh, (family1, q))
    G = discrete_gradient(V, W)
    # N.B. do not scatter_rev G - doing so would transfer rows to other
    # processes where they will be summed to give an incorrect matrix

    # Vector for 'u' needs additional ghosts defined in columns of G
    uvec = dolfinx.la.vector(G.index_map(1), dtype=dtype)
    u = Function(V, uvec, dtype=dtype)
    u.interpolate(lambda x: 2 * x[0] ** p + 3 * x[1] ** p)

    grad_u = Expression(ufl.grad(u), W.element.interpolation_points(), dtype=dtype)
    w_expr = Function(W, dtype=dtype)
    w_expr.interpolate(grad_u)

    # Compute global matrix vector product
    w = Function(W, dtype=dtype)

    # Get the local part of G (no ghost rows)
    nrlocal = G.index_map(0).size_local
    nnzlocal = G.indptr[nrlocal]
    Glocal = scipy.sparse.csr_matrix(
        (G.data[:nnzlocal], G.indices[:nnzlocal], G.indptr[: nrlocal + 1])
    )

    # MatVec
    w.x.array[:nrlocal] = Glocal @ u.x.array
    w.x.scatter_forward()

    atol = 1000 * np.finfo(dtype).resolution
    assert np.allclose(w_expr.x.array, w.x.array, atol=atol)


@pytest.mark.parametrize("p", range(1, 4))
@pytest.mark.parametrize("q", range(1, 4))
@pytest.mark.parametrize("from_lagrange", [True, False])
@pytest.mark.parametrize(
    "cell_type",
    [CellType.quadrilateral, CellType.triangle, CellType.tetrahedron, CellType.hexahedron],
)
def test_interpolation_matrix(cell_type, p, q, from_lagrange):
    """Test that discrete interpolation matrix yields the same result as interpolation."""
    from dolfinx import default_real_type
    from dolfinx.fem import interpolation_matrix

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
    v_el = element(
        lagrange, mesh.basix_cell(), p, shape=(mesh.geometry.dim,), dtype=default_real_type
    )
    s_el = element(nedelec, mesh.basix_cell(), q, dtype=default_real_type)
    if from_lagrange:
        el0 = v_el
        el1 = s_el
    else:
        el0 = s_el
        el1 = v_el

    V = functionspace(mesh, el0)
    W = functionspace(mesh, el1)
    G = interpolation_matrix(V, W).to_scipy()

    u = Function(V)

    def f(x):
        if mesh.geometry.dim == 2:
            return (x[1] ** p, x[0] ** p)
        else:
            return (x[0] ** p, x[2] ** p, x[1] ** p)

    u.interpolate(f)
    w_vec = Function(W)
    w_vec.interpolate(u)

    # Compute global matrix vector product
    w = Function(W)
    ux = np.zeros(G.shape[1])
    ux[: len(u.x.array)] = u.x.array[:]
    w.x.array[: G.shape[0]] = G @ ux
    w.x.scatter_forward()

    atol = 100 * np.finfo(default_real_type).resolution
    assert np.allclose(w_vec.x.array, w.x.array, atol=atol)
