# Copyright (C) 2015-2022 Garth N. Wells, Jørgen S. Dokken
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for the DiscreteOperator class"""

from mpi4py import MPI
from petsc4py import PETSc

import numpy as np
import pytest

import ufl
from basix.ufl import element
from dolfinx import default_real_type
from dolfinx.fem import Expression, Function, assemble_scalar, form, functionspace
from dolfinx.fem.petsc import discrete_gradient, interpolation_matrix
from dolfinx.mesh import CellType, GhostMode, create_mesh, create_unit_cube, create_unit_square


@pytest.mark.skip_in_parallel
@pytest.mark.parametrize(
    "mesh",
    [
        create_unit_square(MPI.COMM_WORLD, 11, 6, ghost_mode=GhostMode.none),
        create_unit_square(MPI.COMM_WORLD, 11, 6, ghost_mode=GhostMode.shared_facet),
        create_unit_cube(MPI.COMM_WORLD, 4, 3, 7, ghost_mode=GhostMode.none),
        create_unit_cube(MPI.COMM_WORLD, 4, 3, 7, ghost_mode=GhostMode.shared_facet),
    ],
)
def test_gradient_petsc(mesh):
    """Test discrete gradient computation for lowest order elements."""
    V = functionspace(mesh, ("Lagrange", 1))
    W = functionspace(mesh, ("Nedelec 1st kind H(curl)", 1))
    G = discrete_gradient(V, W)
    assert G.getRefCount() == 1
    num_edges = mesh.topology.index_map(1).size_global
    m, n = G.getSize()
    assert m == num_edges
    assert n == mesh.topology.index_map(0).size_global
    G.assemble()
    assert np.isclose(G.norm(PETSc.NormType.FROBENIUS), np.sqrt(2.0 * num_edges))
    G.destroy()


@pytest.mark.parametrize("p", range(1, 4))
@pytest.mark.parametrize("q", range(1, 4))
@pytest.mark.parametrize(
    "cell_type",
    [CellType.quadrilateral, CellType.triangle, CellType.tetrahedron, CellType.hexahedron],
)
def test_gradient_interpolation_petsc(cell_type, p, q):
    """Test discrete gradient computation with verification using Expression."""
    comm = MPI.COMM_WORLD
    if cell_type == CellType.triangle:
        mesh = create_unit_square(
            comm, 11, 6, ghost_mode=GhostMode.none, cell_type=cell_type, dtype=default_real_type
        )
        family0 = "Lagrange"
        family1 = "Nedelec 1st kind H(curl)"
    elif cell_type == CellType.quadrilateral:
        mesh = create_unit_square(
            comm, 11, 6, ghost_mode=GhostMode.none, cell_type=cell_type, dtype=default_real_type
        )
        family0 = "Q"
        family1 = "RTCE"
    elif cell_type == CellType.hexahedron:
        mesh = create_unit_cube(
            comm, 3, 3, 2, ghost_mode=GhostMode.none, cell_type=cell_type, dtype=default_real_type
        )
        family0 = "Q"
        family1 = "NCE"
    elif cell_type == CellType.tetrahedron:
        mesh = create_unit_cube(
            comm, 3, 2, 2, ghost_mode=GhostMode.none, cell_type=cell_type, dtype=default_real_type
        )
        family0 = "Lagrange"
        family1 = "Nedelec 1st kind H(curl)"

    V = functionspace(mesh, (family0, p))
    W = functionspace(mesh, (family1, q))
    G = discrete_gradient(V, W)
    G.assemble()

    u = Function(V)
    u.interpolate(lambda x: 2 * x[0] ** p + 3 * x[1] ** p)

    grad_u = Expression(ufl.grad(u), W.element.interpolation_points())
    w_expr = Function(W)
    w_expr.interpolate(grad_u)

    # Compute global matrix vector product
    w = Function(W)
    G.mult(u.x.petsc_vec, w.x.petsc_vec)
    w.x.scatter_forward()

    atol = 100 * np.finfo(default_real_type).resolution
    assert np.allclose(w_expr.x.array, w.x.array, atol=atol)
    G.destroy()


@pytest.mark.parametrize("p", range(1, 4))
@pytest.mark.parametrize("q", range(1, 4))
@pytest.mark.parametrize("from_lagrange", [True, False])
@pytest.mark.parametrize(
    "cell_type",
    [CellType.quadrilateral, CellType.triangle, CellType.tetrahedron, CellType.hexahedron],
)
def test_interpolation_matrix_petsc(cell_type, p, q, from_lagrange):
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
    G = interpolation_matrix(V, W)
    G.assemble()

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
    G.mult(u.x.petsc_vec, w.x.petsc_vec)
    w.x.scatter_forward()

    atol = 100 * np.finfo(default_real_type).resolution
    assert np.allclose(w_vec.x.array, w.x.array, atol=atol)
    G.destroy()


@pytest.mark.skip_in_parallel
def test_nonaffine_discrete_operator_petsc():
    """Check that discrete operator is consistent with normal
    interpolation between non-matching maps on non-affine geometries"""
    points = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 2, 0],
            [1, 2, 0],
            [0, 0, 3],
            [1, 0, 3],
            [0, 2, 3],
            [1, 2, 3],
            [0.5, 0, 0],
            [0, 1, 0],
            [0, 0, 1.5],
            [1, 1, 0],
            [1, 0, 1.5],
            [0.5, 2, 0],
            [0, 2, 1.5],
            [1, 2, 1.5],
            [0.5, 0, 3],
            [0, 1, 3],
            [1, 1, 3],
            [0.5, 2, 3],
            [0.5, 1, 0],
            [0.5, -0.1, 1.5],
            [0, 1, 1.5],
            [1, 1, 1.5],
            [0.5, 2, 1.5],
            [0.5, 1, 3],
            [0.5, 1, 1.5],
        ],
        dtype=default_real_type,
    )

    cells = np.array([range(len(points))], dtype=np.int32)
    cell_type = CellType.hexahedron
    domain = ufl.Mesh(element("Lagrange", cell_type.name, 2, shape=(3,), dtype=default_real_type))
    mesh = create_mesh(MPI.COMM_WORLD, cells, points, domain)
    gdim = mesh.geometry.dim
    W = functionspace(mesh, ("DG", 1, (gdim,)))
    V = functionspace(mesh, ("NCE", 4))
    w, v = Function(W), Function(V)
    w.interpolate(lambda x: x)
    v.interpolate(w)

    G = interpolation_matrix(W, V)
    G.assemble()

    # Compute global matrix vector product
    v_vec = Function(V)
    G.mult(w.x.petsc_vec, v_vec.x.petsc_vec)
    v_vec.x.scatter_forward()
    atol = 10 * np.finfo(default_real_type).resolution
    assert np.allclose(v_vec.x.array, v.x.array, atol=atol)

    s = assemble_scalar(form(ufl.inner(w - v, w - v) * ufl.dx))
    assert np.isclose(s, 0, atol=atol)
    G.destroy()
