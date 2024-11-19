# Copyright (C) 2021 Matthew W. Scroggs and Jack Hale
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from mpi4py import MPI

import numpy as np
import pytest

import dolfinx
import ufl
from basix.ufl import element, mixed_element
from dolfinx import default_real_type
from dolfinx.fem import form, functionspace
from dolfinx.mesh import CellType, GhostMode, create_unit_cube, create_unit_square


@pytest.mark.skip_in_parallel
@pytest.mark.parametrize("cell", [ufl.triangle, ufl.tetrahedron])
@pytest.mark.parametrize("degree", [1, 2])
@pytest.mark.parametrize("rank, family", [(0, "Lagrange"), (1, "Lagrange"), (1, "N1curl")])
def test_mixed_element(rank, family, cell, degree):
    if cell == ufl.triangle:
        mesh = create_unit_square(
            MPI.COMM_WORLD, 1, 1, CellType.triangle, ghost_mode=GhostMode.shared_facet
        )
    else:
        mesh = create_unit_cube(
            MPI.COMM_WORLD, 1, 1, 1, CellType.tetrahedron, ghost_mode=GhostMode.shared_facet
        )

    shape = (mesh.geometry.dim,) * rank
    norms = []
    U_el = element(family, cell.cellname(), degree, shape=shape, dtype=default_real_type)
    for i in range(3):
        U = functionspace(mesh, U_el)
        u = ufl.TrialFunction(U)
        v = ufl.TestFunction(U)
        a = form(ufl.inner(u, v) * ufl.dx)

        A = dolfinx.fem.assemble_matrix(a)
        A.scatter_reverse()
        norms.append(A.squared_norm())

    for i in norms[1:]:
        assert np.isclose(norms[0], i)


@pytest.mark.skip_in_parallel
def test_vector_element():
    # Function space containing a scalar should work
    mesh = create_unit_square(
        MPI.COMM_WORLD, 1, 1, CellType.triangle, ghost_mode=GhostMode.shared_facet
    )
    gdim = mesh.geometry.dim
    U = functionspace(mesh, ("P", 2, (gdim,)))
    u, v = ufl.TrialFunction(U), ufl.TestFunction(U)
    a = form(ufl.inner(u, v) * ufl.dx)
    A = dolfinx.fem.assemble_matrix(a)
    A.scatter_reverse()

    with pytest.raises(ValueError):
        # Function space containing a vector should throw an error rather
        # than segfaulting
        gdim = mesh.geometry.dim
        U = functionspace(mesh, ("RT", 2, (gdim + 1,)))
        u, v = ufl.TrialFunction(U), ufl.TestFunction(U)
        a = form(ufl.inner(u, v) * ufl.dx)
        A = dolfinx.fem.assemble_matrix(a)
        A.scatter_reverse()


@pytest.mark.skip_in_parallel
@pytest.mark.parametrize("d1", range(1, 4))
@pytest.mark.parametrize("d2", range(1, 4))
def test_element_product(d1, d2):
    mesh = create_unit_square(MPI.COMM_WORLD, 2, 2)
    P3 = element(
        "Lagrange", mesh.basix_cell(), d1, shape=(mesh.geometry.dim,), dtype=default_real_type
    )
    P1 = element("Lagrange", mesh.basix_cell(), d2, dtype=default_real_type)
    TH = mixed_element([P3, P1])
    W = functionspace(mesh, TH)

    u = ufl.TrialFunction(W)
    v = ufl.TestFunction(W)
    a = form(ufl.inner(u[0], v[0]) * ufl.dx)
    A = dolfinx.fem.assemble_matrix(a)
    A.scatter_reverse()

    W = functionspace(mesh, P3)
    u = ufl.TrialFunction(W)
    v = ufl.TestFunction(W)
    a = form(ufl.inner(u[0], v[0]) * ufl.dx)
    B = dolfinx.fem.assemble_matrix(a)
    B.scatter_reverse()

    assert np.isclose(A.squared_norm(), B.squared_norm())


def test_single_element_in_mixed_element():
    """
    Check that a mixed element with a single element is equivalent to a single element
    """
    mesh = dolfinx.mesh.create_unit_square(
        MPI.COMM_WORLD, 10, 3, ghost_mode=dolfinx.mesh.GhostMode.shared_facet
    )
    el = element("Lagrange", mesh.basix_cell(), 3)
    me = mixed_element([el])

    V = dolfinx.fem.functionspace(mesh, me)
    assert V.num_sub_spaces == 1

    W = dolfinx.fem.functionspace(mesh, el)
    np.testing.assert_allclose(W.dofmap.list, V.dofmap.list)

    def f(x):
        return x[0] ** 2 + x[1] ** 2

    u = dolfinx.fem.Function(V)
    u.sub(0).interpolate(f)
    w = dolfinx.fem.Function(W)
    w.interpolate(f)

    np.testing.assert_allclose(u.x.array, w.x.array)
