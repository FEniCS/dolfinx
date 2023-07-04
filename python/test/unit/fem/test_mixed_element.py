# Copyright (C) 2021 Matthew W. Scroggs and Jack Hale
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import numpy as np
import pytest

import dolfinx
import ufl
from basix.ufl import element, mixed_element
from dolfinx.fem import FunctionSpace, VectorFunctionSpace, form
from dolfinx.mesh import (CellType, GhostMode, create_unit_cube,
                          create_unit_square)

from mpi4py import MPI


@pytest.mark.skip_in_parallel
@pytest.mark.parametrize("cell", [ufl.triangle, ufl.tetrahedron])
@pytest.mark.parametrize("degree", [1, 2])
@pytest.mark.parametrize("rank, family", [(0, "Lagrange"), (1, "Lagrange"), (1, "N1curl")])
def test_mixed_element(rank, family, cell, degree):
    if cell == ufl.triangle:
        mesh = create_unit_square(MPI.COMM_WORLD, 1, 1, CellType.triangle, ghost_mode=GhostMode.shared_facet)
    else:
        mesh = create_unit_cube(MPI.COMM_WORLD, 1, 1, 1, CellType.tetrahedron, ghost_mode=GhostMode.shared_facet)

    norms = []
    U_el = element(family, cell.cellname(), degree, rank=rank)
    for i in range(3):
        U = FunctionSpace(mesh, U_el)
        u = ufl.TrialFunction(U)
        v = ufl.TestFunction(U)
        a = form(ufl.inner(u, v) * ufl.dx)

        A = dolfinx.fem.assemble_matrix(a)
        A.finalize()
        norms.append(A.squared_norm())

        U_el = mixed_element([U_el])

    for i in norms[1:]:
        assert np.isclose(norms[0], i)


@pytest.mark.skip_in_parallel
def test_vector_element():
    # VectorFunctionSpace containing a scalar should work
    mesh = create_unit_square(MPI.COMM_WORLD, 1, 1, CellType.triangle,
                              ghost_mode=GhostMode.shared_facet)
    U = VectorFunctionSpace(mesh, ("P", 2))
    u = ufl.TrialFunction(U)
    v = ufl.TestFunction(U)
    a = form(ufl.inner(u, v) * ufl.dx)
    A = dolfinx.fem.assemble_matrix(a)
    A.finalize()

    with pytest.raises(ValueError):
        # VectorFunctionSpace containing a vector should throw an error
        # rather than segfaulting
        U = VectorFunctionSpace(mesh, ("RT", 2))
        u = ufl.TrialFunction(U)
        v = ufl.TestFunction(U)
        a = form(ufl.inner(u, v) * ufl.dx)
        A = dolfinx.fem.assemble_matrix(a)
        A.finalize()


@pytest.mark.skip_in_parallel
@pytest.mark.parametrize("d1", range(1, 4))
@pytest.mark.parametrize("d2", range(1, 4))
def test_element_product(d1, d2):
    mesh = create_unit_square(MPI.COMM_WORLD, 2, 2)
    P3 = element("Lagrange", mesh.basix_cell(), d1, rank=1)
    P1 = element("Lagrange", mesh.basix_cell(), d2)
    TH = mixed_element([P3, P1])
    W = FunctionSpace(mesh, TH)

    u = ufl.TrialFunction(W)
    v = ufl.TestFunction(W)
    a = form(ufl.inner(u[0], v[0]) * ufl.dx)
    A = dolfinx.fem.assemble_matrix(a)
    A.finalize()

    W = FunctionSpace(mesh, P3)
    u = ufl.TrialFunction(W)
    v = ufl.TestFunction(W)
    a = form(ufl.inner(u[0], v[0]) * ufl.dx)
    B = dolfinx.fem.assemble_matrix(a)
    B.finalize()

    assert np.isclose(A.squared_norm(), B.squared_norm())
