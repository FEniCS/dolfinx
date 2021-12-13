# Copyright (C) 2021 Matthew W. Scroggs and Jack Hale
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import numpy as np
import pytest

import dolfinx
import ufl
from dolfinx.fem import FunctionSpace, VectorFunctionSpace
from dolfinx.mesh import (CellType, GhostMode, create_unit_cube_mesh,
                          create_unit_square_mesh)
from dolfinx_utils.test.skips import skip_in_parallel

from mpi4py import MPI


@skip_in_parallel
@pytest.mark.parametrize("cell", [ufl.triangle, ufl.tetrahedron])
@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("ElementType, space", [
    (ufl.FiniteElement, "Lagrange"),
    (ufl.VectorElement, "Lagrange"),
    (ufl.FiniteElement, "N1curl")
])
def test_mixed_element(ElementType, space, cell, order):
    if cell == ufl.triangle:
        mesh = create_unit_square_mesh(MPI.COMM_WORLD, 1, 1, CellType.triangle, GhostMode.shared_facet)
    else:
        mesh = create_unit_cube_mesh(MPI.COMM_WORLD, 1, 1, 1, CellType.tetrahedron, GhostMode.shared_facet)

    norms = []
    U_el = ufl.FiniteElement(space, cell, order)
    for i in range(3):
        U = FunctionSpace(mesh, U_el)

        u = ufl.TrialFunction(U)
        v = ufl.TestFunction(U)

        a = ufl.inner(u, v) * ufl.dx

        A = dolfinx.fem.assemble_matrix(a)
        A.assemble()
        norms.append(A.norm())

        U_el = ufl.MixedElement(U_el)

    for i in norms[1:]:
        assert np.isclose(norms[0], i)


@skip_in_parallel
def test_vector_element():
    # VectorFunctionSpace containing a scalar should work
    mesh = create_unit_square_mesh(MPI.COMM_WORLD, 1, 1, CellType.triangle, GhostMode.shared_facet)
    U = VectorFunctionSpace(mesh, ("P", 2))
    u = ufl.TrialFunction(U)
    v = ufl.TestFunction(U)

    a = ufl.inner(u, v) * ufl.dx

    A = dolfinx.fem.assemble_matrix(a)
    A.assemble()

    with pytest.raises(ValueError):
        # VectorFunctionSpace containing a vector should throw an error rather than segfaulting
        U = VectorFunctionSpace(mesh, ("RT", 2))

        u = ufl.TrialFunction(U)
        v = ufl.TestFunction(U)

        a = ufl.inner(u, v) * ufl.dx

        A = dolfinx.fem.assemble_matrix(a)
        A.assemble()
