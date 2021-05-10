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
from dolfinx import UnitSquareMesh, UnitCubeMesh, FunctionSpace
from dolfinx.cpp.mesh import CellType
from dolfinx_utils.test.skips import skip_in_parallel


@skip_in_parallel
@pytest.mark.parametrize("cell", [ufl.triangle, ufl.tetrahedron])
@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("ElementType, space", [
    (ufl.FiniteElement, "Lagrange"),
    (ufl.VectorElement, "Lagrange"),
    (ufl.FiniteElement, "N1curl")
])
def test_mixed_nedelec(ElementType, space, cell, order):
    if cell == ufl.triangle:
        mesh = UnitSquareMesh(MPI.COMM_WORLD, 1, 1, CellType.triangle,
                              dolfinx.cpp.mesh.GhostMode.shared_facet)
    else:
        mesh = UnitCubeMesh(MPI.COMM_WORLD, 1, 1, 1, CellType.tetrahedron,
                            dolfinx.cpp.mesh.GhostMode.shared_facet)

    U_el = ElementType(space, cell, order)
    U = FunctionSpace(mesh, U_el)

    u = ufl.TrialFunction(U)
    v = ufl.TestFunction(U)

    a = ufl.inner(u, v) * ufl.dx

    A = dolfinx.fem.assemble_matrix(a)
    A.assemble()
    norm1 = A.norm()

    U_el_mixed = ufl.MixedElement([ElementType(space, cell, order)])
    U = FunctionSpace(mesh, U_el_mixed)

    u = ufl.TrialFunction(U)
    v = ufl.TestFunction(U)

    a = ufl.inner(u, v) * ufl.dx

    A = dolfinx.fem.assemble_matrix(a)
    A.assemble()

    norm2 = A.norm()

    assert np.isclose(norm1, norm2)
