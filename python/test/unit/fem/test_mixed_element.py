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
def test_mixed_element(ElementType, space, cell, order):
    if cell == ufl.triangle:
        mesh = UnitSquareMesh(MPI.COMM_WORLD, 1, 1, CellType.triangle,
                              dolfinx.cpp.mesh.GhostMode.shared_facet)
    else:
        mesh = UnitCubeMesh(MPI.COMM_WORLD, 1, 1, 1, CellType.tetrahedron,
                            dolfinx.cpp.mesh.GhostMode.shared_facet)

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

        # TODO: remove these lines once MixedElement(MixedElement(Nedelec)) is fixed
        if space == "N1curl" and i == 1:
            break

        U_el = ufl.MixedElement(U_el)

    for i in norms[1:]:
        assert np.isclose(norms[0], i)
