# Copyright (C) 2021 Matthew W. Scroggs and Jack Hale
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Test integral involving + and -."""

from mpi4py import MPI

import pytest
import dolfinx
from dolfinx import UnitSquareMesh, FunctionSpace
from dolfinx.cpp.mesh import CellType

import ufl
from ufl import FiniteElement, MixedElement
from ufl import grad, inner


@pytest.mark.parametrize("N", [1, 2, 3])
def test_plus_minus(N):
    mesh = UnitSquareMesh(MPI.COMM_WORLD, N, N, CellType.triangle,
                          dolfinx.cpp.mesh.GhostMode.shared_facet)

    U_el = MixedElement([FiniteElement("Lagrange", ufl.triangle, 1),
                         FiniteElement("N1curl", ufl.triangle, 1)])
    U = FunctionSpace(mesh, U_el)

    u, p = ufl.TrialFunctions(U)
    v, q = ufl.TestFunctions(U)

    dSp = ufl.Measure('dS', metadata={'quadrature_degree': 1})

    n = ufl.FacetNormal(mesh)
    t = ufl.as_vector((-n[1], n[0]))

    def inner_e0(x, y):
        return (inner(x, t) * inner(y, t))("+") * dSp

    def inner_e1(x, y):
        return (inner(x, t) * inner(y, t))("-") * dSp

    def inner_e2(x, y):
        return inner_e0(x, y) + inner_e1(x, y)

    norms = []
    for inner_e in [inner_e0, inner_e1, inner_e2]:
        a = inner_e(grad(u), q) + inner_e(grad(v), p)

        A = dolfinx.fem.assemble_matrix(a)
        A.assemble()
        print(A.norm())
        norms.append(A.norm())

    assert norms[0] + norms[1] == norms[2]
