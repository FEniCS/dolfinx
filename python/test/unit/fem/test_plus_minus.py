# Copyright (C) 2021 Matthew W. Scroggs and Jack Hale
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Test integral involving + and -."""

from mpi4py import MPI
import numpy as np

import pytest
import dolfinx
from dolfinx import UnitSquareMesh, FunctionSpace
from dolfinx.cpp.mesh import CellType

import ufl
from ufl import FiniteElement, MixedElement
from ufl import grad, inner


@pytest.mark.parametrize("N", [1, 2, 3])
@pytest.mark.parametrize("sign", ["+", "-"])
def test_symmetry(N, sign):
    mesh = UnitSquareMesh(MPI.COMM_WORLD, N, N, CellType.triangle,
                          dolfinx.cpp.mesh.GhostMode.shared_facet)

    U_el = MixedElement([FiniteElement("Lagrange", ufl.triangle, 1),
                         FiniteElement("N1curl", ufl.triangle, 1)])
    U = FunctionSpace(mesh, U_el)

    u, p = ufl.TrialFunctions(U)
    v, q = ufl.TestFunctions(U)

    dSp = ufl.dS

    n = ufl.FacetNormal(mesh)
    t = ufl.as_vector((-n[1], n[0]))

    def inner_e(x, y):
        return (inner(x, t) * inner(y, t))(sign) * dSp

    a = inner_e(grad(u), q) + inner_e(grad(v), p)

    A = dolfinx.fem.assemble_matrix(a)
    A.assemble()
    print(A.norm())

    for i in range(A.size[0]):
        print([float(A[i, j]) for j in range(A.size[1])])

    for i in range(A.size[0]):
        for j in range(A.size[1]):
            assert np.isclose(A[i, j], A[j, i])


@pytest.mark.parametrize("N", [1, 2, 3])
def test_plus_minus_matrix(N):
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

    results = []
    for inner_e in [inner_e0, inner_e1, inner_e2]:
        a = inner_e(grad(u), q) + inner_e(grad(v), p)

        A = dolfinx.fem.assemble_matrix(a)
        A.assemble()
        print(A.norm())
        results.append(A)

    assert (results[0] + results[1]).norm() == results[2].norm()

    for i in range(results[0].size[0]):
        print([float(results[0][i, j]) for j in range(results[0].size[1])])
    print()
    for i in range(results[1].size[1]):
        print([float(results[1][i, j]) for j in range(results[1].size[1])])

    for i in range(results[0].size[0]):
        for j in range(results[0].size[1]):
            assert np.isclose((results[0] + results[1])[i, j], results[2][i, j])
            assert np.isclose(results[0][i, j], results[1][i, j])


@pytest.mark.parametrize("N", [1, 2, 3])
def test_plus_minus_vector2(N):
    mesh = UnitSquareMesh(MPI.COMM_WORLD, N, N, CellType.triangle,
                          dolfinx.cpp.mesh.GhostMode.shared_facet)

    element = FiniteElement("Lagrange", ufl.triangle, 1)
    space = FunctionSpace(mesh, element)

    v = ufl.TestFunction(space)
    n = ufl.FacetNormal(mesh)
    t = ufl.as_vector((-n[1], n[0]))

    results = []
    for sign in ["+", "-"]:
        form = t[0](sign) * ufl.dS
        print(dolfinx.fem.assemble_scalar(form))

        form = inner(grad(v), t)(sign) * ufl.dS
        results.append(dolfinx.fem.assemble_vector(form))
        print(results[-1][:])

    assert np.allclose(results[0], -results[1])
