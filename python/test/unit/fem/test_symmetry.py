# Copyright (C) 2021 Matthew W. Scroggs and Jack Hale
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Test that matrices are symmetric."""

from mpi4py import MPI
import numpy as np

import pytest
import dolfinx
from dolfinx import UnitSquareMesh, FunctionSpace
from dolfinx.cpp.mesh import CellType

import ufl
from ufl import FiniteElement, MixedElement
from ufl import grad, inner


def run_symmetry_test(form):
    A = dolfinx.fem.assemble_matrix(form)
    A.assemble()

    for i in range(A.size[0]):
        for j in range(A.size[1]):
            assert np.isclose(A[i, j], A[j, i])


@pytest.mark.parametrize("N", [1, 2, 3])
@pytest.mark.parametrize("element", ["Lagrange", "N1curl"])
@pytest.mark.parametrize("order", range(1, 4))
@pytest.mark.parametrize("sign", ["+", "-"])
def test_mass_matrix_dS(N, element, order, sign):
    mesh = UnitSquareMesh(MPI.COMM_WORLD, N, N, CellType.triangle,
                          dolfinx.cpp.mesh.GhostMode.shared_facet)

    space = FunctionSpace(mesh, (element, order))

    u = ufl.TrialFunction(space)
    v = ufl.TestFunction(space)

    run_symmetry_test(inner(u, v)(sign) * ufl.dS)


@pytest.mark.parametrize("N", [1, 2, 3])
@pytest.mark.parametrize("sign", ["+", "-"])
def test_facet_normal(N, sign):
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

    run_symmetry_test(inner_e(grad(u), q) + inner_e(grad(v), p))


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
        results.append(A)

    assert (results[0] + results[1]).norm() == results[2].norm()

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

        form = inner(grad(v), t)(sign) * ufl.dS
        results.append(dolfinx.fem.assemble_vector(form))

    assert np.allclose(results[0], -results[1])
