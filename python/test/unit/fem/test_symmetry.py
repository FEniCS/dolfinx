# Copyright (C) 2021 Matthew W. Scroggs and Jack Hale
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Test that matrices are symmetric."""

import pytest

import dolfinx
import ufl
from dolfinx.fem import FunctionSpace
from dolfinx.mesh import (CellType, create_unit_cube_mesh,
                          create_unit_square_mesh)
from dolfinx_utils.test.skips import skip_in_parallel
from ufl import FiniteElement, MixedElement, VectorElement, grad, inner

from mpi4py import MPI


def check_symmetry(A):
    assert A.isSymmetric(1e-8)


def run_symmetry_test(cell_type, element, form_f):
    if cell_type == CellType.triangle or cell_type == CellType.quadrilateral:
        mesh = create_unit_square_mesh(MPI.COMM_WORLD, 2, 2, cell_type)
    else:
        mesh = create_unit_cube_mesh(MPI.COMM_WORLD, 2, 2, 2, cell_type)

    space = FunctionSpace(mesh, element)

    u = ufl.TrialFunction(space)
    v = ufl.TestFunction(space)

    form = form_f(u, v)

    A = dolfinx.fem.assemble_matrix(form)
    A.assemble()
    check_symmetry(A)


parametrize_elements = pytest.mark.parametrize("cell_type, element", [
    (CellType.triangle, "Lagrange"), (CellType.triangle, "N1curl"), (CellType.triangle, "RT"),
    (CellType.triangle, "Regge"),
    (CellType.quadrilateral, "Lagrange"), (CellType.quadrilateral, "RTCE"), (CellType.quadrilateral, "RTCF"),
    (CellType.tetrahedron, "Lagrange"), (CellType.tetrahedron, "N1curl"), (CellType.tetrahedron, "RT"),
    (CellType.tetrahedron, "Regge"),
    (CellType.hexahedron, "Lagrange"), (CellType.hexahedron, "NCE"), (CellType.hexahedron, "NCF")
])
parametrize_lagrange_elements = pytest.mark.parametrize("cell_type, element", [
    (CellType.triangle, "Lagrange"), (CellType.quadrilateral, "Lagrange"),
    (CellType.tetrahedron, "Lagrange"), (CellType.hexahedron, "Lagrange")
])


@skip_in_parallel
@parametrize_elements
@pytest.mark.parametrize("order", range(1, 2))
def test_mass_matrix_dx(cell_type, element, order):
    run_symmetry_test(cell_type, (element, order),
                      lambda u, v: inner(u, v) * ufl.dx)


@skip_in_parallel
@parametrize_lagrange_elements
@pytest.mark.parametrize("order", range(1, 2))
def test_stiffness_matrix_dx(cell_type, element, order):
    run_symmetry_test(cell_type, (element, order),
                      lambda u, v: inner(grad(u), grad(v)) * ufl.dx)


@skip_in_parallel
@parametrize_elements
@pytest.mark.parametrize("order", range(1, 2))
def test_mass_matrix_ds(cell_type, element, order):
    run_symmetry_test(cell_type, (element, order),
                      lambda u, v: inner(u, v) * ufl.ds)


@skip_in_parallel
@parametrize_lagrange_elements
@pytest.mark.parametrize("order", range(1, 2))
def test_stiffness_matrix_ds(cell_type, element, order):
    run_symmetry_test(cell_type, (element, order),
                      lambda u, v: inner(grad(u), grad(v)) * ufl.ds)


@skip_in_parallel
@parametrize_elements
@pytest.mark.parametrize("order", range(1, 2))
@pytest.mark.parametrize("sign", ["+", "-"])
def test_mass_matrix_dS(cell_type, element, order, sign):
    run_symmetry_test(cell_type, (element, order),
                      lambda u, v: inner(u, v)(sign) * ufl.dS)


@skip_in_parallel
@parametrize_lagrange_elements
@pytest.mark.parametrize("order", range(1, 2))
@pytest.mark.parametrize("sign", ["+", "-"])
def test_stiffness_matrix_dS(cell_type, element, order, sign):
    run_symmetry_test(cell_type, (element, order),
                      lambda u, v: inner(grad(u), grad(v))(sign) * ufl.dS)


@skip_in_parallel
@pytest.mark.parametrize("cell_type", [CellType.triangle, CellType.quadrilateral,
                                       CellType.tetrahedron, CellType.hexahedron])
@pytest.mark.parametrize("sign", ["+", "-"])
@pytest.mark.parametrize("order", range(1, 2))
def test_mixed_element_form(cell_type, sign, order):
    if cell_type == CellType.triangle or cell_type == CellType.quadrilateral:
        mesh = create_unit_square_mesh(MPI.COMM_WORLD, 2, 2, cell_type)
    else:
        mesh = create_unit_cube_mesh(MPI.COMM_WORLD, 2, 2, 2, cell_type)

    if cell_type == CellType.triangle:
        U_el = MixedElement([FiniteElement("Lagrange", ufl.triangle, order),
                             FiniteElement("N1curl", ufl.triangle, order)])
    elif cell_type == CellType.quadrilateral:
        U_el = MixedElement([FiniteElement("Lagrange", ufl.quadrilateral, order),
                             FiniteElement("RTCE", ufl.quadrilateral, order)])
    elif cell_type == CellType.tetrahedron:
        U_el = MixedElement([FiniteElement("Lagrange", ufl.tetrahedron, order),
                             FiniteElement("N1curl", ufl.tetrahedron, order)])
    elif cell_type == CellType.hexahedron:
        U_el = MixedElement([FiniteElement("Lagrange", ufl.hexahedron, order),
                             FiniteElement("NCE", ufl.hexahedron, order)])

    U = FunctionSpace(mesh, U_el)

    u, p = ufl.TrialFunctions(U)
    v, q = ufl.TestFunctions(U)

    form = inner(u, v) * ufl.dx + inner(p, q)(sign) * ufl.dS

    A = dolfinx.fem.assemble_matrix(form)
    A.assemble()
    check_symmetry(A)


@skip_in_parallel
@pytest.mark.parametrize("cell_type", [CellType.triangle, CellType.quadrilateral])
@pytest.mark.parametrize("sign", ["+", "-"])
@pytest.mark.parametrize("order", range(1, 2))
def test_mixed_element_vector_element_form(cell_type, sign, order):
    if cell_type == CellType.triangle or cell_type == CellType.quadrilateral:
        mesh = create_unit_square_mesh(MPI.COMM_WORLD, 2, 2, cell_type)
    else:
        mesh = create_unit_cube_mesh(MPI.COMM_WORLD, 2, 2, 2, cell_type)

    if cell_type == CellType.triangle:
        U_el = MixedElement([VectorElement("Lagrange", ufl.triangle, order),
                             FiniteElement("N1curl", ufl.triangle, order)])
    elif cell_type == CellType.quadrilateral:
        U_el = MixedElement([VectorElement("Lagrange", ufl.quadrilateral, order),
                             FiniteElement("RTCE", ufl.quadrilateral, order)])
    elif cell_type == CellType.tetrahedron:
        U_el = MixedElement([VectorElement("Lagrange", ufl.tetrahedron, order),
                             FiniteElement("N1curl", ufl.tetrahedron, order)])
    elif cell_type == CellType.hexahedron:
        U_el = MixedElement([VectorElement("Lagrange", ufl.hexahedron, order),
                             FiniteElement("NCE", ufl.hexahedron, order)])

    U = FunctionSpace(mesh, U_el)

    u, p = ufl.TrialFunctions(U)
    v, q = ufl.TestFunctions(U)

    form = inner(u, v) * ufl.dx + inner(p, q)(sign) * ufl.dS

    A = dolfinx.fem.assemble_matrix(form)
    A.assemble()

    check_symmetry(A)
