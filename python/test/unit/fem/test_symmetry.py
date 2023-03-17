# Copyright (C) 2021 Matthew W. Scroggs and Jack Hale
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Test that matrices are symmetric."""

import pytest

import basix
import dolfinx
import ufl
from basix.ufl import MixedElement, create_element, create_vector_element
from dolfinx.fem import FunctionSpace, form
from dolfinx.mesh import CellType, create_unit_cube, create_unit_square
from ufl import grad, inner

from mpi4py import MPI


def check_symmetry(A):
    assert A.isSymmetric(1e-8)


def run_symmetry_test(cell_type, element, form_f):
    if cell_type == CellType.triangle or cell_type == CellType.quadrilateral:
        mesh = create_unit_square(MPI.COMM_WORLD, 2, 2, cell_type)
    else:
        mesh = create_unit_cube(MPI.COMM_WORLD, 2, 2, 2, cell_type)

    space = FunctionSpace(mesh, element)
    u = ufl.TrialFunction(space)
    v = ufl.TestFunction(space)
    f = form(form_f(u, v))

    A = dolfinx.fem.petsc.assemble_matrix(f)
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


@pytest.mark.skip_in_parallel
@parametrize_elements
@pytest.mark.parametrize("order", range(1, 2))
def test_mass_matrix_dx(cell_type, element, order):
    run_symmetry_test(cell_type, (element, order),
                      lambda u, v: inner(u, v) * ufl.dx)


@pytest.mark.skip_in_parallel
@parametrize_lagrange_elements
@pytest.mark.parametrize("order", range(1, 2))
def test_stiffness_matrix_dx(cell_type, element, order):
    run_symmetry_test(cell_type, (element, order),
                      lambda u, v: inner(grad(u), grad(v)) * ufl.dx)


@pytest.mark.skip_in_parallel
@parametrize_elements
@pytest.mark.parametrize("order", range(1, 2))
def test_mass_matrix_ds(cell_type, element, order):
    run_symmetry_test(cell_type, (element, order),
                      lambda u, v: inner(u, v) * ufl.ds)


@pytest.mark.skip_in_parallel
@parametrize_lagrange_elements
@pytest.mark.parametrize("order", range(1, 2))
def test_stiffness_matrix_ds(cell_type, element, order):
    run_symmetry_test(cell_type, (element, order),
                      lambda u, v: inner(grad(u), grad(v)) * ufl.ds)


@pytest.mark.skip_in_parallel
@parametrize_elements
@pytest.mark.parametrize("order", range(1, 2))
@pytest.mark.parametrize("sign", ["+", "-"])
def test_mass_matrix_dS(cell_type, element, order, sign):
    run_symmetry_test(cell_type, (element, order),
                      lambda u, v: inner(u, v)(sign) * ufl.dS)


@pytest.mark.skip_in_parallel
@parametrize_lagrange_elements
@pytest.mark.parametrize("order", range(1, 2))
@pytest.mark.parametrize("sign", ["+", "-"])
def test_stiffness_matrix_dS(cell_type, element, order, sign):
    run_symmetry_test(cell_type, (element, order),
                      lambda u, v: inner(grad(u), grad(v))(sign) * ufl.dS)


@pytest.mark.skip_in_parallel
@pytest.mark.parametrize("cell_type", [CellType.triangle, CellType.quadrilateral,
                                       CellType.tetrahedron, CellType.hexahedron])
@pytest.mark.parametrize("sign", ["+", "-"])
@pytest.mark.parametrize("order", range(1, 2))
def test_mixed_element_form(cell_type, sign, order):
    if cell_type == CellType.triangle or cell_type == CellType.quadrilateral:
        mesh = create_unit_square(MPI.COMM_WORLD, 2, 2, cell_type)
    else:
        mesh = create_unit_cube(MPI.COMM_WORLD, 2, 2, 2, cell_type)

    U_el = MixedElement([
        create_element(basix.ElementFamily.P, cell_type.name, order),
        create_element(basix.ElementFamily.N1E, cell_type.name, order)])

    U = FunctionSpace(mesh, U_el)
    u, p = ufl.TrialFunctions(U)
    v, q = ufl.TestFunctions(U)
    f = form(inner(u, v) * ufl.dx + inner(p, q)(sign) * ufl.dS)

    A = dolfinx.fem.petsc.assemble_matrix(f)
    A.assemble()
    check_symmetry(A)


@pytest.mark.skip_in_parallel
@pytest.mark.parametrize("cell_type", [CellType.triangle, CellType.quadrilateral])
@pytest.mark.parametrize("sign", ["+", "-"])
@pytest.mark.parametrize("order", range(1, 2))
def test_mixed_element_vector_element_form(cell_type, sign, order):
    if cell_type == CellType.triangle or cell_type == CellType.quadrilateral:
        mesh = create_unit_square(MPI.COMM_WORLD, 2, 2, cell_type)
    else:
        mesh = create_unit_cube(MPI.COMM_WORLD, 2, 2, 2, cell_type)

    U_el = MixedElement([
        create_vector_element(basix.ElementFamily.P, cell_type.name, order),
        create_element(basix.ElementFamily.N1E, cell_type.name, order)])

    U = FunctionSpace(mesh, U_el)
    u, p = ufl.TrialFunctions(U)
    v, q = ufl.TestFunctions(U)
    f = form(inner(u, v) * ufl.dx + inner(p, q)(sign) * ufl.dS)

    A = dolfinx.fem.petsc.assemble_matrix(f)
    A.assemble()

    check_symmetry(A)
