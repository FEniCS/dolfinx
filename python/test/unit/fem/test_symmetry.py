# Copyright (C) 2021 Matthew W. Scroggs and Jack Hale
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Test that matrices are symmetric."""

import basix
import numpy as np
import pytest
import ufl
from basix.ufl import element, mixed_element
from dolfinx.fem import FunctionSpace, form
from dolfinx.mesh import CellType, create_unit_cube, create_unit_square
from mpi4py import MPI
from petsc4py import PETSc
from ufl import grad, inner

import dolfinx


def check_symmetry(A, tol):
    assert A.isSymmetric(tol)


def run_symmetry_test(cell_type, e, form_f):
    if cell_type == CellType.triangle or cell_type == CellType.quadrilateral:
        mesh = create_unit_square(MPI.COMM_WORLD, 2, 2, cell_type)
    else:
        mesh = create_unit_cube(MPI.COMM_WORLD, 2, 2, 2, cell_type)

    space = FunctionSpace(mesh, e)
    u = ufl.TrialFunction(space)
    v = ufl.TestFunction(space)
    f = form(form_f(u, v))

    A = dolfinx.fem.petsc.assemble_matrix(f)
    A.assemble()
    tol = np.sqrt(np.finfo(PETSc.RealType).eps)
    check_symmetry(A, tol)


parametrize_elements = pytest.mark.parametrize("cell_type, family", [
    (CellType.triangle, "Lagrange"), (CellType.triangle, "N1curl"), (CellType.triangle, "RT"),
    (CellType.triangle, "Regge"),
    (CellType.quadrilateral, "Lagrange"), (CellType.quadrilateral, "RTCE"), (CellType.quadrilateral, "RTCF"),
    (CellType.tetrahedron, "Lagrange"), (CellType.tetrahedron, "N1curl"), (CellType.tetrahedron, "RT"),
    (CellType.tetrahedron, "Regge"),
    (CellType.hexahedron, "Lagrange"), (CellType.hexahedron, "NCE"), (CellType.hexahedron, "NCF")
])
parametrize_lagrange_elements = pytest.mark.parametrize("cell_type, family", [
    (CellType.triangle, "Lagrange"), (CellType.quadrilateral, "Lagrange"),
    (CellType.tetrahedron, "Lagrange"), (CellType.hexahedron, "Lagrange")
])


@pytest.mark.skip_in_parallel
@parametrize_elements
@pytest.mark.parametrize("order", range(1, 2))
def test_mass_matrix_dx(cell_type, family, order):
    run_symmetry_test(cell_type, (family, order), lambda u, v: inner(u, v) * ufl.dx)


@pytest.mark.skip_in_parallel
@parametrize_lagrange_elements
@pytest.mark.parametrize("order", range(1, 2))
def test_stiffness_matrix_dx(cell_type, family, order):
    run_symmetry_test(cell_type, (family, order), lambda u, v: inner(grad(u), grad(v)) * ufl.dx)


@pytest.mark.skip_in_parallel
@parametrize_elements
@pytest.mark.parametrize("order", range(1, 2))
def test_mass_matrix_ds(cell_type, family, order):
    run_symmetry_test(cell_type, (family, order), lambda u, v: inner(u, v) * ufl.ds)


@pytest.mark.skip_in_parallel
@parametrize_lagrange_elements
@pytest.mark.parametrize("order", range(1, 2))
def test_stiffness_matrix_ds(cell_type, family, order):
    run_symmetry_test(cell_type, (family, order), lambda u, v: inner(grad(u), grad(v)) * ufl.ds)


@pytest.mark.skip_in_parallel
@parametrize_elements
@pytest.mark.parametrize("order", range(1, 2))
@pytest.mark.parametrize("sign", ["+", "-"])
def test_mass_matrix_dS(cell_type, family, order, sign):
    run_symmetry_test(cell_type, (family, order), lambda u, v: inner(u, v)(sign) * ufl.dS)


@pytest.mark.skip_in_parallel
@parametrize_lagrange_elements
@pytest.mark.parametrize("order", range(1, 2))
@pytest.mark.parametrize("sign", ["+", "-"])
def test_stiffness_matrix_dS(cell_type, family, order, sign):
    run_symmetry_test(cell_type, (family, order), lambda u, v: inner(grad(u), grad(v))(sign) * ufl.dS)


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

    U_el = mixed_element([element(basix.ElementFamily.P, cell_type.name, order),
                          element(basix.ElementFamily.N1E, cell_type.name, order)])

    U = FunctionSpace(mesh, U_el)
    u, p = ufl.TrialFunctions(U)
    v, q = ufl.TestFunctions(U)
    f = form(inner(u, v) * ufl.dx + inner(p, q)(sign) * ufl.dS)

    A = dolfinx.fem.petsc.assemble_matrix(f)
    A.assemble()
    tol = np.sqrt(np.finfo(PETSc.RealType).eps)
    check_symmetry(A, tol)


@pytest.mark.skip_in_parallel
@pytest.mark.parametrize("cell_type", [CellType.triangle, CellType.quadrilateral])
@pytest.mark.parametrize("sign", ["+", "-"])
@pytest.mark.parametrize("order", range(1, 2))
def test_mixed_element_vector_element_form(cell_type, sign, order):
    if cell_type == CellType.triangle or cell_type == CellType.quadrilateral:
        mesh = create_unit_square(MPI.COMM_WORLD, 2, 2, cell_type)
    else:
        mesh = create_unit_cube(MPI.COMM_WORLD, 2, 2, 2, cell_type)

    U_el = mixed_element([
        element(basix.ElementFamily.P, cell_type.name, order, rank=1),
        element(basix.ElementFamily.N1E, cell_type.name, order)])

    U = FunctionSpace(mesh, U_el)
    u, p = ufl.TrialFunctions(U)
    v, q = ufl.TestFunctions(U)
    f = form(inner(u, v) * ufl.dx + inner(p, q)(sign) * ufl.dS)

    A = dolfinx.fem.petsc.assemble_matrix(f)
    A.assemble()

    tol = np.sqrt(np.finfo(PETSc.RealType).eps)
    check_symmetry(A, tol)
