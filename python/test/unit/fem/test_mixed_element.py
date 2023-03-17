# Copyright (C) 2021 Matthew W. Scroggs and Jack Hale
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import numpy as np
import pytest

import dolfinx
import ufl
from basix.ufl import (MixedElement, create_element,
                       create_vector_element)
from dolfinx.fem import FunctionSpace, VectorFunctionSpace, form
from dolfinx.mesh import (CellType, GhostMode, create_unit_cube,
                          create_unit_square)

from mpi4py import MPI


@pytest.mark.skip_in_parallel
@pytest.mark.parametrize("cell", [ufl.triangle, ufl.tetrahedron])
@pytest.mark.parametrize("degree", [1, 2])
@pytest.mark.parametrize("create_element_function, family",
                         [
                             (create_element, "Lagrange"),
                             (create_vector_element, "Lagrange"),
                             (create_element, "N1curl")
                         ])
def test_mixed_element(create_element_function, family, cell, degree):
    if cell == ufl.triangle:
        mesh = create_unit_square(MPI.COMM_WORLD, 1, 1, CellType.triangle, GhostMode.shared_facet)
    else:
        mesh = create_unit_cube(MPI.COMM_WORLD, 1, 1, 1, CellType.tetrahedron, GhostMode.shared_facet)

    norms = []
    U_el = create_element_function(family, cell.cellname(), degree)
    for i in range(3):
        U = FunctionSpace(mesh, U_el)
        u = ufl.TrialFunction(U)
        v = ufl.TestFunction(U)
        a = form(ufl.inner(u, v) * ufl.dx)

        A = dolfinx.fem.petsc.assemble_matrix(a)
        A.assemble()
        norms.append(A.norm())
        A.destroy()

        U_el = MixedElement([U_el])

    for i in norms[1:]:
        assert np.isclose(norms[0], i)


@pytest.mark.skip_in_parallel
def test_vector_element():
    # VectorFunctionSpace containing a scalar should work
    mesh = create_unit_square(MPI.COMM_WORLD, 1, 1, CellType.triangle, GhostMode.shared_facet)
    U = VectorFunctionSpace(mesh, ("P", 2))
    u = ufl.TrialFunction(U)
    v = ufl.TestFunction(U)
    a = form(ufl.inner(u, v) * ufl.dx)
    A = dolfinx.fem.petsc.assemble_matrix(a)
    A.assemble()
    A.destroy()

    with pytest.raises(ValueError):
        # VectorFunctionSpace containing a vector should throw an error
        # rather than segfaulting
        U = VectorFunctionSpace(mesh, ("RT", 2))
        u = ufl.TrialFunction(U)
        v = ufl.TestFunction(U)
        a = form(ufl.inner(u, v) * ufl.dx)
        A = dolfinx.fem.petsc.assemble_matrix(a)
        A.assemble()
        A.destroy()


@pytest.mark.skip_in_parallel
@pytest.mark.parametrize("d1", range(1, 4))
@pytest.mark.parametrize("d2", range(1, 4))
def test_element_product(d1, d2):
    mesh = create_unit_square(MPI.COMM_WORLD, 2, 2)
    P3 = create_vector_element("Lagrange", mesh.ufl_cell().cellname(), d1)
    P1 = create_element("Lagrange", mesh.ufl_cell().cellname(), d2)
    TH = MixedElement([P3, P1])
    W = FunctionSpace(mesh, TH)

    u = ufl.TrialFunction(W)
    v = ufl.TestFunction(W)
    a = form(ufl.inner(u[0], v[0]) * ufl.dx)
    A = dolfinx.fem.petsc.assemble_matrix(a)
    A.assemble()

    W = FunctionSpace(mesh, P3)
    u = ufl.TrialFunction(W)
    v = ufl.TestFunction(W)
    a = form(ufl.inner(u[0], v[0]) * ufl.dx)
    B = dolfinx.fem.petsc.assemble_matrix(a)
    B.assemble()

    assert np.isclose(A.norm(), B.norm())

    A.destroy()
    B.destroy()
