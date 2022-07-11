# Copyright (C) 2022 Matthew W. Scroggs
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import pytest

import dolfinx
import ufl
from dolfinx.fem import FunctionSpace, form
from dolfinx.mesh import (CellType, GhostMode, create_unit_cube,
                          create_unit_square, create_unit_interval)

from mpi4py import MPI


@pytest.mark.parametrize("cell", ["interval", "triangle", "tetrahedron", "quadrilateral", "hexahedron"])
def test_real_element(cell):
    ufl_cell = getattr(ufl, cell)
    dolfinx_cell = getattr(CellType, cell)
    if cell == "interval":
        mesh = create_unit_interval(MPI.COMM_WORLD, 8, GhostMode.shared_facet)
    elif cell.endswith("hedron"):
        mesh = create_unit_cube(MPI.COMM_WORLD, 8, 4, 2, dolfinx_cell, GhostMode.shared_facet)
    else:
        mesh = create_unit_square(MPI.COMM_WORLD, 8, 4, dolfinx_cell, GhostMode.shared_facet)

    element = ufl.FiniteElement("R", ufl_cell, 0)
    U = FunctionSpace(mesh, element)
    u = ufl.TrialFunction(U)
    v = ufl.TestFunction(U)

    a = form(ufl.inner(u, v) * ufl.dx)
    A = dolfinx.fem.petsc.assemble_matrix(a)
    A.assemble()
    assert A.getSize()[0] == A.getSize()[1] == 1

    a = form(v * ufl.dx)
    L = dolfinx.fem.petsc.assemble_vector(a)
    assert L.getSize() == 1


@pytest.mark.parametrize("cell", ["interval", "triangle", "tetrahedron", "quadrilateral", "hexahedron"])
def test_mixed_real_element(cell):
    ufl_cell = getattr(ufl, cell)
    dolfinx_cell = getattr(CellType, cell)
    if cell == "interval":
        mesh = create_unit_interval(MPI.COMM_WORLD, 8, GhostMode.shared_facet)
    elif cell.endswith("hedron"):
        mesh = create_unit_cube(MPI.COMM_WORLD, 8, 4, 2, dolfinx_cell, GhostMode.shared_facet)
    else:
        mesh = create_unit_square(MPI.COMM_WORLD, 8, 4, dolfinx_cell, GhostMode.shared_facet)

    lagrange = ufl.FiniteElement("P", ufl_cell, 1)
    real = ufl.FiniteElement("R", ufl_cell, 0)
    element = ufl.MixedElement(lagrange, real)
    U = FunctionSpace(mesh, element)
    U2 = FunctionSpace(mesh, lagrange)
    u, s = ufl.TrialFunctions(U)
    v, t = ufl.TestFunctions(U)
    u2 = ufl.TrialFunction(U2)
    v2 = ufl.TestFunction(U2)

    a = form(ufl.inner(u, v) * ufl.dx + ufl.inner(s, t) * ufl.dx)
    A = dolfinx.fem.petsc.assemble_matrix(a)
    A.assemble()
    a2 = form(ufl.inner(u2, v2) * ufl.dx)
    A2 = dolfinx.fem.petsc.assemble_matrix(a2)
    A2.assemble()
    assert A.getSize()[0] == A.getSize()[1] == A2.getSize()[0] + 1

    a = form(v * ufl.dx + t * ufl.dx)
    A = dolfinx.fem.petsc.assemble_vector(a)
    a2 = form(v2 * ufl.dx)
    A2 = dolfinx.fem.petsc.assemble_vector(a2)
    assert len(A.array) == len(A2.array) + 1


@pytest.mark.parametrize("cell", ["interval", "triangle", "tetrahedron", "quadrilateral", "hexahedron"])
def test_vector_real_element(cell):
    ufl_cell = getattr(ufl, cell)
    dolfinx_cell = getattr(CellType, cell)
    if cell == "interval":
        mesh = create_unit_interval(MPI.COMM_WORLD, 8, GhostMode.shared_facet)
        dim = 1
    elif cell.endswith("hedron"):
        mesh = create_unit_cube(MPI.COMM_WORLD, 8, 4, 2, dolfinx_cell, GhostMode.shared_facet)
        dim = 3
    else:
        mesh = create_unit_square(MPI.COMM_WORLD, 8, 4, dolfinx_cell, GhostMode.shared_facet)
        dim = 2

    real = ufl.FiniteElement("R", ufl_cell, 0)
    element = ufl.VectorElement(real)
    U = FunctionSpace(mesh, element)
    u = ufl.TrialFunction(U)
    v = ufl.TestFunction(U)

    a = form(ufl.inner(u, v) * ufl.dx)
    A = dolfinx.fem.petsc.assemble_matrix(a)
    A.assemble()
    assert A.getSize()[0] == A.getSize()[1] == dim


#@pytest.mark.parametrize("cell", ["interval", "triangle", "tetrahedron", "quadrilateral", "hexahedron"])
#def test_mean_constraint():
#    pass
