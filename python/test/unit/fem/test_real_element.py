# Copyright (C) 2022 Matthew W. Scroggs
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import numpy as np
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

    element = ufl.FiniteElement("Real", ufl_cell, 0)
    U = FunctionSpace(mesh, element)
    u = ufl.TrialFunction(U)
    v = ufl.TestFunction(U)

    a = form(ufl.inner(u, v) * ufl.dx)
    A = dolfinx.fem.petsc.assemble_matrix(a)
    A.assemble()
    assert np.isclose(A.convert("dense").getDenseArray(), 1.0)
    assert A.getSize()[0] == A.getSize()[1] == 1

    a = form(v * ufl.dx)
    L = dolfinx.fem.petsc.assemble_vector(a)
    assert np.isclose(L, 1.0)
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

    lagrange = ufl.FiniteElement("Lagrange", ufl_cell, 1)
    real = ufl.FiniteElement("Real", ufl_cell, 0)
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
    assert A.getSize() == A2.getSize() + 1


@pytest.mark.parametrize("cell", ["interval", "triangle", "tetrahedron", "quadrilateral", "hexahedron"])
def test_vector_real_element(cell):
    ufl_cell = getattr(ufl, cell)
    dolfinx_cell = getattr(CellType, cell)
    if cell == "interval":
        mesh = create_unit_interval(MPI.COMM_WORLD, 8, GhostMode.shared_facet)
    elif cell.endswith("hedron"):
        mesh = create_unit_cube(MPI.COMM_WORLD, 8, 4, 2, dolfinx_cell, GhostMode.shared_facet)
    else:
        mesh = create_unit_square(MPI.COMM_WORLD, 8, 4, dolfinx_cell, GhostMode.shared_facet)

    dim = mesh.geometry.dim
    real = ufl.FiniteElement("Real", ufl_cell, 0)
    element = ufl.VectorElement(real)
    U = FunctionSpace(mesh, element)
    u = ufl.TrialFunction(U)
    v = ufl.TestFunction(U)

    a = form(ufl.inner(u, v) * ufl.dx)
    A = dolfinx.fem.petsc.assemble_matrix(a)
    A.assemble()
    A_dense = A.convert("dense").getDenseArray()
    assert A.getSize()[0] == A.getSize()[1] == dim
    assert np.isclose(A_dense, np.identity(dim)).all()


@pytest.mark.parametrize("cell", ["interval", "triangle", "tetrahedron", "quadrilateral", "hexahedron"])
def test_real_as_scalar_coefficient(cell):
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

    real_element = ufl.FiniteElement("Real", ufl_cell, 0)
    Vr = FunctionSpace(mesh, real_element)
    r = dolfinx.fem.Function(Vr)
    r.x.array[:] = 6.0

    dx = ufl.Measure("dx", mesh)
    scaled_area = r * r * dx
    A = dolfinx.fem.assemble_scalar(form(scaled_area))
    assert(np.isclose(6.0**2, A))


@pytest.mark.parametrize("cell", ["interval", "triangle", "tetrahedron", "quadrilateral", "hexahedron"])
def test_real_as_vector_coefficient(cell):
    ufl_cell = getattr(ufl, cell)
    dolfinx_cell = getattr(CellType, cell)
    if cell == "interval":
        mesh = create_unit_interval(MPI.COMM_WORLD, 8, GhostMode.shared_facet)
    elif cell.endswith("hedron"):
        mesh = create_unit_cube(MPI.COMM_WORLD, 8, 4, 2, dolfinx_cell, GhostMode.shared_facet)
    else:
        mesh = create_unit_square(MPI.COMM_WORLD, 8, 4, dolfinx_cell, GhostMode.shared_facet)

    real_element = ufl.VectorElement("Real", ufl_cell, 0, dim=4)
    Vr = FunctionSpace(mesh, real_element)
    r = dolfinx.fem.Function(Vr)
    r.x.array[:] = [6.0, 3.0, 4.0, 2.0]

    dx = ufl.Measure("dx", mesh)
    inner_product = ufl.inner(r, r) * dx
    A = dolfinx.fem.assemble_scalar(form(inner_product))
    assert(np.isclose(np.dot(r.x.array[:], r.x.array[:]), A))
