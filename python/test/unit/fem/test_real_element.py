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
    """Tests that assembly of the inner product on a Real space induces 1"""
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

    # Note that the measure = vol(\Omega) = 1
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
def test_dofmap_sizes_mixed_real_element(cell):
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
    """Tests that assembly of the inner product on vector reals induces the identity matrix"""
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

    # Note that the measure = vol(\Omega) = 1
    a = form(ufl.inner(u, v) * ufl.dx)
    A = dolfinx.fem.petsc.assemble_matrix(a)
    A.assemble()
    A_dense = A.convert("dense").getDenseArray()
    assert A.getSize()[0] == A.getSize()[1] == dim
    assert np.isclose(A_dense, np.identity(dim)).all()


@pytest.mark.parametrize("cell", ["interval", "triangle", "tetrahedron", "quadrilateral", "hexahedron"])
def test_real_as_scalar_coefficient(cell):
    """Tests scalar multiplication with scalar real"""
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
    """Tests that the Euclidean inner product is calculated on a vector space
    of reals (i.e. R^d)"""
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
    inner_product = form(ufl.inner(r, r) * dx)
    A = dolfinx.fem.assemble_scalar(inner_product)
    assert(np.isclose(np.dot(r.x.array[:], r.x.array[:]), A))


def test_pure_neumann_constrained_with_real():
    """Equivalent to the classic FEniCS pure Neumann Poisson demo"""
    mesh = create_unit_square(MPI.COMM_WORLD, 10, 10, CellType.triangle, GhostMode.shared_facet)

    element = ufl.MixedElement(ufl.FiniteElement("Lagrange", mesh.ufl_cell(), 1),
                               ufl.FiniteElement("Real", mesh.ufl_cell(), 0))

    V = FunctionSpace(mesh, element)
    u, c = ufl.TrialFunctions(V)
    v, d = ufl.TestFunctions(V)

    dx = ufl.Measure("dx", mesh)
    ds = ufl.Measure("ds", mesh)

    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx + ufl.inner(c, v) * dx + ufl.inner(u, d) * dx
    x = ufl.SpatialCoordinate(mesh)
    f = 10.0 * ufl.exp(-(x[0] - 0.5)**2 + (x[1] - 0.5)**2 / -0.02)
    g = -ufl.sin(5.0 * x[0])
    L = ufl.inner(g, v) * ds + ufl.inner(f, v) * dx

    problem = dolfinx.fem.petsc.LinearProblem(a, L)
    w_h = problem.solve()
    w_h = dolfinx.fem.Function(V)
    u, c = ufl.split(w_h)

    # Check necessary condition for existence
    # (f - c)*dx should be equal to -g*ds?
    interior_data = dolfinx.fem.assemble_scalar(form((f - c) * dx))
    exterior_data = dolfinx.fem.assemble_scalar(form(-g * ds))
    assert np.isclose(interior_data, exterior_data)
