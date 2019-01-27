# Copyright (C) 2018 Igor A. Baratta
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for assembly in complex mode"""

import numpy as np
import pytest
from petsc4py import PETSc

import dolfin
import ufl
from ufl import dx, grad, inner

pytestmark = pytest.mark.skipif(
    not dolfin.has_petsc_complex, reason="Only works in complex mode.")


def test_complex_assembly():
    """Test assembly of complex matrices and vectors"""

    mesh = dolfin.generation.UnitSquareMesh(dolfin.MPI.comm_world, 10, 10)
    P2 = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), 2)
    V = dolfin.function.functionspace.FunctionSpace(mesh, P2)

    u = dolfin.function.argument.TrialFunction(V)
    v = dolfin.function.argument.TestFunction(V)

    g = -2 + 3.0j
    j = 1.0j

    a_real = inner(u, v) * dx
    L1 = inner(g, v) * dx

    bnorm = dolfin.fem.assemble(L1).norm(PETSc.NormType.N1)
    b_norm_ref = abs(-2 + 3.0j)
    assert np.isclose(bnorm, b_norm_ref)
    A0_norm = dolfin.fem.assemble(a_real).norm(PETSc.NormType.FROBENIUS)

    x = dolfin.SpatialCoordinate(mesh)

    a_imag = j * inner(u, v) * dx
    f = 1j * ufl.sin(2 * np.pi * x[0])
    L0 = inner(f, v) * dx
    A1_norm = dolfin.fem.assemble(a_imag).norm(PETSc.NormType.FROBENIUS)
    assert np.isclose(A0_norm, A1_norm)
    b1_norm = dolfin.fem.assemble(L0).norm(PETSc.NormType.N2)

    a_complex = (1 + j) * inner(u, v) * dx
    f = ufl.sin(2 * np.pi * x[0])
    L2 = inner(f, v) * dx
    A2_norm = dolfin.fem.assemble(a_complex).norm(PETSc.NormType.FROBENIUS)
    assert np.isclose(A1_norm, A2_norm / np.sqrt(2))
    b2_norm = dolfin.fem.assemble(L2).norm(PETSc.NormType.N2)
    assert np.isclose(b2_norm, b1_norm)


def test_complex_assembly_solve():
    """Solve a positive definite helmholtz problem and verify solution
    with the method of manufactured solutions

    """

    degree = 3
    mesh = dolfin.generation.UnitSquareMesh(dolfin.MPI.comm_world, 20, 20)
    P = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), degree)
    V = dolfin.function.functionspace.FunctionSpace(mesh, P)

    x = dolfin.SpatialCoordinate(mesh)

    # Define source term
    A = 1 + 2 * (2 * np.pi)**2
    f = (1. + 1j) * A * ufl.cos(2 * np.pi * x[0]) * ufl.cos(2 * np.pi * x[1])

    # Variational problem
    u = dolfin.function.argument.TrialFunction(V)
    v = dolfin.function.argument.TestFunction(V)
    C = 1 + 1j
    a = C * inner(grad(u), grad(v)) * dx + C * inner(u, v) * dx
    L = inner(f, v) * dx

    # Assemble
    A = dolfin.fem.assemble(a)
    b = dolfin.fem.assemble(L)

    # Create solver
    solver = dolfin.cpp.la.PETScKrylovSolver(mesh.mpi_comm())
    dolfin.cpp.la.PETScOptions.set("ksp_type", "preonly")
    dolfin.cpp.la.PETScOptions.set("pc_type", "lu")
    solver.set_from_options()
    x = A.createVecRight()
    solver.set_operator(A)
    solver.solve(x, b)

    # Reference Solution
    @dolfin.function.expression.numba_eval
    def ref_eval(values, x, cell_idx):
        values[:, 0] = np.cos(2 * np.pi * x[:, 0]) * np.cos(2 * np.pi * x[:, 1])
    u_ref = dolfin.interpolate(dolfin.Expression(ref_eval), V)

    xnorm = x.norm(PETSc.NormType.N2)
    x_ref_norm = u_ref.vector().norm(PETSc.NormType.N2)
    assert np.isclose(xnorm, x_ref_norm)
