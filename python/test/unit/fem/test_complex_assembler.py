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
from dolfin.specialfunctions import SpatialCoordinate
from ufl import dx, grad, inner

pytestmark = pytest.mark.skipif(
    not dolfin.has_petsc_complex, reason="Only works in complex mode.")


def test_complex_assembly():
    """Test assembly of complex matrices and vectors"""

    mesh = dolfin.generation.UnitSquareMesh(dolfin.MPI.comm_world, 10, 10)
    P2 = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), 2)
    V = dolfin.functionspace.FunctionSpace(mesh, P2)

    u = dolfin.function.TrialFunction(V)
    v = dolfin.function.TestFunction(V)

    g = -2 + 3.0j
    j = 1.0j

    a_real = inner(u, v) * dx
    L1 = inner(g, v) * dx

    b = dolfin.fem.assemble_vector(L1)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    bnorm = b.norm(PETSc.NormType.N1)
    b_norm_ref = abs(-2 + 3.0j)
    assert bnorm == pytest.approx(b_norm_ref)

    A = dolfin.fem.assemble_matrix(a_real)
    A.assemble()
    A0_norm = A.norm(PETSc.NormType.FROBENIUS)

    x = SpatialCoordinate(mesh)

    a_imag = j * inner(u, v) * dx
    f = 1j * ufl.sin(2 * np.pi * x[0])
    L0 = inner(f, v) * dx
    A = dolfin.fem.assemble_matrix(a_imag)
    A.assemble()
    A1_norm = A.norm(PETSc.NormType.FROBENIUS)
    assert A0_norm == pytest.approx(A1_norm)

    b = dolfin.fem.assemble_vector(L0)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    b1_norm = b.norm(PETSc.NormType.N2)

    a_complex = (1 + j) * inner(u, v) * dx
    f = ufl.sin(2 * np.pi * x[0])
    L2 = inner(f, v) * dx
    A = dolfin.fem.assemble_matrix(a_complex)
    A.assemble()
    A2_norm = A.norm(PETSc.NormType.FROBENIUS)
    assert A1_norm == pytest.approx(A2_norm / np.sqrt(2))
    b = dolfin.fem.assemble_vector(L2)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    b2_norm = b.norm(PETSc.NormType.N2)
    assert b2_norm == pytest.approx(b1_norm)


def test_complex_assembly_solve():
    """Solve a positive definite helmholtz problem and verify solution
    with the method of manufactured solutions

    """

    degree = 3
    mesh = dolfin.generation.UnitSquareMesh(dolfin.MPI.comm_world, 20, 20)
    P = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), degree)
    V = dolfin.functionspace.FunctionSpace(mesh, P)

    x = SpatialCoordinate(mesh)

    # Define source term
    A = 1.0 + 2.0 * (2.0 * np.pi)**2
    f = (1. + 1j) * A * ufl.cos(2 * np.pi * x[0]) * ufl.cos(2 * np.pi * x[1])

    # Variational problem
    u = dolfin.function.TrialFunction(V)
    v = dolfin.function.TestFunction(V)
    C = 1.0 + 1.0j
    a = C * inner(grad(u), grad(v)) * dx + C * inner(u, v) * dx
    L = inner(f, v) * dx

    # Assemble
    A = dolfin.fem.assemble_matrix(a)
    A.assemble()
    b = dolfin.fem.assemble_vector(L)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    # Create solver
    solver = PETSc.KSP().create(mesh.mpi_comm())
    solver.setOptionsPrefix("test_lu_")
    opts = PETSc.Options("test_lu_")
    opts["ksp_type"] = "preonly"
    opts["pc_type"] = "lu"
    solver.setFromOptions()
    x = A.createVecRight()
    solver.setOperators(A)
    solver.solve(b, x)

    # Reference Solution
    def ref_eval(x):
        return np.cos(2 * np.pi * x[0]) * np.cos(2 * np.pi * x[1])
    u_ref = dolfin.function.Function(V)
    u_ref.interpolate(ref_eval)

    diff = (x - u_ref.vector).norm(PETSc.NormType.N2)
    assert diff == pytest.approx(0.0, abs=1e-1)
