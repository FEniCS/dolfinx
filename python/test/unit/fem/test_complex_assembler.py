# Copyright (C) 2018 Igor A. Baratta
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for assembly in complex mode"""

from mpi4py import MPI
from petsc4py import PETSc

import numpy as np
import pytest

import ufl
from basix.ufl import element
from dolfinx.fem import Function, form, functionspace
from dolfinx.fem.petsc import assemble_matrix, assemble_vector
from dolfinx.mesh import create_unit_square
from ufl import dx, grad, inner

pytestmark = pytest.mark.skipif(
    not np.issubdtype(PETSc.ScalarType, np.complexfloating), reason="Only works in complex mode.")  # type: ignore


def test_complex_assembly():
    """Test assembly of complex matrices and vectors"""

    mesh = create_unit_square(MPI.COMM_WORLD, 10, 10)
    P2 = element("Lagrange", mesh.basix_cell(), 2)
    V = functionspace(mesh, P2)
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    g = -2 + 3.0j
    j = 1.0j

    a_real = form(inner(u, v) * dx)
    L1 = form(inner(g, v) * dx)

    b = assemble_vector(L1)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    bnorm = b.norm(PETSc.NormType.N1)
    b_norm_ref = abs(-2 + 3.0j)
    assert bnorm == pytest.approx(b_norm_ref, rel=1e-5)

    A = assemble_matrix(a_real)
    A.assemble()
    A0_norm = A.norm(PETSc.NormType.FROBENIUS)

    x = ufl.SpatialCoordinate(mesh)

    a_imag = form(j * inner(u, v) * dx)
    f = 1j * ufl.sin(2 * np.pi * x[0])
    L0 = form(inner(f, v) * dx)
    A = assemble_matrix(a_imag)
    A.assemble()
    A1_norm = A.norm(PETSc.NormType.FROBENIUS)
    assert A0_norm == pytest.approx(A1_norm)

    b = assemble_vector(L0)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    b1_norm = b.norm(PETSc.NormType.N2)

    a_complex = form((1 + j) * inner(u, v) * dx)
    f = ufl.sin(2 * np.pi * x[0])
    L2 = form(inner(f, v) * dx)
    A = assemble_matrix(a_complex)
    A.assemble()
    A2_norm = A.norm(PETSc.NormType.FROBENIUS)
    assert A1_norm == pytest.approx(A2_norm / np.sqrt(2))
    b = assemble_vector(L2)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    b2_norm = b.norm(PETSc.NormType.N2)
    assert b2_norm == pytest.approx(b1_norm)


def test_complex_assembly_solve():
    """Solve a positive definite helmholtz problem and verify solution
    with the method of manufactured solutions"""

    degree = 3
    mesh = create_unit_square(MPI.COMM_WORLD, 20, 20)
    P = element("Lagrange", mesh.basix_cell(), degree)
    V = functionspace(mesh, P)

    x = ufl.SpatialCoordinate(mesh)

    # Define source term
    A = 1.0 + 2.0 * (2.0 * np.pi)**2
    f = (1. + 1j) * A * ufl.cos(2 * np.pi * x[0]) * ufl.cos(2 * np.pi * x[1])

    # Variational problem
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    C = 1.0 + 1.0j
    a = form(C * inner(grad(u), grad(v)) * dx + C * inner(u, v) * dx)
    L = form(inner(f, v) * dx)

    # Assemble
    A = assemble_matrix(a)
    A.assemble()
    b = assemble_vector(L)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    # Create solver
    solver = PETSc.KSP().create(mesh.comm)
    x = A.createVecRight()
    solver.setOperators(A)
    solver.solve(b, x)

    # Reference Solution
    def ref_eval(x):
        return np.cos(2 * np.pi * x[0]) * np.cos(2 * np.pi * x[1])
    u_ref = Function(V)
    u_ref.interpolate(ref_eval)

    diff = (x - u_ref.vector).norm(PETSc.NormType.N2)
    assert diff == pytest.approx(0.0, abs=1e-1)
