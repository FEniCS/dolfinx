# Copyright (C) 2018 Igor A. Baratta
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for assembly in complex mode"""

from mpi4py import MPI

import numpy as np
import pytest

import dolfinx.la as la
import ufl
from basix.ufl import element
from dolfinx.fem import Function, assemble_matrix, assemble_vector, form, functionspace
from dolfinx.mesh import create_unit_square
from ufl import dx, grad, inner


@pytest.mark.parametrize("complex_dtype", [np.complex64, np.complex128])
def test_complex_assembly(complex_dtype):
    """Test assembly of complex matrices and vectors"""

    real_dtype = np.real(complex_dtype(1.0)).dtype
    mesh = create_unit_square(MPI.COMM_WORLD, 10, 10, dtype=real_dtype)
    P2 = element("Lagrange", mesh.basix_cell(), 2, dtype=real_dtype)
    V = functionspace(mesh, P2)
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    g = -2 + 3.0j

    a_real = form(inner(u, v) * dx, dtype=complex_dtype)
    L1 = form(inner(g, v) * dx, dtype=complex_dtype)

    b = assemble_vector(L1)
    b.scatter_reverse(la.InsertMode.add)
    bnorm = b.norm(la.Norm.l1)
    b_norm_ref = abs(-2 + 3.0j)
    assert bnorm == pytest.approx(b_norm_ref, rel=1e-5)

    A = assemble_matrix(a_real)
    A.scatter_reverse()
    A0_norm = A.squared_norm()

    x = ufl.SpatialCoordinate(mesh)

    a_imag = form(1j * inner(u, v) * dx, dtype=complex_dtype)
    f = 1j * ufl.sin(2 * np.pi * x[0])
    L0 = form(inner(f, v) * dx, dtype=complex_dtype)
    A = assemble_matrix(a_imag)
    A.scatter_reverse()
    A1_norm = A.squared_norm()
    assert A0_norm == pytest.approx(A1_norm)

    b = assemble_vector(L0)
    b.scatter_reverse(la.InsertMode.add)
    b1_norm = b.norm()

    a_complex = form((1 + 1j) * inner(u, v) * dx, dtype=complex_dtype)
    f = ufl.sin(2 * np.pi * x[0])
    L2 = form(inner(f, v) * dx, dtype=complex_dtype)
    A = assemble_matrix(a_complex)
    A.scatter_reverse()
    A2_norm = A.squared_norm()
    assert A1_norm == pytest.approx(A2_norm / 2)
    b = assemble_vector(L2)
    b.scatter_reverse(la.InsertMode.add)
    b2_norm = b.norm(la.Norm.l2)
    assert b2_norm == pytest.approx(b1_norm)


@pytest.mark.parametrize("complex_dtype", [np.complex64, np.complex128])
def test_complex_assembly_solve(complex_dtype, cg_solver):
    """Solve a positive definite helmholtz problem and verify solution
    with the method of manufactured solutions"""

    degree = 3
    real_dtype = np.real(complex_dtype(1.0)).dtype
    mesh = create_unit_square(MPI.COMM_WORLD, 20, 20, dtype=real_dtype)
    P = element("Lagrange", mesh.basix_cell(), degree, dtype=real_dtype)
    V = functionspace(mesh, P)

    x = ufl.SpatialCoordinate(mesh)

    # Define source term
    A = 1.0 + 8.0 * np.pi**2
    C = 1.0 + 1.0j
    f = C * A * ufl.cos(2 * np.pi * x[0]) * ufl.cos(2 * np.pi * x[1])

    # Variational problem
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = form(C * inner(grad(u), grad(v)) * dx + C * inner(u, v) * dx, dtype=complex_dtype)
    L = form(inner(f, v) * dx, dtype=complex_dtype)

    # Assemble
    A = assemble_matrix(a)
    A.scatter_reverse()

    b = assemble_vector(L)
    b.scatter_reverse(la.InsertMode.add)

    u = Function(V, dtype=complex_dtype)
    cg_solver(mesh.comm, A, b, u.x)

    # Reference Solution
    def ref_eval(x):
        return np.cos(2 * np.pi * x[0]) * np.cos(2 * np.pi * x[1])

    u_ref = Function(V, dtype=real_dtype)
    u_ref.interpolate(ref_eval)

    assert np.allclose(np.real(u.x.array), u_ref.x.array, atol=1e-3)
