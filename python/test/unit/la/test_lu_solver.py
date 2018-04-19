"""Unit tests for the LUSolver interface"""

# Copyright (C) 2014 Garth N. Wells
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import dolfin
from dolfin import *
import pytest
from dolfin_utils.test import skip_in_parallel

def test_lu_solver():

    mesh = UnitSquareMesh(12, 12)
    V = FunctionSpace(mesh, "Lagrange", 1)
    u, v = TrialFunction(V), TestFunction(V)

    a = Constant(1.0)*u*v*dx
    L = Constant(1.0)*v*dx
    assembler = dolfin.fem.assembling.Assembler(a, L)
    A, b = assembler.assemble()

    norm = 13.0

    solver = LUSolver()
    x = PETScVector()
    solver.solve(A, x, b)
    assert round(x.norm("l2") - norm, 10) == 0

    solver = LUSolver(A)
    x = PETScVector()
    solver.solve(x, b)
    assert round(x.norm("l2") - norm, 10) == 0

    solver = LUSolver()
    x = PETScVector()
    solver.set_operator(A)
    solver.solve(x, b)
    assert round(x.norm("l2") - norm, 10) == 0

    solver = LUSolver()
    x = PETScVector()
    solver.solve(A, x, b)
    assert round(x.norm("l2") - norm, 10) == 0


def test_lu_solver_reuse():
    """Test that LU re-factorisation is only performed after
    set_operator(A) is called"""

    # Test requires PETSc version 3.5 or later. Use petsc4py to check
    # version number.
    try:
        from petsc4py import PETSc
    except ImportError:
        pytest.skip("petsc4py required to check PETSc version")
    else:
        if not PETSc.Sys.getVersion() >= (3, 5, 0):
            pytest.skip("PETSc version must be 3.5  of higher")

    mesh = UnitSquareMesh(12, 12)
    V = FunctionSpace(mesh, "Lagrange", 1)
    u, v = TrialFunction(V), TestFunction(V)


    a = Constant(1.0)*u*v*dx
    L = Constant(1.0)*v*dx
    assembler = dolfin.fem.assembling.Assembler(a, L)
    A, b = assembler.assemble()
    norm = 13.0

    solver = LUSolver(A)
    x = PETScVector()
    solver.solve(x, b)
    assert round(x.norm("l2") - norm, 10) == 0

    assemble(Constant(0.5)*u*v*dx, tensor=A)
    x = PETScVector()
    solver.solve(x, b)
    assert round(x.norm("l2") - 2.0*norm, 10) == 0

    solver.set_operator(A)
    solver.solve(x, b)
    assert round(x.norm("l2") - 2.0*norm, 10) == 0

    # Reset backend
    parameters["linear_algebra_backend"] = prev_backend
