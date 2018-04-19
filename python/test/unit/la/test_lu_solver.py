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
from dolfin.la import PETScLUSolver, PETScVector

def test_lu_solver():

    mesh = UnitSquareMesh(MPI.comm_world, 12, 12)
    V = FunctionSpace(mesh, "Lagrange", 1)
    u, v = TrialFunction(V), TestFunction(V)

    a = Constant(1.0)*u*v*dx
    L = Constant(1.0)*v*dx
    assembler = dolfin.fem.assembling.Assembler(a, L)
    A, b = assembler.assemble()

    norm = 13.0

    solver = PETScLUSolver(mesh.mpi_comm())
    x = PETScVector(mesh.mpi_comm())
    solver.set_operator(A)
    solver.solve(x, b)
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

    mesh = UnitSquareMesh(MPI.comm_world, 12, 12)
    V = FunctionSpace(mesh, "Lagrange", 1)
    u, v = TrialFunction(V), TestFunction(V)


    a = Constant(1.0)*u*v*dx
    L = Constant(1.0)*v*dx
    assembler = dolfin.fem.assembling.Assembler(a, L)
    A, b = assembler.assemble()
    norm = 13.0

    solver = PETScLUSolver(mesh.mpi_comm())
    x = PETScVector(mesh.mpi_comm())
    solver.solve(A, x, b)
    assert round(x.norm("l2") - norm, 10) == 0

    assemble(Constant(0.5)*u*v*dx, tensor=A)
    x = PETScVector(mesh.mpi_comm())
    solver.solve(x, b)
    assert round(x.norm("l2") - 2.0*norm, 10) == 0

    solver.set_operator(A)
    solver.solve(x, b)
    assert round(x.norm("l2") - 2.0*norm, 10) == 0
