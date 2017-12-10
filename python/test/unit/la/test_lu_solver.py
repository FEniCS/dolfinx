"""Unit tests for the LUSolver interface"""

# Copyright (C) 2014 Garth N. Wells
#
# This file is part of DOLFIN.
#
# DOLFIN is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DOLFIN is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.

from dolfin import *
import pytest
from dolfin_utils.test import skip_if_not_PETSc, skip_in_parallel

backends = ["PETSc", skip_in_parallel("Eigen")]

@pytest.mark.parametrize('backend', backends)
def test_lu_solver(backend):

    # Check whether backend is available
    if not has_linear_algebra_backend(backend):
        pytest.skip('Need %s as backend to run this test' % backend)

    # Set linear algebra backend
    prev_backend = parameters["linear_algebra_backend"]
    parameters["linear_algebra_backend"] = backend

    mesh = UnitSquareMesh(12, 12)
    V = FunctionSpace(mesh, "Lagrange", 1)
    u, v = TrialFunction(V), TestFunction(V)
    A = assemble(Constant(1.0)*u*v*dx)
    b = assemble(Constant(1.0)*v*dx)

    norm = 13.0

    solver = LUSolver()
    x = Vector()
    solver.solve(A, x, b)
    assert round(x.norm("l2") - norm, 10) == 0

    solver = LUSolver(A)
    x = Vector()
    solver.solve(x, b)
    assert round(x.norm("l2") - norm, 10) == 0

    solver = LUSolver()
    x = Vector()
    solver.set_operator(A)
    solver.solve(x, b)
    assert round(x.norm("l2") - norm, 10) == 0

    solver = LUSolver()
    x = Vector()
    solver.solve(A, x, b)
    assert round(x.norm("l2") - norm, 10) == 0

    # Reset backend
    parameters["linear_algebra_backend"] = prev_backend


@pytest.mark.parametrize('backend', backends)
def test_lu_solver_reuse(backend):
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


    # Check whether backend is available
    if not has_linear_algebra_backend(backend):
        pytest.skip('Need %s as backend to run this test' % backend)

    # Set linear algebra backend
    prev_backend = parameters["linear_algebra_backend"]
    parameters["linear_algebra_backend"] = backend

    mesh = UnitSquareMesh(12, 12)
    V = FunctionSpace(mesh, "Lagrange", 1)
    u, v = TrialFunction(V), TestFunction(V)
    b = assemble(Constant(1.0)*v*dx)

    A = assemble(Constant(1.0)*u*v*dx)
    norm = 13.0

    solver = LUSolver(A)
    x = Vector()
    solver.solve(x, b)
    assert round(x.norm("l2") - norm, 10) == 0

    assemble(Constant(0.5)*u*v*dx, tensor=A)
    x = Vector()
    solver.solve(x, b)
    assert round(x.norm("l2") - 2.0*norm, 10) == 0

    solver.set_operator(A)
    solver.solve(x, b)
    assert round(x.norm("l2") - 2.0*norm, 10) == 0

    # Reset backend
    parameters["linear_algebra_backend"] = prev_backend
