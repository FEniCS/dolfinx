#!/usr/bin/env py.test

"Unit tests for matrix-free linear solvers (LinearOperator)"

# Copyright (C) 2012 Anders Logg
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

from dolfin_utils.test import *

backends = ["PETSc"]


# Backends supporting the LinearOperator interface
@pytest.mark.parametrize('backend', backends)
def test_linear_operator(backend):

    # Check whether backend is available
    if not has_linear_algebra_backend(backend):
        pytest.skip('Need %s as backend to run this test' % backend)

    # Set linear algebra backend
    prev_backend = parameters["linear_algebra_backend"]
    parameters["linear_algebra_backend"] = backend

    # Define linear operator
    class MyLinearOperator(LinearOperator):

        def __init__(self, a_action, u):
            LinearOperator.__init__(self, u.vector(), u.vector())
            self.a_action = a_action
            self.u = u

        def size(self, dim):
            return self.u.function_space().dim()

        def mult(self, x, y):

            # Update coefficient vector
            self.u.vector()[:] = x

            # Assemble action
            assemble(self.a_action, tensor=y)

    # Try wrapped and backend implementation of operator
    for _as_backend_type in [(lambda x: x), as_backend_type]:

        # Compute reference value by solving ordinary linear system
        mesh = UnitSquareMesh(8, 8)

        V = FunctionSpace(mesh, "Lagrange", 1)
        u = TrialFunction(V)
        v = TestFunction(V)
        f = Constant(1.0)
        a = dot(grad(u), grad(v))*dx + u*v*dx
        L = f*v*dx
        A = assemble(a)
        b = assemble(L)
        x = Vector()
        solve(A, x, b, "gmres", "none")
        norm_ref = norm(x, "l2")

        # Solve using linear operator defined by form action
        u = Function(V)
        a_action = action(a, coefficient=u)
        O = MyLinearOperator(a_action, u)
        O = _as_backend_type(O)
        solve(O, x, b, "gmres", "none")
        norm_action = norm(x, "l2")

        # Check at least that petsc4py interface is available
        if backend == 'PETSc' and has_petsc4py() and _as_backend_type == as_backend_type:
            from petsc4py import PETSc
            assert isinstance(O.mat(), PETSc.Mat)

    # Reset backend
    parameters["linear_algebra_backend"] = prev_backend
