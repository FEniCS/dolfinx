"Unit tests for matrix-free linear solvers (LinearOperator)"

# Copyright (C) 2012 Anders Logg
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

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
            return self.u.function_space().dim

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
        Ob = _as_backend_type(O)
        solve(Ob, x, b, "gmres", "none")
        norm_action = norm(x, "l2")

        # Check at least that petsc4py interface is available
        if backend == 'PETSc' and has_petsc4py() and _as_backend_type == as_backend_type:
            from petsc4py import PETSc
            assert isinstance(Ob.mat(), PETSc.Mat)

    # Reset backend
    parameters["linear_algebra_backend"] = prev_backend
