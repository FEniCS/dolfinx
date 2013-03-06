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
#
# First added:  2012-08-24
# Last changed: 2012-08-24

import unittest
from dolfin import *

# Backends supporting the LinearOperator interface
backends = ["PETSc", "uBLAS"]

class TestLinearOperator(unittest.TestCase):

    def test_linear_operator(self):

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
                assemble(self.a_action, tensor=y, reset_sparsity=False)

        # Iterate over backends supporting linear operators
        for backend in backends:

            # Check whether backend is available
            if not has_linear_algebra_backend(backend):
                continue

            # Skip testing uBLAS in parallel
            if MPI.num_processes() > 1 and backend == "uBLAS":
                print "Not running uBLAS test in parallel"
                return

            # Set linear algebra backend
            parameters["linear_algebra_backend"] = backend

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
            solve(O, x, b, "gmres", "none")
            norm_action = norm(x, "l2")

if __name__ == "__main__":

    print ""
    print "Testing DOLFIN la/LinearOperator (matrix-free) interface"
    print "--------------------------------------------------------"
    unittest.main()
