"""Unit tests for the KrylovSolver interface"""

# Copyright (C) 2012 Johan Hake
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
# First added:  2012-02-21
# Last changed: 2012-02-21

import unittest
from dolfin import *

# Assemble system
mesh = UnitSquare(32, 32)
V = FunctionSpace(mesh, 'CG', 1)
bc = DirichletBC(V, Constant(0.0), lambda x, on_boundary: on_boundary)
u = TrialFunction(V); v = TestFunction(V);
u1 = Function(V)

# Forms
a, L = inner(grad(u), grad(v))*dx, Constant(1.0)*v*dx
a_L = inner(grad(u1), grad(v))*dx

# Assemble linear algebra objects
A = assemble(a)
b = assemble(L)
bc.apply(A, b)

if has_linear_algebra_backend("PETSc"):
    class PETScKrylovSolverTester(unittest.TestCase):

        def test_krylov_solver(self):
            "Test PETScKrylovSolver"
            # Get solution vector
            tmp = Function(V)
            x = tmp.vector()

            # Solve first using direct solver
            solve(A, x, b, "lu")

            direct_norm = x.norm("l2")

            # Get solution vector
            x_petsc = as_backend_type(x)

            for prec, descr in krylov_solver_preconditioners():
                if MPI.num_processes() > 1 and prec in ["ilu", "icc", "jacobi", "hypre_amg"]:
                    print "FIXME: Preconditioner '%s' does not work in parallel,"\
                          " skipping" % prec
                    continue

                # With simple interface
                solver = PETScKrylovSolver("gmres", prec)
                solver.solve(A, x_petsc, as_backend_type(b))
                self.assertAlmostEqual(x_petsc.norm("l2"), direct_norm, 5)


                # With PETScPreconditioner interface
                solver = PETScKrylovSolver("gmres", PETScPreconditioner(prec))
                solver.solve(A, x_petsc, as_backend_type(b))
                self.assertAlmostEqual(x_petsc.norm("l2"), direct_norm, 5)

        def test_matrix_free(self):
            "Test matrix free Krylov solver"

            if MPI.num_processes() > 1:
                print "FIXME: Matrix free test does not work in parallel, skipping"
                return

            class LinearOperator(PETScLinearOperator):
                def __init__(self, A):
                    self.A = A

                    PETScLinearOperator.__init__(self, A.size(0), A.size(1))

                def mult(self, x, y):

                    # Make ordinary matrix vector product
                    self.A.mult(x, y)

            class MatrixFreeLinearOperator(PETScLinearOperator) :
                def __init__(self, a_L, u, bc):
                    self.a_L = a_L
                    self.u = u
                    self.bc = bc
                    PETScLinearOperator.__init__(self, A.size(0), A.size(1))

                def mult(self, x, y):
                    # Update Function
                    self.u.vector()[:] = x

                    # Assemble matrix vector product
                    assemble(self.a_L, tensor=y, reset_sparsity=False)

                    # Apply Boundary conditions
                    self.bc.apply(y)

            class IdentityPreconditioner(PETScUserPreconditioner):
                def __init__(self) :
                    PETScUserPreconditioner.__init__(self)

                def solve(self, x, b):
                    x[:] = b

            tmp = Function(V)
            x = tmp.vector()
            solve(A, x, b)

            direct_norm = x.norm("l2")

            x_petsc = as_backend_type(x)
            b_petsc = as_backend_type(b)

            solver = PETScKrylovSolver("gmres")
            solver.solve(A, x_petsc, b_petsc)
            self.assertAlmostEqual(x_petsc.norm("l2"), direct_norm, 5)

            # Matrix free solve
            my_prec = IdentityPreconditioner()
            solver = PETScKrylovSolver("gmres", my_prec)
            solver.solve(LinearOperator(A), x_petsc, b_petsc)
            self.assertAlmostEqual(x_petsc.norm("l2"), direct_norm, 5)

            solver.solve(MatrixFreeLinearOperator(a_L, u1, bc), x_petsc, b_petsc)
            self.assertAlmostEqual(x_petsc.norm("l2"), direct_norm, 5)

if __name__ == "__main__":

    # Turn off DOLFIN output
    set_log_active(False)

    print ""
    print "Testing DOLFIN la/KrylovSolver interface"
    print "----------------------------------------"
    unittest.main()
