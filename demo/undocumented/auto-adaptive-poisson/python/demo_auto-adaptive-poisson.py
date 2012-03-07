# Copyright (C) 2011 Marie E. Rognes and Anders Logg
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
# First added:  2010-08-19
# Last changed: 2011-11-14

# Begin demo

from dolfin import *

# Create mesh and define function space
mesh = UnitSquare(8, 8)
V = FunctionSpace(mesh, "Lagrange", 1)

# Define boundary condition
u0 = Function(V)
bc = DirichletBC(V, u0, "x[0] < DOLFIN_EPS || x[0] > 1.0 - DOLFIN_EPS")

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Expression("10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)",
               degree=1)
g = Expression("sin(5*x[0])", degree=1)
a = inner(grad(u), grad(v))*dx
L = f*v*dx + g*v*ds

# Define function for the solution
u = Function(V)

# Define goal functional (quantity of interest)
M = u*dx

# Define error tolerance
tol = 1.e-5

# Solve equation a = L with respect to u and the given boundary
# conditions, such that the estimated error (measured in M) is less
# than tol
solver_parameters = {"error_control":
                     {"dual_variational_solver":
                      {"linear_solver": "gmres"}}}
solve(a == L, u, bc, tol=tol, M=M, solver_parameters=solver_parameters)

## Alternative, more verbose version (+ illustrating how to set parameters)
# problem = LinearVariationalProblem(a, L, u, bc)
# solver = AdaptiveLinearVariationalSolver(problem)
# solver.parameters["error_control"]["dual_variational_solver"]["linear_solver"] = "cg"
# solver.solve(tol, M)

# Plot solution(s)
plot(u.root_node(), title="Solution on initial mesh")
plot(u.leaf_node(), title="Solution on final mesh")
interactive()
