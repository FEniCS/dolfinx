"""
This demo program illustrates how to solve Poisson's equation

    - div grad u(x, y) = f(x, y)

on the unit square with pure Neumann boundary conditions:

    du/dn(x, y) = -sin(5*x)

and source f given by

    f(x, y) = 10*exp(-((x - 0.5)^2 + (y - 0.5)^2) / 0.02)

Since only Neumann conditions are applied, u is only determined up to
a constant by the above equations. An addition constraint is thus
required, for instance

  \int u = 0

This is accomplished in this demo by using a Krylov iterative solver
that removes the component in the null space from the solution vector.
"""

# Copyright (C) 2012 Garth N. Wells
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
# First added:  2012-10-31
# Last changed: 2013-05-30
# Begin demo

from dolfin import *

# Test for PETSc
if not has_linear_algebra_backend("PETSc"):
    info("DOLFIN has not been configured with PETSc. Exiting.")
    exit()

parameters["linear_algebra_backend"] = "PETSc"

# Create mesh and define function space
mesh = UnitSquareMesh(64, 64)
V = FunctionSpace(mesh, "CG", 1)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Expression("10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)", degree=2)
g = Expression("-sin(5*x[0])", degree=2)
a = inner(grad(u), grad(v))*dx
L = f*v*dx + g*v*ds

# Assemble system
A = assemble(a)
b = assemble(L)

# Solution Function
u = Function(V)

# Create Krylov solver
solver = KrylovSolver(A, "gmres")

# Create vector that spans the null space and normalize
null_vec = Vector(u.vector())
V.dofmap().set(null_vec, 1.0)
null_vec *= 1.0/null_vec.norm("l2")

# Create null space basis object and attach to PETSc matrix
null_space = VectorSpaceBasis([null_vec])
as_backend_type(A).set_nullspace(null_space)

# Orthogonalize RHS b with respect to the null space (this gurantees a
# solution exists)
null_space.orthogonalize(b);

# Solve
solver.solve(u.vector(), b)

# Plot solution
plot(u, interactive=True)
