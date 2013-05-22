"""This demo program solves Poisson's equation

    - div grad u(x, y) = f(x, y)

on the unit square with source f given by

    f(x, y) = 10*exp(-((x - 0.5)^2 + (y - 0.5)^2) / 0.02)

and boundary conditions given by

    u(x, y) = 0        for x = 0 or x = 1
du/dn(x, y) = sin(5*x) for y = 0 or y = 1

It demonstrates how to extract petsc4py objects from dolfin objects
and use them in a petsc4py Krylov solver.

Based on "demo/pde/poisson/python/demo_poisson.py"
"""

# Copyright (C) 2007-2011, 2013 Anders Logg, Lawrence Mitchell
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

# Begin demo

from dolfin import *
try:
    from petsc4py import PETSc
except:
    print "*** Warning: you need to have petsc4py installed for this demo to run",
    print "Exiting."
    exit()

if not has_petsc4py():
    print "*** Warning: Dolfin is not compiled with petsc4py support",
    print "Exiting."
    exit()

parameters["linear_algebra_backend"] = "PETSc"

# Create mesh and define function space
mesh = UnitSquareMesh(32, 32)
V = FunctionSpace(mesh, "Lagrange", 1)

# Define Dirichlet boundary (x = 0 or x = 1)
def boundary(x):
    return x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS

# Define boundary condition
u0 = Constant(0.0)
bc = DirichletBC(V, u0, boundary)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Expression("10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)")
g = Expression("sin(5*x[0])")
a = inner(grad(u), grad(v))*dx
L = f*v*dx + g*v*ds

# Compute solution
u = Function(V)

A, b = assemble_system(a, L, bc)
A_petsc = as_backend_type(A).mat()
b_petsc = as_backend_type(b).vec()
x_petsc = as_backend_type(u.vector()).vec()

ksp = PETSc.KSP().create()

ksp.setOperators(A_petsc)

pc = PETSc.PC().create()

pc.setOperators(A_petsc)
pc.setType(pc.Type.JACOBI)

ksp.setPC(pc)

ksp.solve(b_petsc, x_petsc)

# Plot solution
plot(u, interactive=True)
# Save solution to file
file = File("poisson.pvd")
file << u



