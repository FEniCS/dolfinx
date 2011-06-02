# This demo program solves Poisson's using
# using a discontinuous Galerkin formulation
# and enforcing boundary conditions in a symmetric way.
# The demo builds on the standard dg demo
# demo/pde/dg/poisson/python/demo.py

# Copyright (C) 2008 Kent-Andre Mardal
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
# First added:  2008
# Last changed: 2008

from dolfin import *

# Create mesh and finite element
mesh = UnitSquare(10,10)

parameters["linear_algebra_backend"] = "uBLAS"

# Sub domain for Dirichlet boundary condition
class DirichletBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return bool(on_boundary)

# Define variational problem
V = FunctionSpace(mesh, "DG", 1)
u = TrialFunction(V)
v = TestFunction(V)

# Normal component, mesh size and right-hand side
n = FacetNormal(mesh)
h = AvgMeshSize(mesh)
f = Function(V, "500.0*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)")

# Parameters
alpha = 4.0
gamma = 8.0

# Define boundary condition
u0 = Constant(0)
boundary = DirichletBoundary()
bc = DirichletBC(V, u0, boundary)

# Bilinear form
a = dot(grad(u), grad(v))*dx \
   - dot(jump(u, n), avg(grad(v)))*dS \
   - dot(jump(v, n), avg(grad(u)))*dS \
   + alpha/h('+')*dot(jump(u, n), jump(v, n))*dS \
   - dot(mult(u, n), grad(v))*ds \
   - dot(mult(v, n), grad(u))*ds \
   + gamma/h*v*u*ds

# Linear form
L = v*f*dx

# Standard way of computing A and b
A = assemble(a)
b = assemble(L)
x = b.copy()
x.zero()
solve(A, x, b)
file = File("A1.m") ; file << A;
file = File("b1.m") ; file << b;

# Project u
u = Function(V)
u.vector().set(x.data())
u_proj = project(u, V)

# Project solution to piecewise linears
P1 = FunctionSpace(mesh, "CG", 1)
u_proj = project(u, P1)


# Save solution to file
file = File("poisson1.pvd")
file << u_proj

# Plot solution
plot(u_proj, interactive=True)

# Symmetric way of computing A and b
A, b = assemble_system(a, L, bc, mesh)
x = b.copy()
x.zero()
solve(A, x, b)
file = File("A2.m") ; file << A;
file = File("b2.m") ; file << b;

# Project u
u = Function(V)
u.vector().set(x.data())
u_proj = project(u, P1)

# Save solution to file
file = File("poisson2.pvd")
file << u_proj

# Plot solution
plot(u_proj, interactive=True)



