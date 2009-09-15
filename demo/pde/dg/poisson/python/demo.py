"""This demo program solves Poisson's equation

    - div grad u(x, y) = f(x, y)

on the unit square with source f given by

    f(x, y) = 500*exp(-((x-0.5)^2 + (y-0.5)^2)/0.02)

and boundary conditions given by

    u(x, y)     = 0
    du/dn(x, y) = 0

using a discontinuous Galerkin formulation (interior penalty method).
"""

__author__    = "Kristian B. Oelgaard (k.b.oelgaard@tudelft.nl)"
__date__      = "2007-10-02 -- 2008-12-19"
__copyright__ = "Copyright (C) 2007 Kristian B. Oelgaard"
__license__   = "GNU LGPL Version 2.1"

# Modified by Anders Logg, 2008.

from dolfin import *

# Create mesh and define function space
mesh = UnitSquare(24, 24)
V = FunctionSpace(mesh, "DG", 1)

# Define test and trial functions
v = TestFunction(V)
u = TrialFunction(V)

# Define normal component, mesh size and right-hand side
n = FacetNormal(mesh)
h = CellSize(mesh)
h_avg = (h('+') + h('-'))/2
f = Function(V, "500.0*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)")

# Define parameters
alpha = 4.0
gamma = 8.0

# Define bilinear form
a = dot(grad(v), grad(u))*dx \
   - dot(avg(grad(v)), jump(u, n))*dS \
   - dot(jump(v, n), avg(grad(u)))*dS \
   + alpha/h_avg*dot(jump(v, n), jump(u, n))*dS \
   - dot(grad(v), u*n)*ds \
   - dot(v*n, grad(u))*ds \
   + gamma/h*v*u*ds

# Define linear form
L = v*f*dx

# Compute solution
problem = VariationalProblem(a, L)
u = problem.solve()

# Test new computation of the normal vector of FFC
v2 = TestFunction(V)
u2 = TrialFunction(V)

n2 = triangle.n
a2 = dot(grad(v2), grad(u2))*dx \
   - dot(avg(grad(v2)), jump(u2, n2))*dS \
   - dot(jump(v2, n2), avg(grad(u2)))*dS \
   + alpha/h_avg*dot(jump(v2, n2), jump(u2, n2))*dS \
   - dot(grad(v2), u2*n2)*ds \
   - dot(v2*n2, grad(u2))*ds \
   + gamma/h*v2*u2*ds
L2 = v2*f*dx

problem2 = VariationalProblem(a2, L2)
u2 = problem2.solve()

# Print norm of vector for both problems
print "u  ", u.vector().norm("l2")
print "u2 ", u2.vector().norm("l2")

# Project solution to piecewise linears
P1 = FunctionSpace(mesh, "CG", 1)
u_proj = project(u, P1)

# Save solution to file
file = File("poisson.pvd")
file << u_proj

# Plot solution
plot(u_proj, interactive=True)



# Test code

# Project solution to piecewise linears
u_proj2 = project(u2, P1)

# Plot solution
plot(u_proj2, interactive=True)
