# Copyright (C) 2011 Marie E. Rognes
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

# Begin demo

import numpy as np
from dolfin import *
from dolfin.plotting import plot

# Create classes for defining parts of the boundaries and the interior
# of the domain
class Left(SubDomain):
    def inside(self, x, on_boundary):
        return np.isclose(x[:,0], 0.0)

class Right(SubDomain):
    def inside(self, x, on_boundary):
        return np.isclose(x[:,0], 1.0)

class Bottom(SubDomain):
    def inside(self, x, on_boundary):
        return np.isclose(x[:,1], 0.0)

class Top(SubDomain):
    def inside(self, x, on_boundary):
        return np.isclose(x[:,1], 1.0)

class Obstacle(SubDomain):
    def inside(self, x, on_boundary):
        c1 = np.logical_and(x[:, 1] < 0.7, x[:, 1] > 0.5)
        c2 = np.logical_and(x[:, 0] > 0.2, x[:, 0] < 1.0)
        return np.logical_and(c1, c2)

# Initialize sub-domain instances
left = Left()
top = Top()
right = Right()
bottom = Bottom()
obstacle = Obstacle()

# Define mesh
mesh = UnitSquareMesh(MPI.comm_world, 64, 64)

# Initialize mesh function for interior domains
domains = MeshFunction("size_t", mesh, mesh.topology().dim())
domains.set_all(0)
obstacle.mark(domains, 1)

# Initialize mesh function for boundary domains
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
boundaries.set_all(0)
left.mark(boundaries, 1)
top.mark(boundaries, 2)
right.mark(boundaries, 3)
bottom.mark(boundaries, 4)

# Define input data
a0 = Constant(1.0)
a1 = Constant(0.01)
g_L = Expression("- 10*exp(- pow(x[1] - 0.5, 2))", degree=2)
g_R = Constant(1.0)
f = Constant(1.0)

# Define function space and basis functions
V = FunctionSpace(mesh, "CG", 2)
u = TrialFunction(V)
v = TestFunction(V)

# Define Dirichlet boundary conditions at top and bottom boundaries
bcs = [DirichletBC(V, 5.0, boundaries, 2),
       DirichletBC(V, 0.0, boundaries, 4)]

# Define new measures associated with the interior domains and
# exterior boundaries
dx = Measure('dx', domain=mesh, subdomain_data=domains)
ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

# Define variational form
F = (inner(a0*grad(u), grad(v))*dx(0) + inner(a1*grad(u), grad(v))*dx(1)
     - g_L*v*ds(1) - g_R*v*ds(3)
     - f*v*dx(0) - f*v*dx(1))

# Separate left and right hand sides of equation
a, L = lhs(F), rhs(F)

# Solve problem
u = Function(V)
solve(a == L, u, bcs)

w = u.vector().get_local()
print (w.min(), w.max())

# Evaluate integral of normal gradient over top boundary
n = FacetNormal(mesh)
m1 = dot(grad(u), n)*ds(2)
# v1 = assemble(m1)
# print("\int grad(u) * n ds(2) = ", v1)

# Evaluate integral of u over the obstacle
m2 = u*dx(1)
# v2 = assemble(m2)
# print("\int u dx(1) = ", v2)

# Plot solution
import matplotlib.pyplot as plt
plt.figure()
plot(u, title="Solution u")

# Plot solution and gradient
# plt.figure()
# plot(grad(u), title="Projected grad(u)")

# Show plots
plt.show()
