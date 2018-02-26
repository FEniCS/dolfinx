# Copyright (C) 2011-2013 Marie E. Rognes, Martin S. Alnaes
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

# Begin demo

#
# NB! This demo is work in progress and will be updated as we get further in the multidomain features implementation.
#
from dolfin import *
import matplotlib.pyplot as plt


class Left(SubDomain):
    def inside(self, x, on_boundary):
        return (x[0] <= 0.25+DOLFIN_EPS)

class Mid(SubDomain):
    def inside(self, x, on_boundary):
        return (x[0] >= 0.375-DOLFIN_EPS) and (x[0] <= 0.625+DOLFIN_EPS)

class Right(SubDomain):
    def inside(self, x, on_boundary):
        return (x[0] >= 0.75-DOLFIN_EPS)

# Define mesh and subdomains
mesh = UnitSquareMesh(64, 64)
domains = MeshFunction("size_t", mesh, mesh.topology().dim())
domains.set_all(0)
Left().mark(domains, 1)
Mid().mark(domains, 2)
Right().mark(domains, 3)

# Define input data
alpha = Constant(1e-3)
f = Constant(3.0)
g = Constant(5.0)

# Define function space and basis functions
V = FunctionSpace(mesh, "CG", 2)
u = TrialFunction(V)
v = TestFunction(V)

# Define regions as tuples of subdomain labels
DL, DM, DR = (1,2), (2,), (2,3) # ***

# Define new measures associated with the interior domains
dx = Measure('dx', domain=mesh, subdomain_data=domains)

# Make forms for equation
a = u*v*dx() + alpha*dot(grad(u), grad(v))*dx() # ***
L = f*v*dx(DR) + g*v*dx(DL) - (f+g)/2*v*dx(DM) # ***

# Solve problem
u = Function(V)
solve(a == L, u)

# Plot solution and gradient
plt.figure()
plot(domains, title='domains')
plt.figure()
plot(u, title="u")
plt.show()
