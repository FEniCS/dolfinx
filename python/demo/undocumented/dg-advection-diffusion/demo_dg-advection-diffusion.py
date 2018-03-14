""" Steady state advection-diffusion equation,
discontinuous formulation using full upwinding.

Implemented in python from cpp demo by Johan Hake.
"""

# Copyright (C) 2008 Johan Hake
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from dolfin import *
import matplotlib.pyplot as plt


# FIXME: Make mesh ghosted
parameters["ghost_mode"] = "shared_facet"

class DirichletBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return abs(x[0] - 1.0) < DOLFIN_EPS and on_boundary

# Load mesh
mesh = Mesh("../unitsquare_64_64.xml.gz")

# Defining the function spaces
V_dg = FunctionSpace(mesh, "DG", 1)
V_cg = FunctionSpace(mesh, "CG", 1)
V_u  = VectorFunctionSpace(mesh, "CG", 2)

# Create velocity Function
u = Function(V_u, "../unitsquare_64_64_velocity.xml.gz")

# Test and trial functions
v   = TestFunction(V_dg)
phi = TrialFunction(V_dg)

# Diffusivity
kappa = Constant(0.0)

# Source term
f = Constant(0.0)

# Penalty term
alpha = Constant(5.0)

# Mesh-related functions
n = FacetNormal(mesh)
h = CellDiameter(mesh)
h_avg = (h('+') + h('-'))/2

# ( dot(v, n) + |dot(v, n)| )/2.0
un = (dot(u, n) + abs(dot(u, n)))/2.0

# Bilinear form
a_int = dot(grad(v), kappa*grad(phi) - u*phi)*dx

a_fac = kappa*(alpha/h('+'))*dot(jump(v, n), jump(phi, n))*dS \
      - kappa*dot(avg(grad(v)), jump(phi, n))*dS \
      - kappa*dot(jump(v, n), avg(grad(phi)))*dS

a_vel = dot(jump(v), un('+')*phi('+') - un('-')*phi('-') )*dS  + dot(v, un*phi)*ds

a = a_int + a_fac + a_vel

# Linear form
L = v*f*dx

# Set up boundary condition (apply strong BCs)
g = Expression("sin(pi*5.0*x[1])", degree=2)
bc = DirichletBC(V_dg, g, DirichletBoundary(), "geometric")

# Solution function
phi_h = Function(V_dg)

# Assemble and apply boundary conditions
A = assemble(a)
b = assemble(L)
bc.apply(A, b)

# Solve system
solve(A, phi_h.vector(), b)

# Project solution to a continuous function space
up = project(phi_h, V=V_cg)

file = File("temperature.pvd")
file << up

# Plot solution
plot(up)
plt.show()
