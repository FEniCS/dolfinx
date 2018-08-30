# Copyright (C) 2007 Kristian B. Oelgaard
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from dolfin import *
from dolfin import cpp
from dolfin.io import XDMFFile
import matplotlib.pyplot as plt

def boundary_value(n):
    if n < 10:
        return float(n)/10.0
    else:
        return 1.0

# Load mesh and subdomains
xdmf_mesh = XDMFFile(MPI.comm_world, "dolfin_fine.xdmf")
mesh = xdmf_mesh.read_mesh(MPI.comm_world, cpp.mesh.GhostMode.none)
sub_domains = xdmf_mesh.read_mf_size_t(mesh)

h = CellDiameter(mesh)

# Create FunctionSpaces
Q = FunctionSpace(mesh, "CG", 1)
V = VectorFunctionSpace(mesh, "CG", 2)

# Create velocity Function from file
xdmf_velocity = XDMFFile(MPI.comm_world, "dolfin_fine_velocity.xdmf")
velocity = xdmf_velocity.read_checkpoint(V, "velocity")

# Initialise source function and previous solution function
f  = Constant(0.0)
u0 = Function(Q)

# Parameters
T = 5.0
dt = 0.1
t = dt
c = 0.00005

# Test and trial functions
u, v = TrialFunction(Q), TestFunction(Q)

# Mid-point solution
u_mid = 0.5*(u0 + u)

# Residual
r = u - u0 + dt*(dot(velocity, grad(u_mid)) - c*div(grad(u_mid)) - f)

# Galerkin variational problem
F = v*(u - u0)*dx + dt*(v*dot(velocity, grad(u_mid))*dx \
                      + c*dot(grad(v), grad(u_mid))*dx)

# Add SUPG stabilisation terms
vnorm = sqrt(dot(velocity, velocity))
F += (h/(2.0*vnorm))*dot(velocity, grad(v))*r*dx

# Create bilinear and linear forms
a = lhs(F)
L = rhs(F)

# Set up boundary condition
g = Constant(boundary_value(0))
bc = DirichletBC(Q, g, sub_domains, 1)

# Assemble matrix
A = assemble(a)
bc.apply(A)

# Create linear solver and factorize matrix
solver = LUSolver(A)
solver.parameters["reuse_factorization"] = True

# Output file
out_file = VTKFile("results/temperature.pvd")

# Set intial condition
u = u0

# Time-stepping, plot initial condition.
i = 0
plt.figure()
plot(u, title=r"t = {0:1.1f}".format(0.0))
i += 1

while t - T < DOLFIN_EPS:
    # Assemble vector and apply boundary conditions
    b = assemble(L)
    bc.apply(b)

    # Solve the linear system (re-use the already factorized matrix A)
    solver.solve(u.vector(), b)

    # Copy solution from previous interval
    u0 = u

    # Plot solution
    if i % 5 == 0:
        plt.figure()
        plot(u, title=r"t = {0:1.1f}".format(t))

    # Save the solution to file
    out_file << (u, t)

    # Move to next interval and adjust boundary condition
    t += dt
    i += 1
    g.assign(boundary_value(int(t/dt)))

plt.show()
