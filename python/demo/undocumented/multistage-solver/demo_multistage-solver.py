# Copyright (C) 2007 Kristian B. Oelgaard
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later


from dolfin import *

print("RKSolver is temporarily unavailable")
exit(0)

# Load mesh and subdomains
mesh = Mesh("../dolfin_fine.xml.gz")
sub_domains = MeshFunction("size_t", mesh, "../dolfin_fine_subdomains.xml.gz");
h = CellDiameter(mesh)

# Create FunctionSpaces
Q = FunctionSpace(mesh, "CG", 1)
V = VectorFunctionSpace(mesh, "CG", 2)

# Create velocity Function from file
velocity = Function(V);
File("../dolfin_fine_velocity.xml.gz") >> velocity

# Initialise source function and previous solution function
f  = Constant(0.0)
u0 = Function(Q)

# Parameters
T = 5.0
dt = 0.1
t = Constant(0.0)
c = 0.00005

# Test and trial functions
u, v = Function(Q), TestFunction(Q)

# Residual
r = dot(velocity, grad(u)) - c*div(grad(u)) - f

# Galerkin variational problem (rhs)
F = -(v*dot(velocity, grad(u)) + c*dot(grad(v), grad(u)))*dx

# Add SUPG stabilisation terms
vnorm = sqrt(dot(velocity, velocity))
F -= h/(2.0*vnorm)*dot(velocity, grad(v))*r*dx

# Set up boundary condition
g = Expression("(t<=ramp_stop) ? t : 1.0", t=t, ramp_stop=1.0, degeree=1)
bc = DirichletBC(Q, g, sub_domains, 1)

# Output file
out_file = File("results/temperature.pvd")

scheme = BDF1(F, u, t, [bc])
solver = RKSolver(scheme)

# Time-stepping
while float(scheme.t()) < T:

    solver.step(dt)

    # Plot solution
    plot(u)

    # Save the solution to file
    out_file << (u, float(scheme.t()))
