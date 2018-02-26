"""This demo solves the Stokes equations, using linear elements
enriched with a bubble for the velocity and linear elements for the
pressure (Mini elements). The sub domains for the different boundary
conditions used in this simulation are computed by the demo program in
src/demo/mesh/subdomains."""

# Copyright (C) 2007 Kristian B. Oelgaard
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later


import matplotlib.pyplot as plt
from dolfin import *

# Load mesh and subdomains
mesh = Mesh("../dolfin_fine.xml.gz")
sub_domains = MeshFunction("size_t", mesh, "../dolfin_fine_subdomains.xml.gz")

# Build function spaces on Mini element
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
B = FiniteElement("Bubble",   mesh.ufl_cell(), 3)
V = VectorElement(P1 + B)
Q = P1
W = FunctionSpace(mesh, V * Q)

# No-slip boundary condition for velocity
# NOTE: Projection here is inefficient workaround of issue #489, FFC issue #69
noslip = project(Constant((0, 0)), W.sub(0).collapse())
bc0 = DirichletBC(W.sub(0), noslip, sub_domains, 0)

# Inflow boundary condition for velocity
# NOTE: Projection here is inefficient workaround of issue #489, FFC issue #69
inflow = project(Expression(("-sin(x[1]*pi)", "0.0"), degree=2), W.sub(0).collapse())
bc1 = DirichletBC(W.sub(0), inflow, sub_domains, 1)

# Collect boundary conditions
bcs = [bc0, bc1]

# Define variational problem
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)
f = Constant((0, 0))
a = (inner(grad(u), grad(v)) - div(v)*p + q*div(u))*dx
L = inner(f, v)*dx

# Compute solution
w = Function(W)
solve(a == L, w, bcs)

# Split the mixed solution using deepcopy
# (needed for further computation on coefficient vector)
(u, p) = w.split(True)

print("Norm of velocity coefficient vector: %.15g" % u.vector().norm("l2"))
print("Norm of pressure coefficient vector: %.15g" % p.vector().norm("l2"))

# Split the mixed solution using a shallow copy
(u, p) = w.split()

# Save solution in VTK format
ufile_pvd = File("velocity.pvd")
ufile_pvd << u
pfile_pvd = File("pressure.pvd")
pfile_pvd << p

# Plot solution
plt.figure()
plot(u, title="velocity")

plt.figure()
plot(p, title="pressure")

# Display plots
plt.show()
