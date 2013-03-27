# This demo solves the Stokes equations, using stabilized
# first order elements for the velocity and pressure. The
# sub domains for the different boundary conditions used
# in this simulation are computed by the demo program in
# src/demo/mesh/subdomains.
#
# Original implementation: ../cpp/main.cpp by Anders Logg

# Copyright (C) 2007 Kristian B. Oelgaard
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
# Modified by Anders Logg, 2009.
#
# First added:  2007-11-15
# Last changed: 2009-11-26

from dolfin import *

# Load mesh and subdomains
mesh = Mesh("../dolfin_fine.xml.gz")
sub_domains = MeshFunction("size_t", mesh, "../dolfin_fine_subdomains.xml.gz");

# Define function spaces
scalar = FunctionSpace(mesh, "CG", 1)
vector = VectorFunctionSpace(mesh, "CG", 1)
system = vector * scalar

# Create functions for boundary conditions
noslip = Constant((0, 0))
inflow = Expression(("-sin(x[1]*pi)", "0"))
zero   = Constant(0)

# No-slip boundary condition for velocity
bc0 = DirichletBC(system.sub(0), noslip, sub_domains, 0)

# Inflow boundary condition for velocity
bc1 = DirichletBC(system.sub(0), inflow, sub_domains, 1)

# Boundary condition for pressure at outflow
bc2 = DirichletBC(system.sub(1), zero, sub_domains, 2)

# Collect boundary conditions
bcs = [bc0, bc1, bc2]

# Define variational problem
(v, q) = TestFunctions(system)
(u, p) = TrialFunctions(system)
f = Constant((0, 0))
h = CellSize(mesh)
beta  = 0.2
delta = beta*h*h
a = (inner(grad(v), grad(u)) - div(v)*p + q*div(u) + \
    delta*inner(grad(q), grad(p)))*dx
L = inner(v + delta*grad(q), f)*dx

# Compute solution
w = Function(system)
solve(a == L, w, bcs)
u, p = w.split()

# Save solution in VTK format
ufile_pvd = File("velocity.pvd")
ufile_pvd << u
pfile_pvd = File("pressure.pvd")
pfile_pvd << p

# Plot solution
plot(u)
plot(p)
interactive()
