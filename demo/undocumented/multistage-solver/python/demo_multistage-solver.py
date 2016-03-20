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
# Modified by Anders Logg, 2008
# Modified by Johan Hake, 2008
# Modified by Garth N. Wells, 2009
# Modified by Johan Hake, 2013
#
# First added:  2007-11-14
# Last changed: 2013-04-05
#
# This demo solves the time-dependent convection-diffusion equation by
# a SUPG stabilized method. The velocity field used
# in the simulation is the output from the Stokes (Taylor-Hood) demo.
# The sub domains for the different boundary conditions are computed
# by the demo program in src/demo/subdomains.
#
# FIXME: Add shock capturing term and then revert back to the Stokes
#        velocity
# FIXME: This demo showcase experimental features of a RKSolver (time integrator)
# FIXME: using a MultiStageScheme. It could be removed or changed anytime.

from __future__ import print_function
from dolfin import *

print("RKSolver is temporarily unavailable")
exit(0)

# Load mesh and subdomains
mesh = Mesh("../dolfin_fine.xml.gz")
sub_domains = MeshFunction("size_t", mesh, "../dolfin_fine_subdomains.xml.gz");
h = CellSize(mesh)

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

# Hold plot
#interactive()
