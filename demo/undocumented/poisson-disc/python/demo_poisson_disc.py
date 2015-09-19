"""This demo program solves Poisson's equation

    - div grad u(x, y) = f(x, y)

on the unit square with source f given by

    f(x, y) = 10*exp(-((x - 0.5)^2 + (y - 0.5)^2) / 0.02)

and boundary conditions given by

    u(x, y) = 0        for x = 0 or x = 1
du/dn(x, y) = sin(5*x) for y = 0 or y = 1
"""

# Copyright (C) 2007-2011 Anders Logg
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

# Begin demo

from __future__ import print_function

from dolfin import *
parameters["form_compiler"]["representation"] = "uflacs"

def compute(nsteps):
    # Create mesh and define function space
    degree = 1
    gdim = 2
    mesh = UnitDiscMesh(mpi_comm_world(), nsteps, degree, gdim)
    nc = mesh.num_cells()
    V = FunctionSpace(mesh, "Lagrange", degree)

    # Define boundary condition
    u0 = Constant(0.0)
    bc = DirichletBC(V, u0, "on_boundary")

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Constant(1.0)
    a = inner(grad(u), grad(v))*dx
    L = f*v*dx

    # Compute solution
    u = Function(V)
    solve(a == L, u, bc)

    # Compute error norm
    x = SpatialCoordinate(mesh)
    uexact = (1.0 - x**2) / 4.0
    M = (u - uexact)**2*dx(degree=5)
    M0 = uexact**2*dx(degree=5)
    area = assemble(1.0*dx(mesh))
    return nc, sqrt(assemble(M) / assemble(M0)), area

# Print convergence, getting rate of 3.5
import math
preverr = None
for nsteps in (1, 2, 4, 8, 16, 32, 64):
    nc, err, area = compute(nsteps)
    if preverr is None:
        conv = 0.0
    else:
        conv = math.log(preverr/err, 2)
    print("steps = %d, cells = %d, sqrt(cells) = %d, |M|/|M0| = %.4g, area = %f, conv = %.4g:" % (nsteps, nc, sqrt(nc), err, area, conv))
    preverr = err

# Save solution in VTK format
# file = File("poisson.xdmf")
# file << u

# Plot solution
#plot(u, interactive=True)
