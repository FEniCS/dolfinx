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
import math
parameters["form_compiler"]["representation"] = "uflacs"

def compute(nsteps, coordinate_degree, element_degree, gdim):
    # Create mesh and define function space

    print(nsteps)
    print(coordinate_degree)
    print(gdim)
    mesh = UnitDiscMesh(mpi_comm_world(), nsteps, coordinate_degree, gdim)
    V = FunctionSpace(mesh, "Lagrange", element_degree)

    # Compute domain area and average h
    area = assemble(1.0*dx(mesh))
    h = (area / mesh.num_cells())**(1.0 / mesh.topology().dim())

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

    # Compute relative error norm
    x = SpatialCoordinate(mesh)
    uexact = (1.0 - x**2) / 4.0
    M = (u - uexact)**2*dx(degree=5)
    M0 = uexact**2*dx(degree=5)
    err = sqrt(assemble(M) / assemble(M0))

    return err, h, area, mesh.num_cells(), u

def compute_rates():
    "Compute convergence rates for degrees 1 and 2."
    gdim = 2
    tdim = gdim
    for coordinate_degree in (1, 2):
        for element_degree in (1, 2):
            print("\nUsing coordinate degree %d, element degree %d" % (coordinate_degree, element_degree))
            ufile = XDMFFile(mpi_comm_world(), "poisson-disc-degree-x%d-e%d.xdmf" % (coordinate_degree, element_degree))
            encoding = XDMFFile.Encoding_HDF5 if has_hdf5() else XDMFFile.Encoding_ASCII
            preverr = None
            prevh = None
            for i, nsteps in enumerate((1, 8, 64)):
                err, h, area, num_cells, u = compute(nsteps, coordinate_degree, element_degree, gdim)
                if preverr is None:
                    conv = 0.0
                    print("conv =  N/A, h = %.3e, err = %.3e, area = %.16f, num_cells = %d" % (h, err, area, num_cells))
                else:
                    conv = math.log(preverr/err, prevh/h)
                    print("conv = %1.2f, h = %.3e, err = %.3e, area = %.16f, num_cells = %d" % (conv, h, err, area, num_cells))
                preverr = err
                prevh = h

                # Save solution to file
                u.rename('u', 'u')

                if MPI.size(mpi_comm_world()) > 1 and encoding == XDMFFile.Encoding_ASCII:
                    print("XDMF file output not supported in parallel without HDF5")
                else:
                    ufile.write(u, encoding)

            # Plot solution
            #plot(u, title="u, degree=%d" % degree)
    #interactive()

compute_rates()