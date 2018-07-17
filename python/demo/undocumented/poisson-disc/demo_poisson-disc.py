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
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

# Begin demo



from dolfin import *
import math
parameters["form_compiler"]["representation"] = "uflacs"

def compute(nsteps, coordinate_degree, element_degree, gdim):
    # Create mesh and define function space

    print(nsteps)
    print(coordinate_degree)
    print(gdim)
    mesh = UnitDiscMesh.create(MPI.comm_world, nsteps, coordinate_degree, gdim)
    V = FunctionSpace(mesh, "Lagrange", element_degree)

    # Compute domain area and average h
    area = assemble(1.0*dx(mesh, degree=2))
    h = (area / mesh.num_cells())**(1.0 / mesh.topology.dim)

    # Define boundary condition
    u0 = Constant(0.0)
    bc = DirichletBC(V, u0, "on_boundary")

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Constant(1.0)
    a = inner(grad(u), grad(v))*dx(degree=3)
    L = f*v*dx(degree=3)

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
    for coordinate_degree in (1, 2):
        for element_degree in (1, 2):
            print(f"Using coordinate degree {coordinate_degree}, element degree {element_degree}")
            encoding = "b" if has_hdf5() else ""
            ufile = XDMFFile(MPI.comm_world, f"poisson-disc-degree-x{coordinate_degree}-e{element_degree}.xdmf", "w" + encoding)
            preverr = None
            prevh = None
            for i, nsteps in enumerate((1, 8, 64)):
                err, h, area, num_cells, u = compute(nsteps, coordinate_degree, element_degree, gdim)
                if preverr is None:
                    conv = 0.0
                    print(f"conv =  N/A, h = {h:.3e}, err = {err:.3e}, area = {area:.16f}, num_cells = {num_cells:d}")
                else:
                    conv = math.log(preverr/err, prevh/h)
                    print(f"conv = {conv:1.2f}, h = {h:.3e}, err = {err:.3e}, area = {area:.16f}, num_cells = {num_cells:d}")
                preverr = err
                prevh = h

                # Save solution to file
                u.rename('u')
                ufile.write(u)

compute_rates()
