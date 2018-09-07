"""This demo program solves Poisson's equation

    - div grad u(x, y) = f(x, y)

on the unit square with source f given by

    f(x, y) = exp(-100(x^2 + y^2))

and homogeneous Dirichlet boundary conditions.

Note that we use a simplified error indicator, ignoring
edge (jump) terms and the size of the interpolation constant.
"""

# Copyright (C) 2008 Rolv Erlend Bredesen
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import os, matplotlib
if 'DISPLAY' not in os.environ:
    matplotlib.use('agg')

from dolfin import *
from dolfin.cpp.refinement import refine
import dolfin
import dolfin.plotting
from numpy import array, sqrt
from math import pow

TOL = 5e-4           # Error tolerance
REFINE_RATIO = 0.50  # Refine 50 % of the cells in each iteration
MAX_ITER = 20        # Maximal number of iterations

# Create initial mesh
mesh = UnitSquareMesh(MPI.comm_world, 4, 4)
source_str = "exp(-100.0*(pow(x[0], 2) + pow(x[1], 2)))"
source = eval("lambda x: " + source_str)

import numpy

class Boundary(SubDomain):
    def inside(self, x, on_boundary):
        return numpy.full(x.shape[:1], on_boundary)

subdomain = Boundary()

# Adaptive algorithm
for level in range(MAX_ITER):

    # Define variational problem
    V = FunctionSpace(mesh, "CG", 1)
    v = TestFunction(V)
    u = TrialFunction(V)
    f = Expression(source_str, degree=2)
    a = dot(grad(v), grad(u))*dx
    L = v*f*dx

    # Define boundary condition
    u0 = Constant(0.0)
    bc = DirichletBC(V, u0, subdomain)

    # Compute solution
    u = Function(V)
    solve(a == L, u, bc)

    # Compute error indicators
    h = array([c.h() for c in Cells(mesh)])
    K = array([c.volume() for c in Cells(mesh)])
    R = array([abs(source([c.midpoint()[0], c.midpoint()[1]])) for c in Cells(mesh)])
    gamma = h*R*sqrt(K)

    # Compute error estimate
    E = sum([g*g for g in gamma])
    E = sqrt(MPI.sum(mesh.mpi_comm(), E))
    print("Level %d: E = %g (TOL = %g)" % (level, E, TOL))

    # Check convergence
    if E < TOL:
        print("Success, solution converged after %d iterations" % level)
        break

    # Mark cells for refinement
    cell_markers = MeshFunction("bool", mesh, mesh.topology.dim, False)
    gamma_0 = sorted(gamma, reverse=True)[int(len(gamma)*REFINE_RATIO)]
    gamma_0 = MPI.max(mesh.mpi_comm(), gamma_0)
    for c in Cells(mesh):
        cell_markers[c] = gamma[c.index()] > gamma_0

    # Refine mesh
    mesh = refine(mesh, cell_markers)
    mesh.geometry.coord_mapping = dolfin.fem.create_coordinate_map(mesh)

    # Plot mesh
    dolfin.plotting.plot(mesh)
