"""This demo program demonstrates how to manipulate (higher-order) mesh
coordinates."""

# Copyright (C) 2016 Jan Blechta
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from dolfin import *
import matplotlib.pyplot as plt


# Create mesh
comm = MPI.comm_world
mesh = UnitDiscMesh.create(comm, 20, 2, 2)
plt.figure()
plot(mesh)

# Fetch coordinate function
C = FunctionSpace(mesh, mesh.ufl_coordinate_element())
c = Function(C)
get_coordinates(c, mesh.geometry())

# Deform coordinates harmonically subject to BC
u, v = TrialFunction(C), TestFunction(C)
a = inner(grad(u), grad(v))*dx
L = dot(Constant((0, 0)), v)*dx
bc1 = DirichletBC(C, (-1, -1), "x[0] < -0.5")
bc2 = DirichletBC(C, c, "x[0] >= -0.5")
displacement = Function(C)
solve(a == L, displacement, [bc1, bc2])
c_vec = c.vector()
c_vec += displacement.vector()

# Set coordinates
set_coordinates(mesh.geometry(), c)
plt.figure()
plot(mesh)

# We can create (cubic) mesh from function
C3 = VectorFunctionSpace(mesh, "Lagrange", 4)
c3 = interpolate(c, C3)
mesh3 = create_mesh(c3)
plt.figure()
plot(mesh3)

# Display plots
plt.show()
