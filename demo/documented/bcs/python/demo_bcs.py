"""This demo illustrates how to set boundary conditions for meshes
that include boundary indicators. The mesh used in this demo was
generated with VMTK (http://www.vmtk.org/)."""

# Copyright (C) 2008-2012 Anders Logg
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
# Modified by Martin Alnaes 2012
#
# First added:  2008-05-23
# Last changed: 2012-10-16
# Begin demo

import matplotlib.pyplot as plt
from dolfin import *

# Create mesh and define function space
mesh = Mesh("../aneurysm.xml.gz")
V = FunctionSpace(mesh, "CG", 1)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(0.0)
a = dot(grad(u), grad(v))*dx
L = f*v*dx

# Define boundary condition values
u0 = Constant(0.0)
u1 = Constant(1.0)
u2 = Constant(2.0)
u3 = Constant(3.0)

if has_pybind11():
    markers = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 9999)
    for (f, v) in mesh.domains().markers(mesh.topology().dim()-1).items():
        markers[f] = v

    # Define boundary conditions
    bc0 = DirichletBC(V, u0, markers, 0)
    bc1 = DirichletBC(V, u1, markers, 1)
    bc2 = DirichletBC(V, u2, markers, 2)
    bc3 = DirichletBC(V, u3, markers, 3)
else:
    # Define boundary conditions
    bc0 = DirichletBC(V, u0, 0)
    bc1 = DirichletBC(V, u1, 1)
    bc2 = DirichletBC(V, u2, 2)
    bc3 = DirichletBC(V, u3, 3)

# Set PETSc MUMPS paramter (this is required to prevent a memory error
# in some cases when using MUMPS LU solver).
if has_petsc():
    PETScOptions.set("mat_mumps_icntl_14", 40.0)

# Compute solution
u = Function(V)
solve(a == L, u, [bc0, bc1, bc2, bc3])

# Write solution to file
File("u.pvd") << u

# Plot solution
plot(u)
plt.show()
