"""This demo demonstrates how to solve a mixed Poisson type equation
defined over a sphere (the surface of a ball in 3D) including how to
create a cell_orientation map, needed for some forms defined over
manifolds."""

# Copyright (C) 2012 Marie E. Rognes
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
# First added:  2012-12-09
# Last changed: 2012-12-09

# Begin demo

from dolfin import *
import numpy

# Create mesh
#n = 16
#ball = Mesh(Sphere(Point(0.0, 0.0, 0.0), 1.0), n)
#mesh = BoundaryMesh(ball)
#file = File("sphere16.xml")
#file << mesh
mesh = Mesh("sphere16.xml")

# Define global normal
global_normal = Expression(("x[0]", "x[1]", "x[2]"))

# Create orientation map (2 == up, 1 == down, 0 == undefined) For each
# cell, compute the local normal defined by the cross product of the
# first and second edge, compare this to the globally defined normal
# (evaluated at the cell midpoint) via the dot product. If the local
# and global normal point in different directions, the cell is
# considered a 'down' cell, otherwise it is an 'up' cell.
mf = mesh.data().create_mesh_function("cell_orientation", 2)
coords = mesh.coordinates()
for cell in cells(mesh):
    ind = [v.index() for v in vertices(cell)]
    v1 = coords[ind[1], :] - coords[ind[0], :]
    v2 = coords[ind[2], :] - coords[ind[0], :]
    local_normal = numpy.cross(v1, v2)
    p = cell.midpoint()
    orientation = numpy.inner(global_normal(p), local_normal)
    if orientation > 0:
        mf[cell.index()] = 2
    elif orientation < 0:
        mf[cell.index()] = 1
    else:
        raise Exception, "Not expecting orthogonal local/global normal"

# Define function spaces and basis functions
V = FunctionSpace(mesh, "RT", 1)
Q = FunctionSpace(mesh, "DG", 0)
W = V * Q

(sigma, u) = TrialFunctions(W)
(tau, v) = TestFunctions(W)
g = Constant(1.0)

# Define forms
a = (inner(sigma, tau) + div(sigma)*v + div(tau)*u)*dx
L = g*v*dx

# Solve problem
w = Function(W)
solve(a == L, w)
(sigma, u) = w.split()

# Plot CG1 representation of solutions
sigma_cg = project(sigma, VectorFunctionSpace(mesh, "CG", 1))
u_cg = project(u, FunctionSpace(mesh, "CG", 1))
plot(sigma_cg, interactive=True)
plot(u_cg)

# Store solutions
file = File("sigma.pvd")
file << sigma
file = File("u.pvd")
file << u

interactive()
