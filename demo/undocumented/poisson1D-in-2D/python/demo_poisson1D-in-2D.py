"""This demo program solves Poisson's equation

    - div grad u(xi) = f(xi)

on a unit interval embedded in 2D and rotated pi/4 radians
anticlockwise from the x axis, where xi is the distance along the
interval (ie the domain is 0<xi<1).  The source f is given by

    f(xi) = 9*pi^2*sin(3*pi*xi)

The boundary conditions are given by

    u(xi) = 0 for xi = 0
    du/dxi = 0 for xi = 1
"""

# Copyright (C) 2012 Imperial College London and others.
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
# Modified by David Ham 2012
#
# First added:  2012-12-04
# Last changed: 2012-12-04

from dolfin import *
import numpy

# Create mesh and function space
squaremesh = UnitSquareMesh(50,2)

boundarymesh = BoundaryMesh(squaremesh, "exterior")

# Subdomain to extract bottom boundary.
class BottomEdge(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], 0.0)

mesh = SubMesh(boundarymesh, BottomEdge())

V = FunctionSpace(mesh, "CG", 1)

class Rotation(object):
    """Class implementing rotations of a unit interval through the angle theta."""
    def __init__(self, theta):
        self.theta = theta
        self.mat = self._mat(theta)
        self.invmat = self._mat(-theta)

    def _mat(self, theta):
        return numpy.matrix([[numpy.cos(theta), -numpy.sin(theta)],
                             [numpy.sin(theta),  numpy.cos(theta)]])

    def to_interval(self, x):
        """Map the point x back to the horizontal line."""
        return numpy.dot(self.invmat[0,:], x)

    def to_interval_c(self):
        """Return a c expression mapping x back to the line."""
        return "(x[0]*%f + x[1]*%f)" % (numpy.cos(self.theta), numpy.sin(self.theta))

    def rotate(self, mesh):
        """Rotate mesh through theta."""

        mesh.coordinates()[:,:] = \
            numpy.dot(mesh.coordinates()[:,:], self.mat.T)

rotation = Rotation(numpy.pi/4)

rotation.rotate(mesh)

# Sub domain for Dirichlet boundary condition
class DirichletBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and rotation.to_interval(x) < DOLFIN_EPS


# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Expression("9.0*pi*pi*sin(3.0*pi*%s)" % rotation.to_interval_c(), degree=2)
g = Expression("3.0*pi*cos(3.0*pi*%s)" % rotation.to_interval_c(), degree=2)

a = dot(grad(u), grad(v))*dx
L = f*v*dx + g*v*ds

# Define boundary condition
u0 = Constant(0.0)
bc = DirichletBC(V, u0, DirichletBoundary())

# Compute solution
u = Function(V)
solve(a == L, u, bc)

# Save solution to file
file = File("poisson.pvd")
file << u

# Plot solution
#plot(u, interactive=True)
