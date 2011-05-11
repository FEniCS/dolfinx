"This demo illustrates basic plotting."

# Copyright (C) 2007-2008 Anders Logg
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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
#
# First added:  2007-05-29
# Last changed: 2009-10-15

from dolfin import *
from math import sqrt

import sys

# Read and plot mesh from file
mesh = Mesh("dolfin-2.xml.gz")
mesh.order()

# Decide which demos to run
try:
    demos = [int(sys.argv[-1])]
except:
    demos = [0, 1, 2, 3]

# Have some fun with the mesh
if 0 in demos:

    R = 0.15
    H = 0.025
    X = 0.3
    Y = 0.4
    dX = H
    dY = 1.5*H
    coordinates = mesh.coordinates()
    original = coordinates.copy()

    for i in xrange(100):

        if X < H or X > 1.0 - H:
            dX = -dX
        if Y < H or Y > 1.0 - H:
            dY = -dY
        X += dX
        Y += dY

        for j in xrange(mesh.num_vertices()):
            x, y = coordinates[j]
            r = sqrt((x - X)**2 + (y - Y)**2)
            if r < R:
                coordinates[j] = [X + (r/R)**2*(x - X), Y + (r/R)**2*(y - Y)]

        plot(mesh)

        for j in xrange(mesh.num_vertices()):
            coordinates[j] = original[j]

# Plot scalar function
if 1 in demos:
    V = FunctionSpace(mesh, "CG", 1)
    f = Expression("t * 100 * exp(-10.0 * (pow(x[0] - t, 2) + pow(x[1] - t, 2)))", element=V.ufl_element())
    f.t = 0.0
    for i in range(100):
        f.t += 0.01
        plot(f, mesh=mesh, rescale=True, title="Scalar function")

# Plot vector function
if 2 in demos:
    mesh = UnitSquare(16, 16)
    V = VectorFunctionSpace(mesh, "CG", 1)
    f = Expression(("-(x[1] - t)*exp(-10.0*(pow(x[0] - t, 2) + pow(x[1] - t, 2)))",
                  " (x[0] - t)*exp(-10.0*(pow(x[0] - t, 2) + pow(x[1] - t, 2)))"), element=V.ufl_element())
    f.t = 0.0
    for i in range(200):
        f.t += 0.005
        plot(f, mesh=mesh, rescale=True, title="Vector function")

if 3 in demos:
    import numpy
    mesh = UnitSquare(10, 10)
    V = VectorFunctionSpace(mesh, "CG", 1)
    f = Expression(("-(x[1] - t)*exp(-10.0*(pow(x[0] - t, 2) + pow(x[1] - t, 2)))",
                    " (x[0] - t)*exp(-10.0*(pow(x[0] - t, 2) + pow(x[1] - t, 2)))"), element=V.ufl_element())

    pts = numpy.array([
        [.24, .24],
        [.24, .74],
        [.74, .24],
        [.74, .74]
        ], dtype='d')
    f.t = 0.0
    for i in range(150):
        f.t += 0.005
        plot(f, mesh=mesh, eval_pts=pts, rescale=True, title="Vector function")

