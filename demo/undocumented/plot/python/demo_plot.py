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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
#
# Modified by Fredrik Valdmanis 2012
# Modified by Benjamin Kehlet 2012
# Modified by Joachim B Haga 2012
#
# First added:  2007-05-29
# Last changed: 2012-11-12

from dolfin import *
import os.path
from math import sqrt
import numpy

import sys

# Read mesh from file
mesh = Mesh(os.path.join(os.path.pardir, "dolfin-2.xml.gz"))

# Decide which demos to run
demos = map(int, sys.argv[1:]) or [0, 1, 2]

# Have some fun with the mesh
if 0 in demos:

    R = 0.15
    H = 0.025
    X = 0.3
    Y = 0.4
    dX = 0.5*H
    dY = 0.75*H
    coordinates = mesh.coordinates()
    original = coordinates.copy()

    for i in xrange(200):

        if X < H or X > 1.0 - H:
            dX = -dX
        if Y < H or Y > 1.0 - H:
            dY = -dY
        X += dX
        Y += dY


        if 0:
            # Straight-forward (slow) loop implementation
            for j in xrange(mesh.num_vertices()):
                x, y = coordinates[j]
                r = sqrt((x - X)**2 + (y - Y)**2)
                if r < R:
                    coordinates[j] = [X + (r/R)**2*(x - X), Y + (r/R)**2*(y - Y)]
        else:
            # numpy (fast) vectorised implementation
            translated = coordinates - [X,Y]
            r = numpy.sqrt(numpy.sum(translated**2, axis=1))
            r2 = (r/R)**2
            translated[:,0] *= r2
            translated[:,1] *= r2
            newcoords = [X,Y] + translated
            coordinates[r<R] = newcoords[r<R]

        plot(mesh, title="Plotting mesh")

        coordinates[:] = original

# Plot scalar function
if 1 in demos:
    V = FunctionSpace(mesh, "CG", 1)
    f = Expression("t * 100 * exp(-10.0 * (pow(x[0] - t, 2) + pow(x[1] - t, 2)))", element=V.ufl_element(), t=0.0)
    for i in range(100):
        f.t += 0.01
        plot(f, mesh=mesh, rescale=True, title="Plotting scalar function")

# Plot vector function
if 2 in demos:
    mesh = UnitSquareMesh(16, 16)
    V = VectorFunctionSpace(mesh, "CG", 1)
    f = Expression(("-(x[1] - t)*exp(-10.0*(pow(x[0] - t, 2) + pow(x[1] - t, 2)))",\
                  " (x[0] - t)*exp(-10.0*(pow(x[0] - t, 2) + pow(x[1] - t, 2)))"), \
                   element=V.ufl_element(), t=0.0)
    for i in range(200):
        f.t += 0.005
        plot(f, mesh=mesh, rescale=True, title="Plotting vector function")

interactive()
