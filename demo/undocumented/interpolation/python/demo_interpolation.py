"""
This program demonstrates interpolation onto a mesh which is not
completely covered by the original mesh. This situation may arise
during mesh refinement as a result of round-off errors.

When interpolating, DOLFIN tries to locate each point in the original
mesh. If that fails, the closest cell is found and the values
extrapolated from there.
"""

# Copyright (C) 2010 Anders Logg
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
# First added:  2010-03-22
# Last changed: 2013-01-23

from dolfin import *

if not has_cgal():
    print "DOLFIN must be compiled with CGAL to run this demo."
    exit(0)

# Set option to allow extrapolation
parameters["allow_extrapolation"] = True

# Create mesh and function space
mesh = UnitSquareMesh(16, 16)
V = FunctionSpace(mesh, "CG", 1)

# Create a function on the original mesh
f = Expression("sin(5.0*x[0])*sin(5.0*x[1])")
v = interpolate(f, V)

# FIXME: We would like to do refined_mesh = refine(mesh) here
# FIXME: but that breaks in parallel

# Refine mesh and create a new function space
refined_mesh = refine(mesh)
W = FunctionSpace(refined_mesh, "CG", 1)

# Displace mesh slightly
x = mesh.coordinates()
x += 0.1

# Interpolate to the new mesh
w = interpolate(v, W)

# Plot function
plot(w, interactive=True)
