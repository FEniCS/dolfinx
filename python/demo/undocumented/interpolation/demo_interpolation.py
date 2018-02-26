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
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from dolfin import *
import matplotlib.pyplot as plt


# Set option to allow extrapolation
parameters["allow_extrapolation"] = True

# Create mesh and function space
mesh = UnitSquareMesh(16, 16)
V = FunctionSpace(mesh, "CG", 1)

# Create a function on the original mesh
f = Expression("sin(5.0*x[0])*sin(5.0*x[1])", degree=2)
v = interpolate(f, V)

# Refine mesh and create a new function space
refined_mesh = refine(mesh)
W = FunctionSpace(refined_mesh, "CG", 1)

# Displace mesh slightly
x = mesh.coordinates()
x += 0.1

# Interpolate to the new mesh
w = interpolate(v, W)

# Plot function
plot(w)
plt.show()
