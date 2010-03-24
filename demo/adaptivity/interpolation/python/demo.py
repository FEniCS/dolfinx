"""
This program demonstrates interpolation onto a mesh which is not
completely covered by the original mesh. This situation may arise
during mesh refinement as a result of round-off errors.

When interpolating, DOLFIN tries to locate each point in the original
mesh. If that fails, the closest cell is found and the values
extrapolated from there.
"""

__author__ = "Anders Logg (logg@simula.no)"
__date__ = "2010-03-22"
__copyright__ = "Copyright (C) 2010 " + __author__
__license__  = "GNU LGPL Version 2.1"

# Last changed: 2010-03-24

from dolfin import *

# Set option to allow extrapolation
parameters["allow_extrapolation"] = True

# Create mesh and function space
mesh = UnitSquare(16, 16)
V = FunctionSpace(mesh, "CG", 1)

# Create a function on the original mesh
f = Expression("sin(5.0*x[0])*sin(5.0*x[1])")
v = interpolate(f, V)

# FIXME: We would like to do refined_mesh = refine(mesh) here
# FIXME: but that breaks in parallel

# Refine mesh and create a new function space
refined_mesh = UnitSquare(32, 32)
W = FunctionSpace(refined_mesh, "CG", 1)

# Displace mesh slightly
x = mesh.coordinates()
x += 0.1

# Interpolate to the new mesh
w = interpolate(v, W)

# Plot function
plot(w, interactive=True)
