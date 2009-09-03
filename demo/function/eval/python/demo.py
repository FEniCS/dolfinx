"Demonstrating function evaluation at arbitrary points."

__author__ = "Anders Logg (logg@simula.no)"
__date__ = "2008-03-17 -- 2008-12-12"
__copyright__ = "Copyright (C) 2008 Anders Logg"
__license__  = "GNU LGPL Version 2.1"

from dolfin import *
from numpy import array

if not has_gts():
    print "DOLFIN must be compiled with GTS to run this demo."
    exit(0)

# Create mesh and a point in the mesh
mesh = UnitCube(8, 8, 8);
x = array((0.31, 0.32, 0.33))
values = array((0.0,))

# A user-defined function
V = FunctionSpace(mesh, "CG", 2)
f = Function(V, "sin(3.0*x[0])*sin(3.0*x[1])*sin(3.0*x[2])")

# Project to a discrete function
g = project(f, V)

# Evaluate user-defined function f
f.eval(values, x)
print "f(x) =", values[0]

# Evaluate discrete function g (projection of f)
g.eval(values, x)
print "g(x) =", values[0]
