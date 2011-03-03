"""This script demonstrate how to project and interpolate functions
between different finite element spaces."""

__author__ = "Anders Logg (logg@simula.no)"
__date__ = "2008-10-06 -- 2009-10-15"
__copyright__ = "Copyright (C) 2008 Anders Logg"
__license__  = "GNU LGPL Version 2.1"

from dolfin import *

# Create mesh and define function spaces
mesh = UnitSquare(64, 64)
P1 = FunctionSpace(mesh, "CG", 1)

# Define function
v = Expression("sin(10.0*x[0])*sin(10.0*x[1])", degree=2)

# Compute projection (L2-projection)
Pv = project(v, P1)

# Compute interpolation (evaluating dofs)
PIv = Function(P1)
PIv.interpolate(v)

# Plot functions
plot(v, mesh=mesh,  title="v")
plot(Pv,  title="Pv")
plot(PIv, title="PI v")
interactive()
