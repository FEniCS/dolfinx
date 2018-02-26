"""This script demonstrate how to project and interpolate functions
between different finite element spaces."""

# Copyright (C) 2008 Anders Logg
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from dolfin import *
import matplotlib.pyplot as plt


# Create mesh and define function spaces
mesh = UnitSquareMesh(64, 64)
P1 = FunctionSpace(mesh, "CG", 1)

# Define function
v = Expression("sin(10.0*x[0])*sin(10.0*x[1])", degree=2)

# Compute projection (L2-projection)
Pv = project(v, V=P1)

# Compute interpolation (evaluating dofs)
PIv = Function(P1)
PIv.interpolate(v)

# Plot functions
plt.figure()
plot(v, mesh=mesh, title="v")
plt.figure()
plot(Pv,  title="Pv")
plt.figure()
plot(PIv, title="PI v")
plt.show()
