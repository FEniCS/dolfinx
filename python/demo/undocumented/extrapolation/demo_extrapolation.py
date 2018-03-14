# Copyright (C) 2010 Anders Logg
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from dolfin import *
import matplotlib.pyplot as plt


# Create mesh and function spaces
mesh = UnitSquareMesh(8, 8)
P1 = FunctionSpace(mesh, "CG", 1)
P2 = FunctionSpace(mesh, "CG", 2)

# Create exact dual
dual = Expression("sin(5.0*x[0])*sin(5.0*x[1])", degree=2)

# Create P1 approximation of exact dual
z1 = Function(P1)
z1.interpolate(dual)

# Create P2 approximation from P1 approximation
z2 = Function(P2)
z2.extrapolate(z1)

# Plot approximations
plt.figure()
plot(z1, title="z1")
plt.figure()
plot(z2, title="z2")
plt.show()
