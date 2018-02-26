"""This script demonstrates the L2 projection of a function onto a
non-matching mesh."""

# Copyright (C) 2009 Garth N. Wells
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from dolfin import *
import matplotlib.pyplot as plt


# Create mesh and define function spaces
mesh0 = UnitSquareMesh(16, 16)
mesh1 = UnitSquareMesh(64, 64)

# Create expression on P3
u0 = Expression("sin(10.0*x[0])*sin(10.0*x[1])", degree=3)

# Define projection space
P1 = FunctionSpace(mesh1, "CG", 1)

# Define projection variation problem
v  = TestFunction(P1)
u1 = TrialFunction(P1)
a  = v*u1*dx
L  = v*u0*dx

# Compute solution
u1 = Function(P1)
solve(a == L, u1)

# Plot functions
plt.figure()
plot(u0, mesh=mesh0, title="u0")
plt.figure()
plot(u1, title="u1")
plt.show()
