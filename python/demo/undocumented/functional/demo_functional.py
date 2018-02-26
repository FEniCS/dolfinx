"""This demo program computes the value of the functional

    M(v) = int v^2 + (grad v)^2 dx

on the unit square for v = sin(x) + cos(y). The exact
value of the functional is M(v) = 2 + 2*sin(1)*(1 - cos(1))

The functional M corresponds to the energy norm for a
simple reaction-diffusion equation."""

# Copyright (C) 2007 Kristian B. Oelgaard
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later


from dolfin import *

# Create mesh and define function space
mesh = UnitSquareMesh(16, 16)
V = FunctionSpace(mesh, "CG", 2)

# Define the function v
v = Expression("sin(x[0]) + cos(x[1])",
               element=V.ufl_element(),
               domain=mesh)

# Define functional
M = (v*v + dot(grad(v), grad(v)))*dx(mesh)

# Evaluate functional
value = assemble(M)

exact_value = 2.0 + 2.0*sin(1.0)*(1.0 - cos(1.0))
print("The energy norm of v is: %.15g" % value)
print("It should be:            %.15g" % exact_value)
