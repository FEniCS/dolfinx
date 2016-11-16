"""This script demonstrate how to use sub function assignment."""

# Copyright (C) 2013 Johan Hake
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
# First added:  2013-11-08
# Last changed: 2013-11-08

from dolfin import *

import time

# Create mesh and define function spaces
mesh = UnitSquareMesh(256, 256)
V = FunctionSpace(mesh, "CG", 1)
VV = VectorFunctionSpace(mesh, "CG", 1)

# Define function
v0 = Expression("sin(2*pi*x[0])*sin(2*pi*x[1])", degree=2)
v1 = Expression("cos(2*pi*x[0])*cos(2*pi*x[1])", degree=2)

vv1 = Function(VV)
vv2 = Function(VV)
vv3 = Function(VV)

u0 = Function(V)
u1 = Function(V)

u0.interpolate(v0)
u1.interpolate(v1)

# Compute projection (L2-projection)
t0 = time.time()
vv0 = project(as_vector((u0, u1)), V=VV)
tp = time.time() - t0

# Compute interpolation (evaluating dofs)
t0 = time.time()
vv1.interpolate(Expression(("v0", "v1"), v0=v0, v1=v1, degree=2))
ti = time.time() - t0

# Assign mixed function from two scalar functions
t0 = time.time()
assign(vv2, [u0, u1])
ta0 = time.time() - t0

# Assign mixed function from two scalar functions using FunctionAssigner
assigner = FunctionAssigner(VV, [V, V])
t0 = time.time()
assigner.assign(vv3, [u0, u1])
ta1 = time.time() - t0

# Plot functions
plot(vv0, title="Projection; time=%.3fs" % tp)
plot(vv1, title="Interpolation; time=%.3fs" % ti)
plot(vv2, title="Assignment; time=%.3fs" % ta0)
plot(vv3, title="Assignment (cached); time=%.3fs" % ta1)
interactive()
