"""This demo program plots a bunch of nonlinear
special functions that are available in UFL."""

# Copyright (C) 2010 Martin S. Alnaes
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
# First added:  2011-10-24
# Last changed: 2011-10-24

from __future__ import print_function
from dolfin import *

# Form compiler options
parameters["form_compiler"]["optimize"]     = True
parameters["form_compiler"]["cpp_optimize"] = True

n = 100
eps = 1e-8

mesh = IntervalMesh(n, -2.0, +2.0)
x = SpatialCoordinate(mesh)[0]

plot(cos(x), title='cos', mesh=mesh)
plot(sin(x), title='sin', mesh=mesh)
plot(tan(x), title='tan', mesh=mesh)

mesh = IntervalMesh(n, 0.0+eps, 1.0-eps)
x = SpatialCoordinate(mesh)[0]

plot(acos(x), title='acos', mesh=mesh)
plot(asin(x), title='asin', mesh=mesh)
plot(atan(x), title='atan', mesh=mesh)
plot(exp(x), title='exp', mesh=mesh)
plot(ln(x), title='ln', mesh=mesh)
plot(sqrt(x), title='sqrt', mesh=mesh)
plot(bessel_J(0, x), title='bessel_J(0, x)', mesh=mesh)
plot(bessel_J(1, x), title='bessel_J(1, x)', mesh=mesh)
plot(bessel_Y(0, x), title='bessel_Y(0, x)', mesh=mesh)
plot(bessel_Y(1, x), title='bessel_Y(1, x)', mesh=mesh)
plot(bessel_I(0, x), title='bessel_I(0, x)', mesh=mesh)
plot(bessel_I(1, x), title='bessel_I(1, x)', mesh=mesh)
plot(bessel_K(0, x), title='bessel_K(0, x)', mesh=mesh)
plot(bessel_K(1, x), title='bessel_K(1, x)', mesh=mesh)
