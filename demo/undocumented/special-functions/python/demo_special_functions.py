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

from dolfin import *

# Form compiler options
parameters["form_compiler"]["optimize"]     = True
parameters["form_compiler"]["cpp_optimize"] = True

# Create meshes
n = 100
eps = 1e-8
mesh = Interval(n, -2.0, +2.0)
mesh2 = Interval(n, 0.0+eps, 1.0-eps)
x = interval.x

k = 0
def plotstrings(strings, mesh):
    global k
    for string in strings:
        figure(k)
        k += 1
        expr = eval(string)
        plot(expr, title=string, mesh=mesh)

plotstrings(('cos(x)', 'sin(x)', 'tan(x)'), mesh)
plotstrings(('acos(x)', 'asin(x)', 'atan(x)'), mesh2)
plotstrings(('exp(x)', 'ln(x)', 'sqrt(x)', 'erf(x)'), mesh2)

for nu in (0, 1):
    plotstrings(['bessel_%s(%d, x)' % (c, nu) for c in ('J', 'Y', 'I', 'K')], mesh2)

print "Note that you must press 'q' in the first plot window ('cos(x)') to quit."
interactive()
