# Copyright (C) 2010 Marie E. Rognes
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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
#
# First added:  2010-04-28
# Last changed: 2010-04-28

from dolfin import *
from time import *

set_log_level(INFO)

# Create mesh and function spaces
n = 4
mesh = UnitCube(n, n, n)

k = 1
V = VectorFunctionSpace(mesh, "CG", k+1)
Q = FunctionSpace(mesh, "CG", k)
W = V * Q

V2 = VectorFunctionSpace(mesh, "CG", k+2)
Q2 = FunctionSpace(mesh, "CG", k+1)
W2 = V2 * Q2

# Create exact dual
dual = Expression(("sin(5.0*x[0])*sin(5.0*x[1])", "x[1]", "1.0",
                   "pow(x[0], 3)"), degree=3)

# Create P1 approximation of exact dual
z1 = Function(W)
z1.interpolate(dual)

# Create P2 approximation from P1 approximation
tic = time()
z2 = Function(W2)
z2.extrapolate(z1)
elapsed = time() - tic;

print "\nTime elapsed: %0.3g s." % elapsed
