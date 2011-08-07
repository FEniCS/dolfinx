"Demonstrating function evaluation at arbitrary points."

# Copyright (C) 2008 Anders Logg
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
# Modified by Johan Hake, 2009
#
# First added:  2008-03-17
# Last changed: 2009-10-06

from dolfin import *
from numpy import array

not_working_in_parallel("This demo")

if not has_cgal():
    print "DOLFIN must be compiled with CGAL to run this demo."
    exit(0)

# Create mesh and a point in the mesh
mesh = UnitCube(8, 8, 8);
x = (0.31, 0.32, 0.33)

# A user-defined function
Vs = FunctionSpace(mesh, "CG", 2)
Vv = VectorFunctionSpace(mesh, "CG", 2)
fs = Expression("sin(3.0*x[0])*sin(3.0*x[1])*sin(3.0*x[2])", degree=2)
fv = Expression(("sin(3.0*x[0])*sin(3.0*x[1])*sin(3.0*x[2])",
               "1.0 + 3.0*x[0] + 4.0*x[1] + 0.5*x[2]","2"), element = Vv.ufl_element())

# Project to a discrete function
g = project(fs, V=Vs)

print """
Evaluate user-defined scalar function fs
fs(x) = %f
Evaluate discrete function g (projection of fs)
g(x) = %f
Evaluate user-defined vector valued function fv
fs(x) = %s"""%(fs(x),g(x),str(fv(x)))
