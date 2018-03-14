"Demonstrating function evaluation at arbitrary points."

# Copyright (C) 2008 Anders Logg
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later


from dolfin import *
from numpy import array

# Create mesh and a point in the mesh
mesh = UnitCubeMesh(8, 8, 8);
x = (0.31, 0.32, 0.33)

# A user-defined function
Vs = FunctionSpace(mesh, "CG", 2)
Vv = VectorFunctionSpace(mesh, "CG", 2)
fs = Expression("sin(3.0*x[0])*sin(3.0*x[1])*sin(3.0*x[2])", degree=2)
fv = Expression(("sin(3.0*x[0])*sin(3.0*x[1])*sin(3.0*x[2])",
                 "1.0 + 3.0*x[0] + 4.0*x[1] + 0.5*x[2]","2"), element = Vv.ufl_element())

# Project to a discrete function
g = project(fs, V=Vs)

print("""
Evaluate user-defined scalar function fs
fs(x) = %f
Evaluate discrete function g (projection of fs)
g(x) = %f
Evaluate user-defined vector valued function fv
fs(x) = %s""" % (fs(x), g(x), str(fv(x))) )
