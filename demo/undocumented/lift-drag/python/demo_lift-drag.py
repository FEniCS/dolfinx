"""This demo demonstrates how to compute functionals (or forms in
general) over subsets of the mesh. The two functionals lift and
drag are computed for the pressure field around a dolphin. Here, we
use the pressure field obtained from solving the Stokes equations
(see demo program in the sub directory
src/demo/pde/stokes/taylor-hood).

The calculation only includes the pressure contribution (not shear
forces).
"""

# Copyright (C) 2007 Kristian B. Oelgaard
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
# Modified by Anders Logg, 2008.
# Modified by Garth N. Wells, 2009.
#
# First added:  2007-11-14
# Last changed: 2008-12-27

from __future__ import print_function
from dolfin import *

# Read the mesh from file
mesh =  Mesh("../dolfin_fine.xml.gz")

# Create FunctionSpace for pressure field
Vp = FunctionSpace(mesh, "CG", 1)
p = Function(Vp, "../dolfin_fine_pressure.xml.gz")

# Define sub domain for the dolphin
class Fish(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] > DOLFIN_EPS and x[0] < (1.0 - DOLFIN_EPS) and \
               x[1] > DOLFIN_EPS and x[1] < (1.0 - DOLFIN_EPS)

markers = FacetFunctionSizet(mesh, 1)
Fish().mark(markers, 1);

# Define functionals for drag and lift
ds = ds(subdomain_data=markers)
n = FacetNormal(mesh)
D = -p*n[0]*ds(1)
L = p*n[1]*ds(1)

# Assemble functionals over sub domain
drag = assemble(D)
lift = assemble(L)

print("Lift: %f" % lift)
print("Drag: %f" % drag)
