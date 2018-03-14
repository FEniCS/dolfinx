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
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later


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

markers = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 1)
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
