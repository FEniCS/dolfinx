"""This demo demonstrates how to compute functionals (or forms in
general) over subsets of the mesh. The two functionals lift and
drag are computed for the pressure field around a dolphin. Here, we
use the pressure field obtained from solving the Stokes equations
(see demo program in the sub directory
src/demo/pde/stokes/taylor-hood).

The calculation only includes the pressure contribution (not shear
forces).
"""

__author__ = "Kristian B. Oelgaard (k.b.oelgaard@tudelft.nl)"
__date__ = "2007-11-14 -- 2008-12-27"
__copyright__ = "Copyright (C) 2007 Kristian B. Oelgaard"
__license__  = "GNU LGPL Version 2.1"

# Modified by Anders Logg, 2008.
# Modified by Garth N. Wells, 2009.

from dolfin import *

# Read the mesh from file
mesh =  Mesh("../mesh.xml.gz")

# Create FunctionSpace for pressure field
Vp = FunctionSpace(mesh, "CG", 1)
p = Function(Vp, "../pressure.xml.gz")

# Define sub domain for the dolphin
class Fish(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] > DOLFIN_EPS and x[0] < (1.0 - DOLFIN_EPS) and \
               x[1] > DOLFIN_EPS and x[1] < (1.0 - DOLFIN_EPS)

# Define functionals for drag and lift
n = FacetNormal(mesh)
D = -p*n[0]*ds
L = p*n[1]*ds

# Assemble functionals over sub domain
fish = Fish()
drag = assemble(D, mesh=mesh, exterior_facet_domains=fish)
lift = assemble(L, mesh=mesh, exterior_facet_domains=fish)

print "Lift: %f" %lift
print "Drag: %f" %drag
