# This demo demonstrates how to compute functionals (or forms
# in general) over subsets of the mesh. The two functionals
# lift and drag are computed for the pressure field around
# a dolphin. Here, we use the pressure field obtained from
# solving the Stokes equations (see demo program in the
# sub directory src/demo/pde/stokes/taylor-hood).
#
# Original implementation: ../cpp/main.cpp by Anders Logg
#
__author__ = "Kristian B. Oelgaard (k.b.oelgaard@tudelft.nl)"
__date__ = "2007-11-14 -- 2007-11-28"
__copyright__ = "Copyright (C) 2007 Kristian B. Oelgaard"
__license__  = "GNU LGPL Version 2.1"

from dolfin import *

# FIXME: Not working, see notice below
import sys
print "This demo is not working, please fix me"
sys.exit(1)

# Create element
element = FiniteElement("Lagrange", "triangle", 1)

# Read velocity field from file and get the mesh
p = Function(element, "../pressure.xml.gz")
mesh =  p.mesh()

# Define sub domain for the dolphin
class Fish(SubDomain):
    def inside(self, x, on_boundary):
        return bool(x[0] > DOLFIN_EPS and x[0] < (1.0 - DOLFIN_EPS) and 
              x[1] > DOLFIN_EPS and x[1] < (1.0 - DOLFIN_EPS) and
              on_boundary)


n = FacetNormal("triangle", mesh)

# Functionals for drag and lift
D = -p*n[0]*ds
L =  p*n[1]*ds

# Assemble functionals over sub domain
fish =  Fish()

# FIXME: Need to assemble over subdomain, but not yet supported in Python

drag = assemble(D, mesh, fish)
lift = assemble(L, mesh, fish)

print "Lift: %f" %lift
print "Drag: %f" %drag




