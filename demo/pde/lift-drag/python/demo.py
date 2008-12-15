"""This demo demonstrates how to compute functionals (or forms in
general) over subsets of the mesh. The two functionals lift and drag
are computed for the pressure field around a dolphin. Here, we use the
pressure field obtained from solving the Stokes equations (see demo
program in the sub directory src/demo/pde/stokes/taylor-hood)."""

__author__ = "Kristian B. Oelgaard (k.b.oelgaard@tudelft.nl)"
__date__ = "2007-11-14 -- 2008-12-07"
__copyright__ = "Copyright (C) 2007 Kristian B. Oelgaard"
__license__  = "GNU LGPL Version 2.1"

# Modified by Anders Logg, 2008.

from dolfin import *

# FIXME: Not working, see notice below
import sys
#print "This demo is not working, please fix me"
#sys.exit(1)

# Read velocity field from file and get the mesh
u = Function("../velocity.xml.gz")
p = Function("../pressure.xml.gz")
mesh =  p.function_space().mesh()

# Define sub domain for the dolphin
class Fish(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] > DOLFIN_EPS and x[0] < (1.0 - DOLFIN_EPS) and \
               x[1] > DOLFIN_EPS and x[1] < (1.0 - DOLFIN_EPS)

# Define strain-rate tensor
def epsilon(u):
    return 0.5*(grad(u) + transp(grad(u)))

# Define stress tensor
def sigma(u, p):
    nu = 1.0
    return mult(2.0*nu, epsilon(u)) - mult(p, Identity(len(u)))

# Define functionals for drag and lift
n = FacetNormal(mesh)

# FIXME: Defining the lift and drag functional using both pressure and velocity
#        from file won't work, as the two meshes are not the "same" meshes.
D = mult(sigma(u, p), -n)[0]*ds
L = mult(sigma(u, p), -n)[1]*ds

# Use only the Pressure field
D = p*n[0]*ds
L = p*n[1]*ds

# Assemble functionals over sub domain
fish =  Fish()
drag = assemble(D, exterior_facet_domains = fish)
lift = assemble(L, exterior_facet_domains = fish)

print "Lift: %f" %lift
print "Drag: %f" %drag
