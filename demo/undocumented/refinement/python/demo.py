"This demo illustrates mesh refinement."

__author__ = "Anders Logg"
__date__ = "2007-06-01 -- 2009-10-08"
__copyright__ = "Copyright (C) 2007-2009 Anders Logg"
__license__  = "GNU LGPL Version 2.1"

from dolfin import *

# Create mesh of unit square
mesh = UnitSquare(32, 32)
plot(mesh)

# Uniform refinement
mesh = refine(mesh)
plot(mesh)

# Uniform refinement
mesh = refine(mesh)
plot(mesh)

print "check"

# Refine mesh close to x = (0.5, 0.5)
p = Point(0.5, 0.5)
for i in range(5):

    print "marking for refinement"

    # Mark cells for refinement
    cell_markers = MeshFunction("bool", mesh, mesh.topology().dim())
    for c in cells(mesh):
        if c.midpoint().distance(p) < 0.1:
            cell_markers[c] = True
        else:
            cell_markers[c] = False

    # Refine mesh
    mesh = refine(mesh, cell_markers)

    # Plot mesh
    plot(mesh)

interactive()
