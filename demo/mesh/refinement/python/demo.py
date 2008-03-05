__author__ = "Anders Logg"
__date__ = "2007-06-01 -- 2007-06-01"
__copyright__ = "Copyright (C) 2007 Anders Logg"
__license__  = "GNU LGPL Version 2.1"

from dolfin import *

# Create mesh of unit square
mesh = UnitSquare(5, 5)
plot(mesh)

# Uniform refinement
mesh.refine()
plot(mesh)

# Refine mesh close to x = (0.5, 0.5)
p = Point(0.5, 0.5)
for i in range(5):

    # Mark cells for refinement
    cell_markers = MeshFunction("bool", mesh, mesh.topology().dim())
    for c in cells(mesh):
        if c.midpoint().distance(p) < 0.1:
            cell_markers.set(c, True)
        else:
            cell_markers.set(c, False)

    # Refine mesh
    mesh.refine(cell_markers)
    
    # Smooth mesh
    mesh.smooth()

    # Plot mesh
    plot(mesh)
