"""This demo demonstrates how to move the vertex coordinates
of a boundary mesh and then updating the interior vertex
coordinates of the original mesh by suitably interpolating
the vertex coordinates (useful for implementation of ALE
methods)."""

__author__ = "Solveig Bruvoll (solveio@ifi.uio.no) and Anders Logg (logg@simula.no)"
__date__ = "2008-05-02 -- 2008-05-28"
__copyright__ = "Copyright (C) 2008 Solveig Bruvoll and Anders Logg"
__license__  = "GNU LGPL Version 2.1"

from dolfin import *

# Create mesh
mesh = UnitSquare(20, 20)

# Create boundary mesh
boundary = BoundaryMesh(mesh)

# Move vertices in boundary
for x in boundary.coordinates():
    x[0] *= 3.0
    x[1] += 0.1*sin(5.0*x[0])

# Move mesh
mesh.move(boundary, hermite)

# Plot mesh
plot(mesh, interactive=True)
