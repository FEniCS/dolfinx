__author__ = "Anders Logg (logg@simula.no)"
__date__ = "2007-05-29 -- 2007-05-30"
__copyright__ = "Copyright (C) 2007 Anders Logg"
__license__  = "GNU GPL Version 2"

from dolfin import *
from math import sqrt

print "Press q to continue..."

# Read and plot mesh from file
mesh = Mesh("dolfin-2.xml.gz")
plot(mesh, interactive=False)

# Have some fun with the mesh
R = 0.15
H = 0.025
X = 0.3
Y = 0.4
dX = H
dY = 1.5*H
coordinates = mesh.coordinates()
original = coordinates.copy()
for i in xrange(500):

    if X < H or X > 1.0 - H:
        dX = -dX
    if Y < H or Y > 1.0 - H:
        dY = -dY
    X += dX
    Y += dY

    for j in xrange(mesh.numVertices()):
        x, y = coordinates[j]
        r = sqrt((x - X)**2 + (y - Y)**2)
        if r < R:
            coordinates[j] = [X + (r/R)**2*(x - X), Y + (r/R)**2*(y - Y)]

    update(mesh)

    for j in xrange(mesh.numVertices()):
        coordinates[j] = original[j]
