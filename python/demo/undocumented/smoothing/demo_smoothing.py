"This demo program demonstrates various algorithms for mesh smoothing."

# Copyright (C) 2010, 2015 Anders Logg
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from dolfin import *
import matplotlib.pyplot as plt


parameters["refinement_algorithm"] = "plaza"

# Create rectangular mesh
mesh = RectangleMesh(Point(0.0, 0.0), Point(2.4, 0.4), 60, 10)

# Define a circular hole
center = Point(0.2, 0.2)
radius = 0.05
class Hole(SubDomain):

    def inside(self, x, on_boundary):
        r = sqrt((x[0] - center[0])**2 + (x[1] - center[0])**2)
        return r < 1.5*radius # slightly larger

    def snap(self, x):
        r = sqrt((x[0] - center[0])**2 + (x[1] - center[1])**2)
        if r < 1.5*radius:
            x[0] = center[0] + (radius / r)*(x[0] - center[0])
            x[1] = center[1] + (radius / r)*(x[1] - center[1])

# Mark hole and extract submesh
hole = Hole()
sub_domains = MeshFunction("size_t", mesh, mesh.topology().dim())
sub_domains.set_all(0)
hole.mark(sub_domains, 1)
mesh = SubMesh(mesh, sub_domains, 0)
mesh.snap_boundary(hole)

# Refine and snap mesh
plt.figure()
plot(mesh, title="Mesh 0")
num_refinements = 3
for i in range(num_refinements):

    # Mark cells for refinement
    markers = MeshFunction("bool", mesh, mesh.topology().dim())
    markers.set_all(False)
    for cell in cells(mesh):
        if cell.midpoint().distance(center) < 2*radius:
            markers[cell.index()] = True

    # Refine mesh
    mesh = refine(mesh, markers)

    # Snap boundary
    mesh.snap_boundary(hole)

    # Plot mesh
    plt.figure()
    plot(mesh, title=("Mesh %d" % (i + 1)))

plt.show()
