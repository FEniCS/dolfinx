"""This demo demonstrates how to move the vertex coordinates of a
boundary mesh and then updating the interior vertex coordinates of the
original mesh by suitably interpolating the vertex coordinates (useful
for implementation of ALE methods)."""

# Copyright (C) 2008 Solveig Bruvoll and Anders Logg
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from dolfin import *
import matplotlib.pyplot as plt


# Create mesh
mesh = UnitSquareMesh(20, 20)

# Create boundary mesh
boundary = BoundaryMesh(mesh, "exterior")

# Move vertices in boundary
for x in boundary.coordinates():
    x[0] *= 3.0
    x[1] += 0.1*sin(5.0*x[0])

# Move mesh
ALE.move(mesh, boundary)

# Plot mesh
plot(mesh)
plt.show()

# Write mesh to file
File("deformed_mesh.pvd") << mesh
