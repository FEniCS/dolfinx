"This demo illustrates mesh refinement."

# Copyright (C) 2007-2009 Anders Logg
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later


from dolfin import *
import matplotlib.pyplot as plt


# Create mesh of unit square
mesh = UnitSquareMesh(8, 8)
plt.figure(1)
plot(mesh)

info(mesh)
print()

# Uniform refinement
mesh = refine(mesh)
plt.figure(2)
plot(mesh)

info(mesh)
print()

# Uniform refinement
mesh = refine(mesh)
plt.figure(3)
plot(mesh)

info(mesh)
print()

# Refine mesh close to x = (0.5, 0.5)
p = Point(0.5, 0.5)
for i in range(5):

    print("marking for refinement")

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
    plt.figure()
    plot(mesh)

plt.show()
