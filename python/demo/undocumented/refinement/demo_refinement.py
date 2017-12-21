"This demo illustrates mesh refinement."

# Copyright (C) 2007-2009 Anders Logg
#
# This file is part of DOLFIN.
#
# DOLFIN is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DOLFIN is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
#
# First added:  2007-06-01
# Last changed: 2012-11-12


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
