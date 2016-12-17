# Copyright (C) 2013, 2015 Anders Logg
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

from dolfin import *

# There is an issue with the Matplotlib backend that needs to be investigated

# Some parameters
L = 10.0
h = 0.1
M = 64
N = 32
num_steps = 1000
(x_B, y_B) = (0.75*L, 0.5*L)
(x_C, y_C) = (0.33*L, 0.75*L)
(dx_B, dy_B) = (2*h, h)
(dx_C, dy_C) = (h, 2*h)

# Create meshes: a box and two circles
mesh_A = RectangleMesh(Point(0, 0), Point(L, L), M, M)
mesh_B = Mesh("../circle.xml.gz")
mesh_C = Mesh("../circle.xml.gz")

# Displace circles to initial positions
mesh_B.translate(Point(x_B, y_B))
mesh_C.translate(Point(x_C, y_C))

# Create mesh function for plotting
f = CellFunction("uint", mesh_A)

# Build bounding box trees for background mesh
tree_A = BoundingBoxTree()
tree_A.build(mesh_A)

# Create bounding box trees for circles
tree_B = BoundingBoxTree()
tree_C = BoundingBoxTree()

# Loop over time steps
for n in range(num_steps):

    # Make it bounce
    x_B += dx_B; y_B += dy_B
    x_C += dx_C; y_C += dy_C
    if x_B > L - 1.0 or x_B < 1.0: dx_B = -dx_B
    if y_B > L - 1.0 or y_B < 1.0: dy_B = -dy_B
    if x_C > L - 1.0 or x_C < 1.0: dx_C = -dx_C
    if y_C > L - 1.0 or y_C < 1.0: dy_C = -dy_C

    # Translate circles
    mesh_B.translate(Point(dx_B, dy_B))
    mesh_C.translate(Point(dx_C, dy_C))

    # Rebuild trees
    tree_B.build(mesh_B)
    tree_C.build(mesh_C)

    # Compute collisions
    entities_AB, entities_B = tree_A.compute_collisions(tree_B)
    entities_AC, entities_C = tree_A.compute_collisions(tree_C)
    entities_BC = set(entities_AB).intersection(set(entities_AC))

    # Mark mesh function
    f.set_all(0)
    for i in entities_AB:
        f.set_value(i, 1)
    for i in entities_AC:
        f.set_value(i, 2)
    for i in entities_BC:
        f.set_value(i, 3)

    # Plot
    #p = plot(f, wireframe=(n > num_steps / 2))
    #p.write_png("collision-detection-%.4d" % n)

#interactive()

# Generate movie using
# ffmpeg -r 25 -b 1800 -i collision-detection-%04d.png collision-detection.mp4
