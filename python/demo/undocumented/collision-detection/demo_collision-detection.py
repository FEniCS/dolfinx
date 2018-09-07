# Copyright (C) 2013, 2015 Anders Logg
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import numpy
from dolfin import *
from dolfin.io import XDMFFile
from dolfin import cpp

# There is an issue with the Matplotlib backend that needs to be investigated

# Some parameters
L = 10.0
h = 0.1
M = 64
N = 32
num_steps = 1000
x_b = numpy.array((0.75*L, 0.5*L))
x_c = numpy.array((0.33*L, 0.75*L))
dx_b = numpy.array((2*h, h))
dx_c = numpy.array((h, 2*h))

# Create meshes: a box and two circles
mesh_A = RectangleMesh.create(MPI.comm_world, [Point(0, 0), Point(L, L)], [M, M],
                              CellType.Type.triangle, cpp.mesh.GhostMode.none)
xdmf = XDMFFile(MPI.comm_world, "circle.xdmf")
mesh_B = xdmf.read_mesh(MPI.comm_world, cpp.mesh.GhostMode.none)
mesh_C = xdmf.read_mesh(MPI.comm_world, cpp.mesh.GhostMode.none)

# Displace circles to initial positions

mesh_B.geometry.points += x_b
mesh_C.geometry.points += x_c

# Create mesh function for plotting
f = MeshFunction("size_t", mesh_A, mesh_A.topology.dim, 0)

# Build bounding box trees for background mesh
tree_A = BoundingBoxTree(mesh_A.geometry.dim)
tree_A.build(mesh_A, mesh_A.topology.dim)

# Create bounding box trees for circles
tree_B = BoundingBoxTree(mesh_B.geometry.dim)
tree_C = BoundingBoxTree(mesh_C.geometry.dim)

# Loop over time steps
for n in range(num_steps):

    # Make it bounce
    x_b += dx_b
    x_c += dx_c

    if x_b[0] > L - 1.0 or x_b[0] < 1.0: dx_b[0] = -dx_b[0]
    if x_b[1] > L - 1.0 or x_b[1] < 1.0: dx_b[1] = -dx_b[1]
    if x_c[0] > L - 1.0 or x_c[0] < 1.0: dx_c[0] = -dx_c[0]
    if x_c[1] > L - 1.0 or x_c[1] < 1.0: dx_c[1] = -dx_c[1]

    # Translate circles
    mesh_B.geometry.points += dx_b
    mesh_C.geometry.points += dx_c

    # Rebuild trees
    tree_B.build(mesh_B, mesh_B.topology.dim)
    tree_C.build(mesh_C, mesh_C.topology.dim)

    # Compute collisions
    entities_AB, entities_B = tree_A.compute_collisions(tree_B)
    entities_AC, entities_C = tree_A.compute_collisions(tree_C)
    entities_BC = set(entities_AB).intersection(set(entities_AC))

    # Mark mesh function
    f.set_all(0)
    for i in entities_AB:
        f[int(i)] = 1
    for i in entities_AC:
        f[int(i)] = 2
    for i in entities_BC:
        f[int(i)] = 3
