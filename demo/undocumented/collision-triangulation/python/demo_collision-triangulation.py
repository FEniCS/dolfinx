# Copyright (C) 2014 Anders Logg
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
# First added:  2014-01-07
# Last changed: 2014-01-08

import numpy
from dolfin import *

# Creating a mesh from a triangulation (for visualization). Note that
# this function completely disregards common vertices and creates a
# completely disconnected mesh.
def triangulation_to_mesh_2d(triangulation):
    editor = MeshEditor()
    mesh = Mesh()
    editor.open(mesh, 2, 2)
    num_cells = len(triangulation) / 6
    num_vertices = len(triangulation) / 2
    editor.init_cells(num_cells)
    editor.init_vertices(num_vertices)
    for i in xrange(num_cells):
        editor.add_cell(i, 3*i, 3*i + 1, 3*i + 2)
    for i in xrange(num_vertices):
        editor.add_vertex(i, triangulation[2*i], triangulation[2*i + 1])
    editor.close()
    return mesh

# Number of steps
num_steps = 1000

# Create meshes
mesh_A = UnitSquareMesh(4, 4)
mesh_B = UnitSquareMesh(4, 4)

# Create plotter (want triangulations to appear in the same window)
plotter = VTKPlotter(mesh_A)

# Loop over time steps
for n in range(num_steps):

    print "Step %d / %d" % (n + 1, num_steps)

    # Rotate overlapping mesh
    mesh_B.rotate(3)

    # Triangulate collisions
    triangulation = numpy.array([])
    for cell_A in cells(mesh_A):
        for cell_B in cells(mesh_B):
            T = cell_A.triangulate_intersection(cell_B)
            triangulation = numpy.append(triangulation, T)

    # Create mesh from triangulation
    triangulation = triangulation_to_mesh_2d(triangulation)

    # Plot triangulation
    plotter.plot(triangulation)
    #plotter.write_png("collision-triangulation-%.4d" % n)

# Hold plot
interactive()

# Generate movie using
# ffmpeg -r 25 -b 1800 -i collision-triangulation-%04d.png collision-triangulation.mp4
