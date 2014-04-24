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
# First added:  2014-04-07
# Last changed: 2014-04-24

from dolfin import *
import pylab as pl

# Colors for plotting
red    = "#ff3c00"
green  = "#59ce55"
yellow = "#fff0aa"
blue   = "#b4d8e7"
white  = "#ffffff"
black  = "#000000"

# Function for plotting a triangle (using matplotlib)
def plot_triangle(c, mesh, color=None, alpha_fill=None, alpha_line=None):
    cell = Cell(mesh, c)
    xy = cell.get_vertex_coordinates()
    x = [xy[0], xy[2], xy[4]]
    y = [xy[1], xy[3], xy[5]]
    if not color is None: pl.fill(x, y, color=color, alpha=alpha_fill)
    pl.plot(x + [x[0]], y + [y[0]], color='k', alpha=alpha_line)

# Parameters
N  = 20
dv = 5
R  = 0.6
num_frames = 3

# Uncomment to generate a longer movie
#num_frames = 360

# Create and load meshes
mesh = RectangleMesh(-R, -R, R, R, N, N)
propeller = Mesh("../propeller_2d_coarse.xml.gz")

# Iterate over rotations
for frame in range(num_frames):

    print "Frame %d out of %d..." % (frame + 1, num_frames)

    # Rotate propeller
    propeller.rotate(dv)

    # Build multimesh
    multimesh = MultiMesh()
    multimesh.add(mesh)
    multimesh.add(propeller)
    multimesh.build()

    # Extract data
    cut_cells = multimesh.cut_cells(0)
    uncut_cells = multimesh.uncut_cells(0)
    covered_cells = multimesh.covered_cells(0)

    # Clear plot
    pl.clf()

    # Plot cells in background mesh
    for c in cut_cells:
        plot_triangle(c, mesh, yellow)
    for c in uncut_cells:
        plot_triangle(c, mesh, blue)
    for c in covered_cells:
        plot_triangle(c, mesh, white, alpha_line=0.1)

    # Plot propeller mesh
    for c in range(propeller.num_cells()):
        plot_triangle(c, propeller, color=white, alpha_fill=0.25)

    # Plot quadrature points
    for c in cut_cells:
        points, weights = multimesh.quadrature_rule_cut_cell(0, c)
        for i in range(len(weights)):
            w = weights[i]
            x = points[2*i]
            y = points[2*i + 1]
            if w > 0:
                pl.plot(x, y, '.', markersize=5, color=red)
            else:
                pl.plot(x, y, '.', markersize=2, color=black)

    pl.axis("equal")
    pl.axis("off")
    pl.savefig("multimesh_propeller_%.4d.png" % frame, dpi=300)

# Show last frame
pl.show()

# Generate movie using
#ffmpeg -r 25 -b 1800 -i multimesh_propeller_%04d.png multimesh_propeller.mp4
