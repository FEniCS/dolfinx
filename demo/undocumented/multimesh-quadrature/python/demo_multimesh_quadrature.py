# Copyright (C) 2014-2017 Anders Logg
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
# Last changed: 2017-05-17
#
# This demo program illustrates multimesh quadrature on a pair
# of overlapping meshes using red dots to mark positive quadature
# weights and black dots to mark negative quadature weights.
#
# To create movies from the generated PNG files, enter the output
# directory and run the following commmands:
# 
# ffmpeg -i multimesh_quadrature_%04d.png multimesh_quadrature.mp4
# ffmpeg -i multimesh_quadrature_%04d.png multimesh_quadrature_compressed.mp4

from dolfin import *
import os

# Don't plot when DOLFIN_NOPLOT is set
plot = os.environ.get("DOLFIN_NOPLOT") is None
if plot:
    import pylab as pl

# Colors for plotting
red    = "#ff3c00"
green  = "#59ce55"
yellow = "#fff0aa"
blue   = "#b4d8e7"
white  = "#ffffff"
black  = "#000000"

def plot_triangle(c, mesh, color=None, alpha_fill=None, alpha_line=None):
    if not plot: return
    cell = Cell(mesh, c)
    xy = cell.get_vertex_coordinates()
    x = [xy[0], xy[2], xy[4]]
    y = [xy[1], xy[3], xy[5]]
    if not color is None: pl.fill(x, y, color=color, alpha=alpha_fill)
    pl.plot(x + [x[0]], y + [y[0]], color='k', alpha=alpha_line)

def plot_point_red(x, y):
    if not plot: return
    pl.plot(x, y, '.', markersize=5, color=red)

def plot_point_black(x, y):
    if not plot: return
    pl.plot(x, y, '.', markersize=2, color=black)

def clear_plot():
    if not plot: return
    pl.clf()

def save_plot(frame, compress):
    if not plot: return
    if not os.path.isdir("output"): os.makedirs("output")
    pl.axis("equal")
    pl.axis("off")
    c = "_compressed" if compress else ""
    pl.savefig("output/multimesh_quadrature%s_%.4d.png" % (c, frame), dpi=300)

def show_plot():
    if not plot: return
    pl.show()

# Parameters
N = 8
dv = 5
R = 0.6
num_frames = 3

# Uncomment to generate a longer movie
#num_frames = 360

# Iterate over quadrature compression on/off
for compress in [True, False]:
	print("Compression:", compress)

	# Create and load meshes
	mesh = RectangleMesh(Point(-R, -R), Point(R, R), N, N)
	propeller = Mesh("../propeller_2d_coarse.xml.gz")

	# Iterate over frames (rotations of the propeller)
	for frame in range(num_frames):
	    print("Frame %d out of %d..." % (frame + 1, num_frames))

	    # Rotate propeller
	    propeller.rotate(dv)

	    # Build multimesh
	    multimesh = MultiMesh()
	    multimesh.parameters.compress_volume_quadrature = compress
	    multimesh.add(mesh)
	    multimesh.add(propeller)
	    multimesh.build()

	    # Extract data
	    cut_cells = multimesh.cut_cells(0)
	    uncut_cells = multimesh.uncut_cells(0)
	    covered_cells = multimesh.covered_cells(0)

	    # Clear plot
	    clear_plot()

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
	        points, weights = multimesh.quadrature_rules_cut_cells(0, c)
	        for i in range(len(weights)):
	            w = weights[i]
	            x = points[2*i]
	            y = points[2*i + 1]
	            if w > 0:
	                plot_point_red(x, y)
	            else:
	                plot_point_black(x, y)

	    # Save plot
	    save_plot(frame, compress)

# Show last frame
show_plot()
