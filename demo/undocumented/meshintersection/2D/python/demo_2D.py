"""This demo shows the intersection of the boundary of a unit square
(omega1) with a unit circle (omega2) rotating around the center of the
square.
This demo aims at the rotator in cavity problem and demonstrates
the intersection of the rotator mesh with a square.
background mesh.
"""

# Copyright (C) 2008 Kristoffer Selim
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
# Modified by Andre Massing
# Modified by Benjamin Kehlet 2012
#
# First added:  2008-10-14
# Last changed: 2012-07-19

from dolfin import *
from numpy import max, array

if not has_cgal():
    print "DOLFIN must be compiled with CGAL to run this demo."
    exit(0)

# Create meshes (omega0 overlapped by omega1)
omega0 = UnitCircle(5)
omega1 = UnitSquare(5, 5)

# Access mesh geometry
x = omega0.coordinates()

# Move and scale circle
x *= 0.5
x += 1.0

# Iterate over angle
theta = 0.0
dtheta = 0.05*DOLFIN_PI
intersection = MeshFunction("sizet", omega0, omega0.topology().dim())
_first = True

p = VTKPlotter(intersection)
p.parameters["rescale"] = True
p.parameters["wireframe"] = False
#p.parameters["axes"] = True
p.parameters["scalarbar"] = False


p.add_polygon(array([0.0, 0.0,
                     1.0, 0.0,
                     1.0, 1.0,
                     0.0, 1.0,
                     0.0, 0.0]))

while theta < 2*DOLFIN_PI + dtheta:
    # Compute intersection with boundary of square
    boundary = BoundaryMesh(omega1)
    cells = omega0.intersected_cells(boundary)

    # Mark intersected values
    intersection.array()[:] = 0
    intersection.array()[cells] = 1

    p.plot()

    # Rotate circle around (0.5, 0.5)
    xr = x[:, 0].copy() - 0.5
    yr = x[:, 1].copy() - 0.5
    x[:,0] = 0.5 + (cos(dtheta)*xr - sin(dtheta)*yr)
    x[:,1] = 0.5 + (sin(dtheta)*xr + cos(dtheta)*yr)
    omega0.intersection_operator().clear()

    theta += dtheta

# Repeat the same with the rotator in the cavity example.
background_mesh = Rectangle(-2.0, -2.0, 2.0, 2.0, 30, 30)
structure_mesh = Mesh("../rotator.xml.gz")

# Access mesh geometry
x = structure_mesh.coordinates()
print "Maximum value is max(x()): ", max(x)

# Iterate over angle
theta = 0.0
dtheta = 0.1*DOLFIN_PI
intersection = MeshFunction("sizet", background_mesh, background_mesh.topology().dim())
_first = True

while theta < 2*DOLFIN_PI + dtheta:

  cells = background_mesh.intersected_cells(structure_mesh)

  # Mark intersected values
  intersection.array()[:] = 0
  intersection.array()[cells] = 1

  plot(intersection, rescale=True, wireframe=True)

  # Rotate rotator
  xr = x[:, 0].copy()
  yr = x[:, 1].copy()

  x[:,0] = (cos(dtheta)*xr - sin(dtheta)*yr)
  x[:,1] = (sin(dtheta)*xr + cos(dtheta)*yr)

  theta += dtheta

# Hold plot
interactive()
