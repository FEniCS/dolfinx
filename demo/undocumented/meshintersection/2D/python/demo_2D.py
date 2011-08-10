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
#
# First added:  2008-10-14
# Last changed: 2011-02-14

from dolfin import *
from numpy import max

if not has_cgal():
    print "DOLFIN must be compiled with CGAL to run this demo."
    exit(0)

# Set to False if you do not want to create movies
# (default should be True since you probably want to :)
create_movies = True

# Create meshes (omega0 overlapped by omega1)
omega0 = UnitCircle(20)
omega1 = UnitSquare(20, 20)

# Access mesh geometry
x = omega0.coordinates()

# Move and scale circle
x *= 0.5
x += 1.0

# Iterate over angle
theta = 0.0
dtheta = 0.05*DOLFIN_PI
intersection = MeshFunction("uint", omega0, omega0.topology().dim())
_first = True

while theta < 2*DOLFIN_PI + dtheta:

    # Compute intersection with boundary of square
    boundary = BoundaryMesh(omega1)
    cells = omega0.intersected_cells(boundary)

    # Mark intersected values
    intersection.array()[:] = 0
    intersection.array()[cells] = 1

    # Plot intersection
    if _first:
        p = plot(intersection, rescale=False)
#        p = plot(intersection, rescale=True, wireframe=False, axes=True,scalar_bar=False)
        p.add_polygon([[0, 0, -0.01],
                       [1, 0, -0.01],
                       [1, 1, -0.01],
                       [0, 1, -0.01],
                       [0, 0, -0.01]])
        p.ren.ResetCamera()
        p.update(intersection)
        _first = False
        interactive()
    else:
        plot(intersection)

    p.update(intersection)
    if create_movies:
      p.write_png()

    # Rotate circle around (0.5, 0.5)
    xr = x[:, 0].copy() - 0.5
    yr = x[:, 1].copy() - 0.5
    x[:,0] = 0.5 + (cos(dtheta)*xr - sin(dtheta)*yr)
    x[:,1] = 0.5 + (sin(dtheta)*xr + cos(dtheta)*yr)
    omega0.intersection_operator().clear()

    theta += dtheta

if create_movies:
  p.movie("circle_square_intersection.avi", cleanup=True)

# Hold plot
interactive()

# Repeat the same with the rotator in the cavity example.
background_mesh = Rectangle(-2.0, -2.0, 2.0, 2.0, 30, 30)
structure_mesh = Mesh("rotator.xml.gz")

# Access mesh geometry
x = structure_mesh.coordinates()
print "Maximum value is max(x()): ", max(x)

# Iterate over angle
theta = 0.0
dtheta = 0.1*DOLFIN_PI
intersection = MeshFunction("uint", background_mesh, background_mesh.topology().dim())
_first = True

while theta < 2*DOLFIN_PI + dtheta:

  cells = background_mesh.intersected_cells(structure_mesh)

  # Mark intersected values
  intersection.array()[:] = 0
  intersection.array()[cells] = 1

  if _first :
    q = plot(intersection, rescale=True, wireframe=True, warpscalar=False)
    q = plot(intersection, rescale=False, wireframe=True)
    q.ren.ResetCamera()
    _first = False

  else :
    plot(intersection)

  q.update(intersection)
  if create_movies:
    q.write_png()

  # Rotate rotator
  xr = x[:, 0].copy()
  yr = x[:, 1].copy()

  x[:,0] = (cos(dtheta)*xr - sin(dtheta)*yr)
  x[:,1] = (sin(dtheta)*xr + cos(dtheta)*yr)

  theta += dtheta

if create_movies:
  q.movie("rotator_cavity_intersection.avi", cleanup=True)

# Hold plot
interactive()
