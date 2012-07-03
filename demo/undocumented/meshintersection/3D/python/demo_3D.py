"""This demo shows the intersection of the boundary of a unit square
(omega1) with a unit circle (omega2) rotating around the center of the
square.
@todo Change camera perspective/ viewpoint to improve intersection visibility.
"""

# Copyright (C) 2009 Andre Massing
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
# First added:  2008-11-17
# Last changed: 2008-11-17

from dolfin import *
from numpy import *

if not has_cgal():
    print "DOLFIN must be compiled with CGAL to run this demo."
    exit(0)

#Set to False if you do not want to create movies
#(default should be True since you probably want to :)
create_movies = False

sphere = UnitSphere(20)
cube = UnitCube(20, 20, 20)

# Access mesh geometry
x = sphere.coordinates()

# Start center and propagtion speed for the sphere.
dt = 0.1
t = -0.61

# Scale and move the circle.
x *= 0.7
x[:] += t

intersection = MeshFunction("uint", cube, cube.topology().dim())
_first = True

counter = 0
while t < 1.4 :

    # Compute intersection with boundary of square
    boundary = BoundaryMesh(sphere)
    cells = cube.intersected_cells(boundary)

    # Mark intersected values
    intersection.array()[:] = 0
    intersection.array()[cells] = 1

    counter +=1

    # Plot intersection
    if _first:
        p = plot(intersection, rescale=True, wireframe=False, axes=True,scalar_bar=False)
        # FIXME: This needs to be updated to the new plotting functionality
        # p.elevate(-50)
        # p.azimuth(40)
        # p.update(intersection)
        _first = False
        interactive()

    else:
        plot(intersection)

    # FIXME: This needs to be updated to the new plotting functionality
    # p.update(intersection)
    # if create_movies:
    #   p.write_png()

    #Propagate sphere along the line t(1,1,1).
    x[:,0] += dt
    x[:,1] += dt
    x[:,2] += dt

    t += dt

# FIXME: This needs to be updated to the new plotting functionality
# if create_movies:
#   p.movie("sphere_cube_intersection.avi", cleanup=True)

# Hold plot
interactive()
