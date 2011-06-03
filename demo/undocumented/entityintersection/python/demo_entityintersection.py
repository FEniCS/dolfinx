"""This demo shows the intersection between entities of two completely unrelated meshes.
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
# First added:  2010-03-02
# Last changed: 2010-03-02

from dolfin import *

if not has_cgal():
    print "DOLFIN must be compiled with CGAL to run this demo."
    exit(0)

cube = UnitCube(3,3,2)
sphere = UnitSphere(3)

#Low level interface, direct access to do_intersect.
#exact variant without floating point errors is provided by
#"do_intersect_exact" function.

print "Total number of cells in Cube: %i" % cube.num_cells() 
print "Total number of cells in Sphere: %i" % sphere.num_cells()
print """Intersecting pairwise cells of a cube and sphere mesh
Cube cell index | Sphere cell index
------------------------------"""

#@todo: Lowlevel interface in python is by a factor 10 or more faster!
#Investigate!!!

for c1 in cells(cube):
  for c2 in cells(sphere):
    if do_intersect(c1,c2):
      print "%i | %i" % (c1.index(),c2.index())

#High level interface via Meshentity intersects member function.
for c1 in cells(cube):
  for c2 in cells(sphere):
    if c1.intersects(c2):
      print "%i | %i" % (c1.index(),c2.index())
