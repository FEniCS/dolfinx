"""This demo shows the intersection between entities of two completely unrelated meshes.
"""

__author__ = "Andre Massing (massing@simula.no)"
__date__ = "2010-03-02"
__copyright__ = "Copyright (C) 2009 Andre Massing"
__license__  = "GNU LGPL Version 2.1"


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
