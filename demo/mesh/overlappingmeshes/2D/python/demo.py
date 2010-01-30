
__author__ = "Andre Massing (massing@simula.no)"
__date__ = "2010-01-27 "
__copyright__ = "Copyright (C) 2010 Andre Massing"
__license__  = "GNU LGPL Version 2.1"

from dolfin import *
from numpy import *

if not has_cgal():
    print "DOLFIN must be compiled with CGAL to run this demo."
    exit(0)

# Create meshes (omega0 overlapped by omega1)
#mesh_1 = UnitCircle(20)
mesh_1 = UnitSquare(10, 10)

mesh_2 = UnitSquare(10, 10)

# Access mesh geometry
x = mesh_2.coordinates()

# Move and scale second mesh
#x *= 0.5
x += 0.5

overlap = OverlappingMeshes(mesh_1,mesh_2)
overlap.compute_overlap_map()

overlapped_domain = overlap.overlapped_domain();
print overlapped_domain

p = plot(overlapped_domain, rescale=False)

interactive()
