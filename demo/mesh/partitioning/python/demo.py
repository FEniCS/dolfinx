__author__ = "Anders Logg and Magnus Vikstrom"
__date__ = "2007-06-01 -- 2008-04-11"
__copyright__ = "Copyright (C) 2007-2008 Anders Logg and Magnus Vikstrom"
__license__  = "GNU LGPL Version 2.1"

from dolfin import *

# Create mesh
mesh = UnitCube(16, 16, 16)

# Partition mesh
partitions = MeshFunction("uint")
try:
    mesh.partition(partitions, 20)
except:
    print "Sorry, this demo requires SCOTCH."

# Plot mesh partition
plot(partitions, interactive=True)
