__author__ = "Anders Logg and Magnus Vikstrom"
__date__ = "2007-06-01 -- 2007-06-01"
__copyright__ = "Copyright (C) 2007 Anders Logg and Magnus Vikstrom"
__license__  = "GNU LGPL Version 2.1"

from dolfin import *

# Create mesh
mesh = UnitCube(16, 16, 16)

# Partition mesh
partitions = MeshFunction("uint")
mesh.partition(20, partitions)

# Plot mesh partition
plot(partitions)
