from dolfin import *
from viper import *

# Create mesh
mesh = UnitSquare(8, 8)

# FIXME: Should work just like this, need to call partitions.init(...)
# MeshFunction<unsigned int> partitions;

# Partition mesh
partitions = MeshFunction('uint', mesh, 2);
mesh.partition(10, partitions);

# Plot mesh partition
plot(partitions);
