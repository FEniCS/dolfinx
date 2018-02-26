# Copyright (C) 2009 Anders Logg
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from dolfin import *
from numpy import ones
import matplotlib.pyplot as plt


# TimeSeries requires DOLFIN to be configured with HDF5
if has_hdf5() is False:
    print("This demo requires DOLFIN to be configured with HDF5")
    exit()

# Create empty time series
series = TimeSeries(MPI.comm_world, "primal")

# Create a mesh and a vector
mesh = UnitSquareMesh(2, 2)
x = Vector()

# Add a bunch of meshes and vectors to the series
t = 0.0
while t < 1.0:

    # Refine mesh
    mesh = refine(mesh);

    # Set some vector values
    x = Vector(mesh.mpi_comm(), mesh.num_vertices());
    x[:] = ones(x.size())

    # Append to series
    series.store(mesh, t)
    series.store(x, t)

    t += 0.2

# Retrieve mesh and vector at some point in time
series.retrieve(mesh, 0.29)
x = Vector()
series.retrieve(x, 0.31, False)

# Plot mesh
plot(mesh)
plt.show()
