# Copyright (C) 2009 Anders Logg
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
# First added:  2009-11-11
# Last changed: 2011-02-25

from dolfin import *
from numpy import ones

not_working_in_parallel("This demo")

# Create empty time series
series = TimeSeries("primal")

# Create a mesh and a vector
mesh = UnitSquare(2, 2)
x = Vector()

# Add a bunch of meshes and vectors to the series
t = 0.0
while t < 1.0:

    # Refine mesh and resize vector
    mesh = refine(mesh);
    x.resize(mesh.num_vertices());

    # Set some vector values
    x[:] = ones(x.size())

    # Append to series
    series.store(mesh, t)
    series.store(x, t)

    t += 0.2

# Retrieve mesh and vector at some point in time
series.retrieve(mesh, 0.29)
series.retrieve(x, 0.31, False)

# Plot mesh
plot(mesh, interactive=True)
