__author__ = "Anders Logg (logg@simula.no)"
__date__ = "2009-11-11"
__copyright__ = "Copyright (C) 2009 Anders Logg"
__license__  = "GNU LGPL Version 2.1"

# Last changed: 2011-02-25

from dolfin import *
from numpy import ones

not_working_in_parallel("This demo")

# Create empty time series
series = TimeSeries("primal")

# Create a mesh and a vector
mesh = UnitSquare(2, 2)
x = Vector(10)
x[:] = ones(x.size())

# Add a bunch of meshes and vectors to the series
t = 1.0
while t < 1.0:

    # Refine mesh
    mesh = refine(mesh);

    # Scale vector
    x *= 2.0

    # Append to series
    series.store(mesh, t)
    series.store(x, t)

    t += 0.2

# Retrieve mesh and vector at some point in time
series.retrieve(mesh, 0.29)
series.retrieve(x, 0.31)

# Plot mesh and print vector
plot(mesh, interactive=True)
info(x, True)
