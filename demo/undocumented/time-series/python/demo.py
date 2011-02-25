__author__ = "Anders Logg (logg@simula.no)"
__date__ = "2009-11-11"
__copyright__ = "Copyright (C) 2009 Anders Logg"
__license__  = "GNU LGPL Version 2.1"

# Last changed: 2011-02-25

from dolfin import *

set_log_level(DEBUG)

not_working_in_parallel("This demo")

# Create empty time series
series = TimeSeries("primal")

# Create a mesh and a vector
mesh = UnitSquare(2, 2)
x = Vector()

# Add a bunch of meshes and vectors to the series
t = 1.0
while t > 0.0:

    # Refine mesh and resize vector
    mesh = refine(mesh);
    x.resize(mesh.num_vertices())

    # Append to series
    series.store(mesh, t)
    series.store(x, t)

    print t

    t -= 0.2

# Retrieve mesh and vector at some point in time
series.retrieve(mesh, 0.29)
series.retrieve(x, 0.31)

# Plot mesh
#plot(mesh, interactive=True)
