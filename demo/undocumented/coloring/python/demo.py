""" This demo colors the cells of a mesh such that cells with the same
color are not neighbors. 'Neighbors' can be in the sense of shared
vertices, edges or facets.
"""

__author__ = "Garth N. Wells (gnw20@cam.ac.uk)"
__date__ = "2010-11-16"
__copyright__ = "Copyright (C) 2010 Garth N. Wells"
__license__  = "GNU LGPL Version 2.1"

# Modified by Anders Logg, 2010.
# Last changed: 2010-11-17

from dolfin import *

if not has_trilinos():
    print "DOLFIN has not been configured with Trilinos. Exiting."
    exit()

# Create mesh
mesh = UnitCube(24, 24, 24)

# Compute vertex-based coloring
colors = mesh.color("vertex")
plot(colors, title="Vertex-based cell coloring", interactive=True)

# Compute edge-based coloring
colors = mesh.color("edge")
plot(colors, title="Edge-based cell coloring", interactive=True)

# Compute facet-based coloring
colors = mesh.color("facet")
plot(colors, title="Facet-based cell coloring", interactive=True)
