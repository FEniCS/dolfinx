""" This demo colors the cells of a mesh such that cells with the same
color are not neighbors. 'Neighbors' can be in the sense of shared
vertices, edges or facets.
"""

__author__ = "Garth N. Wells (gnw20@cam.ac.uk)"
__date__ = "2010-11-16"
__copyright__ = "Copyright (C) 2010 Garth N. Wells"
__license__  = "GNU LGPL Version 2.1"

from dolfin import *

if not has_trilinos():
    print "DOLFIN has not been configured with Trilinos. Exiting."
    exit()

# Create mesh
mesh = UnitCube(24, 24, 24)

# Create a vertex-based coloring object and color cells
coloring = CellColoring(mesh, "vertex")
colors = coloring.compute_local_cell_coloring()
#plot(colors, title="Vertex-based cell coloring", interactive=True)

# Create a edge-based coloring object and color cells
coloring = CellColoring(mesh, "edge")
colors = coloring.compute_local_cell_coloring()
#plot(colors, title="Edge-based cell coloring", interactive=True)

# Create a facet-based coloring object and color cells
coloring = CellColoring(mesh, "facet")
colors = coloring.compute_local_cell_coloring()
#plot(colors, title="Facet-based cell coloring", interactive=True)
