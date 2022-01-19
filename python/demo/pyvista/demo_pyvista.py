# Copyright (C) 2021-2022 Jørgen S. Dokken
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# ===============================
# Using pyvista for visualization
# ===============================


import numpy as np

import dolfinx.io
import dolfinx.plot
import ufl
from dolfinx.fem import (Function, FunctionSpace, LinearProblem,
                         VectorFunctionSpace)
from dolfinx.mesh import (CellType, MeshTags, compute_midpoints,
                          create_unit_cube, create_unit_square)

from mpi4py import MPI

try:
    import pyvista
except ModuleNotFoundError:
    print("pyvista is required for this demo")
    exit(0)

# If environment variable PYVISTA_OFF_SCREEN is set to true save a png
# otherwise create interactive plot
if pyvista.OFF_SCREEN:
    pyvista.start_xvfb(wait=0.1)

# Set some global options for all plots
transparent = False
figsize = 800
pyvista.rcParams["background"] = [0.5, 0.5, 0.5]

# Plotting a 2D dolfinx.Function with pyvista using warp by scalar
# ================================================================

# We start by creating a unit square mesh and interpolating a function into a first order Lagrange space
mesh = create_unit_square(MPI.COMM_WORLD, 12, 12, cell_type=CellType.quadrilateral)
V = FunctionSpace(mesh, ("Lagrange", 1))
u = Function(V)
u.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(2 * x[1] * np.pi))

# As we want to visualize the function u, we have to create a grid to attached the dof values to
# We do this by creating a topology and geometry based on the function space V
pyvista_cells, cell_types = dolfinx.plot.create_vtk_topology(V)
grid = pyvista.UnstructuredGrid(pyvista_cells, cell_types, mesh.geometry.x)

# We obtain the dof values from u and discard complex values if running in complex mode
# (as they are currently zero)
point_values = u.x.array.real if np.iscomplexobj(u.x.array) else u.x.array
grid.point_data["u"] = point_values

# We set the function "u" as the active scalar for the mesh, and warp
# the mesh in z-direction by its values
grid.set_active_scalars("u")
warped = grid.warp_by_scalar()

# We create a plotting window consisting of to plots, one of the scalar values, and one where
# the mesh is warped by these values
subplotter = pyvista.Plotter(shape=(1, 2))
subplotter.subplot(0, 0)
subplotter.add_text("Mesh with scalar function", font_size=14, color="black", position="upper_edge")
subplotter.add_mesh(grid, show_edges=True, show_scalar_bar=True)
subplotter.view_xy()

subplotter.subplot(0, 1)
subplotter.add_text("Warped function", position="upper_edge", font_size=14, color="black")
sargs = dict(height=0.8, width=0.1, vertical=True, position_x=0.05,
             position_y=0.05, fmt="%1.2e",
             title_font_size=40, color="black", label_font_size=25)
subplotter.set_position([-3, 2.6, 0.3])
subplotter.set_focus([3, -1, -0.15])
subplotter.set_viewup([0, 0, 1])
subplotter.add_mesh(warped, show_edges=True, scalar_bar_args=sargs)
if pyvista.OFF_SCREEN:
    subplotter.screenshot("2D_function_warp.png", transparent_background=transparent, window_size=[figsize, figsize])
else:
    subplotter.show()


# Plotting a 2D MeshTags and using subplots
# =========================================

# We continue using the mesh from the previous section, and find all
# cells satisfying the condition below


def in_circle(x):
    # Mark sphere with radius < sqrt(2)
    return np.array((x.T[0] - 0.5)**2 + (x.T[1] - 0.5)**2 < 0.2**2, dtype=np.int32)


# Create a dolfinx.MeshTag for all cells. If midpoint is inside the
# circle, it gets value 1, otherwise 0.
num_cells = mesh.topology.index_map(mesh.topology.dim).size_local
midpoints = compute_midpoints(mesh, mesh.topology.dim, list(np.arange(num_cells, dtype=np.int32)))
cell_tags = MeshTags(mesh, mesh.topology.dim, np.arange(num_cells), in_circle(midpoints))

# As the dolfinx.MeshTag contains a value for every cell in the
# geometry, we can attach it directly to the grid
grid.cell_data["Marker"] = cell_tags.values
grid.set_active_scalars("Marker")

# We create a plotter consisting of two windows, and add a plot of the
# Meshtags to the first window.
subplotter = pyvista.Plotter(shape=(1, 2))
subplotter.subplot(0, 0)
subplotter.add_text("Mesh with markers", font_size=14, color="black", position="upper_edge")
subplotter.add_mesh(grid, show_edges=True, show_scalar_bar=False)
subplotter.view_xy()

# We can also visualize subsets of data, by creating a smaller topology,
# only consisting of thos entities that has value one in the
# dolfinx.MeshTag
pyvista_cells, cell_types = dolfinx.plot.create_vtk_topology(
    mesh, mesh.topology.dim, cell_tags.indices[cell_tags.values == 1])

# We add this grid to the second plotter
sub_grid = pyvista.UnstructuredGrid(pyvista_cells, cell_types, mesh.geometry.x)
subplotter.subplot(0, 1)
subplotter.add_text("Subset of mesh", font_size=14, color="black", position="upper_edge")
subplotter.add_mesh(sub_grid, show_edges=True, edge_color="black")

if pyvista.OFF_SCREEN:
    subplotter.screenshot("2D_markers.png", transparent_background=transparent,
                          window_size=[2 * figsize, figsize])
else:
    subplotter.show()

# Plotting a dolfinx.fem.Function
# ===============================

# In the previous sections we have considered CG-1 spaces, which have a
# 1-1 correspondence with the vertices of the geometry. To be able to
# plot higher order function spaces, both CG and DG spaces, we have to
# adjust our plotting technique.

# We start by interpolating a discontinuous function into a second order DG
# space Note that we use the `cell_tags` from the previous section to
# get the cells for each of the regions
cells0 = cell_tags.indices[cell_tags.values == 0]
cells1 = cell_tags.indices[cell_tags.values == 1]
V = FunctionSpace(mesh, ("DG", 2))
uh = Function(V)
uh.interpolate(lambda x: x[0], cells0)
uh.interpolate(lambda x: x[1], cells1)

# To get a topology that has a 1-1 correspondence with the degrees of
# freedom in the function space, we call
# `dolfinx.plot.create_vtk_topology`.
topology, cell_types = dolfinx.plot.create_vtk_topology(V)
num_dofs_local = uh.function_space.dofmap.index_map.size_local
geometry = uh.function_space.tabulate_dof_coordinates()

# We discard the complex values if using PETSc in complex mode
values = uh.x.array.real if np.iscomplexobj(uh.x.array) else uh.x.array

# We create a pyvista mesh from the topology and geometry, and attach
# the coefficients of the degrees of freedom
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
grid.point_data["DG"] = values
grid.set_active_scalars("DG")

# We would also like to visualize the underlying mesh and obtain that as
# we have done previously
num_cells = mesh.topology.index_map(mesh.topology.dim).size_local
cell_entities = np.arange(num_cells, dtype=np.int32)
pyvista_cells, cell_types = dolfinx.plot.create_vtk_topology(mesh, mesh.topology.dim, cell_entities)
org_grid = pyvista.UnstructuredGrid(pyvista_cells, cell_types, mesh.geometry.x)


# We visualize the data
plotter = pyvista.Plotter()
plotter.add_text("Second order discontinuous elements",
                 position="upper_edge", font_size=14, color="black")
sargs = dict(height=0.1, width=0.8, vertical=False, position_x=0.1, position_y=0, color="black")
plotter.add_mesh(grid, show_edges=False, scalar_bar_args=sargs, line_width=0)
plotter.add_mesh(org_grid, color="white", style="wireframe", line_width=5)
plotter.add_mesh(grid.copy(), style="points", point_size=15, render_points_as_spheres=True, line_width=0)
plotter.view_xy()
if pyvista.OFF_SCREEN:
    plotter.screenshot(f"DG_{MPI.COMM_WORLD.rank}.png",
                       transparent_background=transparent, window_size=[figsize, figsize])
else:
    plotter.show()

# Plotting a vector-function with pyvista
# ===========================================

# The previous section has considered how to plot scalar valued functions.
# In this section we will consider how to plot vector valued functions.

mesh = create_unit_cube(MPI.COMM_WORLD, 4, 3, 5, cell_type=CellType.tetrahedron)

# We create a pyvista plotter
plotter = pyvista.Plotter()
plotter.add_text("Mesh and corresponding vectors",
                 position="upper_edge", font_size=14, color="black")

# Next, we create a pyvista.UnstructuredGrid based on the mesh
pyvista_cells, cell_types = dolfinx.plot.create_vtk_topology(mesh, mesh.topology.dim)
geometry = mesh.geometry.x
grid_0 = pyvista.UnstructuredGrid(pyvista_cells, cell_types, geometry)

# We add this grid (as a wireframe) to the plotter
plotter.add_mesh(grid_0, style="wireframe", line_width=2, color="black")

# We create a function space consisting of first order Nédélec (first kind) elements
# We want to interpolate a simple vector function


def int_u(x):
    return (x[2]**2, np.zeros(x.shape[1]), -x[0] * x[2])


element = ufl.FiniteElement("N1curl", mesh.ufl_cell(), 2)
V = FunctionSpace(mesh, element)
u = Function(V)
u.interpolate(int_u)


# We only support plotting of Lagrange and Discontinuous Lagrange functions.
# Therefore we interpolate the function into a first order Discontinuous Lagrange space.
output_element = ufl.VectorElement("DG", mesh.ufl_cell(), 2)
V_output = FunctionSpace(mesh, output_element)
u_out = Function(V_output)
u_out.interpolate(u)

# We create a second grid, whose geometry and topology is based on the output function space
cells_1, cell_types_1 = dolfinx.plot.create_vtk_topology(V_output)
geometry_1 = V_output.tabulate_dof_coordinates()
grid_1 = pyvista.UnstructuredGrid(cells_1, cell_types_1, geometry_1)

# If dolfinx is executed in complex mode, we discard the complex values
# and reshape it to match the 3D vector structure
output_values = u_out.x.array.real.reshape(geometry_1.shape[0], V_output.dofmap.index_map_bs)
# Create point cloud of vertices, and add the vertex values to the cloud
grid_1.point_data["u"] = output_values
glyphs = grid_1.glyph(orient="u", factor=0.1)

# We add in the glyphs corresponding to the plotter
plotter.add_mesh(glyphs)

# Save as png if we are using a container with no rendering
if pyvista.OFF_SCREEN:
    plotter.screenshot("3D_wireframe_with_vectors.png", transparent_background=transparent,
                       window_size=[figsize, figsize])
else:
    plotter.show()


# Plotting a dolfinx.fem.Function with vector values
# ==================================================

# In the previous sections, we have considered how to plot scalar valued
# functions. This section will show you how to plot vector valued
# functions. We start by interpolating an expression into a second order
# CG space.
def vel(x):
    vals = np.zeros((2, x.shape[1]))
    vals[0] = np.sin(x[1])
    vals[1] = 0.1 * x[0]
    return vals


mesh = create_unit_square(MPI.COMM_WORLD, 6, 6, CellType.triangle)
V = VectorFunctionSpace(mesh, ("Lagrange", 2))
uh = Function(V)
uh.interpolate(vel)

# We use the `dolfinx.plot.create_vtk_topology`
# function, as in the previous section. However, we input a set of cell
# entities, which can restrict the plotting to subsets of our mesh
num_cells = mesh.topology.index_map(mesh.topology.dim).size_local
cell_entities = np.arange(num_cells, dtype=np.int32)
topology, cell_types = dolfinx.plot.create_vtk_topology(V, cell_entities)

# As we deal with a vector function space, we need to adjust the values
# in the underlying one dimensional array in dolfinx.Function, by
# reshaping the data, and add an extra column to make it a 3D vector
geometry = V.tabulate_dof_coordinates()
num_dofs = geometry.shape[0]
values = np.zeros((num_dofs, 3), dtype=np.float64)
values[:, :mesh.geometry.dim] = uh.x.array.real.reshape(num_dofs, V.dofmap.index_map_bs)

# Create a point cloud of glyphs
function_grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
function_grid["vectors"] = values
function_grid.set_active_vectors("vectors")
glyphs = function_grid.glyph(orient="vectors", factor=0.1)

# To add streamlines in 2D, you need to supply the two points creating a
# sampling line
# NOTE: This crashes if ran in parallel, seems to be a pyvista issue
# with pointa, pointb
if MPI.COMM_WORLD.size == 0:
    streamlines = function_grid.streamlines(vectors="vectors", return_source=False,
                                            pointa=(0.5, 0.0, 0), pointb=(0.5, 1, 0))

# Create pyvista mesh from the mesh
pyvista_cells, cell_types = dolfinx.plot.create_vtk_topology(mesh, mesh.topology.dim, cell_entities)
grid = pyvista.UnstructuredGrid(pyvista_cells, cell_types, mesh.geometry.x)

# Add mesh, glyphs and streamlines to plotter
plotter = pyvista.Plotter()
plotter.add_text("Second order vector function.",
                 position="upper_edge", font_size=14, color="black")
sargs = dict(height=0.1, width=0.8, vertical=False, position_x=0.1, position_y=0, color="black")

plotter.add_mesh(grid, show_edges=True, color="black", style="wireframe")
plotter.add_mesh(glyphs)
if MPI.COMM_WORLD.size == 0:
    plotter.add_mesh(streamlines.tube(radius=0.001), show_scalar_bar=False)
plotter.view_xy()
if pyvista.OFF_SCREEN:
    plotter.screenshot(f"vectors_{MPI.COMM_WORLD.rank}.png",
                       transparent_background=transparent, window_size=[figsize, figsize])
else:
    plotter.show()

# Plotting a dolfinx.Function with vector values
# ===============================================

# In this section we illustrate how to visualize streamlines in 3D


def vel(x):
    vals = np.zeros((3, x.shape[1]))
    vals[0] = -(x[1] - 0.5)
    vals[1] = x[0] - 0.5
    vals[2] = 0
    return vals


mesh = create_unit_cube(MPI.COMM_WORLD, 4, 4, 4, CellType.hexahedron)
V = VectorFunctionSpace(mesh, ("DG", 2))
uh = Function(V)
uh.interpolate(vel)

topology, cell_types = dolfinx.plot.create_vtk_topology(V)
geometry = uh.function_space.tabulate_dof_coordinates()
num_dofs = geometry.shape[0]
values = np.zeros((num_dofs, 3), dtype=np.float64)
values[:, :mesh.geometry.dim] = uh.x.array.real.reshape(num_dofs, V.dofmap.index_map_bs)

# Create a point cloud of glyphs
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
grid["vectors"] = values
grid.set_active_vectors("vectors")
glyphs = grid.glyph(orient="vectors", factor=0.1)
streamlines = grid.streamlines(vectors="vectors", return_source=False, source_radius=1, n_points=150)

# Create Create plotter
plotter = pyvista.Plotter()
plotter.add_text("Vector function as streamlines.",
                 position="upper_edge", font_size=20, color="black")
sargs = dict(height=0.1, width=0.8, vertical=False, position_x=0.1, position_y=0, color="black")
plotter.add_mesh(grid, style="wireframe")
plotter.add_mesh(glyphs)
plotter.add_mesh(streamlines.tube(radius=0.001))
plotter.view_xy()
if pyvista.OFF_SCREEN:
    plotter.screenshot(f"streamlines_{MPI.COMM_WORLD.rank}.png",
                       transparent_background=transparent, window_size=[figsize, figsize])
else:
    plotter.show()
