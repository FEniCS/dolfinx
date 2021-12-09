# Copyright (C) 2021 Jørgen S. Dokken
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
from dolfinx.generation import UnitCubeMesh, UnitSquareMesh
from dolfinx.mesh import CellType, MeshTags, compute_midpoints

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


# Plotting a 3D dolfinx.Function with pyvista
# ===========================================

# Interpolate a simple scalar function in 3D
def int_u(x):
    return x[0] + 3 * x[1] + 5 * x[2]


mesh = UnitCubeMesh(MPI.COMM_WORLD, 4, 3, 5, cell_type=CellType.tetrahedron)
V = FunctionSpace(mesh, ("Lagrange", 1))
u = Function(V)
u.interpolate(int_u)

# Extract mesh data from DOLFINx (only plot cells owned by the
# processor) and create a pyvista UnstructuredGrid
num_cells = mesh.topology.index_map(mesh.topology.dim).size_local
cell_entities = np.arange(num_cells, dtype=np.int32)
pyvista_cells, cell_types = dolfinx.plot.create_vtk_topology(mesh, mesh.topology.dim, cell_entities)
grid = pyvista.UnstructuredGrid(pyvista_cells, cell_types, mesh.geometry.x)

# Compute the function values at the vertices, this is equivalent to a
# P1 Lagrange interpolation, and can be directly attached to the Pyvista
# mesh. Discard complex value if running DOLFINx with complex PETSc as
# backend
vertex_values = u.compute_point_values()
if np.iscomplexobj(vertex_values):
    vertex_values = vertex_values.real

# Create point cloud of vertices, and add the vertex values to the cloud
grid.point_data["u"] = vertex_values
grid.set_active_scalars("u")

# Create a pyvista plotter which is used to visualize the output
plotter = pyvista.Plotter()
plotter.add_text("Mesh and corresponding dof values",
                 position="upper_edge", font_size=14, color="black")

# Some styling arguments for the colorbar
sargs = dict(height=0.6, width=0.1, vertical=True, position_x=0.825, position_y=0.2, fmt="%1.2e",
             title_font_size=40, color="black", label_font_size=25)

# Plot the mesh (as a wireframe) with the finite element function
# visualized as the point cloud
plotter.add_mesh(grid, style="wireframe", line_width=2, color="black")

# To be able to visualize the mesh and nodes at the same time, we have
# to copy the grid
plotter.add_mesh(grid.copy(), style="points", render_points_as_spheres=True,
                 scalars=vertex_values, point_size=10)
plotter.set_position([1.5, 0.5, 4])

# Save as png if we are using a container with no rendering
if pyvista.OFF_SCREEN:
    plotter.screenshot("3D_wireframe_with_nodes.png", transparent_background=transparent,
                       window_size=[figsize, figsize])
else:
    plotter.show()

# Create a new plotter, and plot the values as a surface over the mesh
plotter = pyvista.Plotter()
plotter.add_text("Function values over the surface of a mesh",
                 position="upper_edge", font_size=14, color="black")

# Define some styling arguments for a colorbar
sargs = dict(height=0.1, width=0.8, vertical=False, position_x=0.1,
             position_y=0.05, fmt="%1.2e",
             title_font_size=40, color="black", label_font_size=25)

# Adjust camera to show the entire mesh
plotter.set_position([-2, -2, 2.1])
plotter.set_focus([1, 1, -0.01])
plotter.set_viewup([0, 0, 1])

# Add mesh with edges
plotter.add_mesh(grid, show_edges=True, scalars="u", scalar_bar_args=sargs)
if pyvista.OFF_SCREEN:
    plotter.screenshot("3D_function.png", transparent_background=transparent, window_size=[figsize, figsize])
else:
    plotter.show()

# Plotting a 2D dolfinx.Function with pyvista using warp by scalar
# ================================================================

# As in the previous section, we interpolate a function into a Lagrange
# function space

mesh = UnitSquareMesh(MPI.COMM_WORLD, 12, 12, cell_type=CellType.quadrilateral)
V = FunctionSpace(mesh, ("Lagrange", 1))
u = Function(V)
u.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(2 * x[1] * np.pi))

# As in the previous section, we extract the geometry and topology of
# the mesh, and attach values to the vertices
num_cells = mesh.topology.index_map(mesh.topology.dim).size_local
cells = np.arange(num_cells, dtype=np.int32)
pyvista_cells, cell_types = dolfinx.plot.create_vtk_topology(mesh, mesh.topology.dim, cells)
grid = pyvista.UnstructuredGrid(pyvista_cells, cell_types, mesh.geometry.x)
point_values = u.compute_point_values()
if np.iscomplexobj(point_values):
    point_values = point_values.real
grid.point_data["u"] = point_values

# We set the function "u" as the active scalar for the mesh, and warp
# the mesh in z-direction by its values
grid.set_active_scalars("u")
warped = grid.warp_by_scalar()

# Plot mesh with scalar bar
plotter = pyvista.Plotter()
plotter.add_text("Warped function", position="upper_edge", font_size=14, color="black")
sargs = dict(height=0.8, width=0.1, vertical=True, position_x=0.05,
             position_y=0.05, fmt="%1.2e",
             title_font_size=40, color="black", label_font_size=25)
plotter.set_position([-3, 2.6, 0.3])
plotter.set_focus([3, -1, -0.15])
plotter.set_viewup([0, 0, 1])
plotter.add_mesh(warped, show_edges=True, scalar_bar_args=sargs)
if pyvista.OFF_SCREEN:
    plotter.screenshot("2D_function_warp.png", transparent_background=transparent, window_size=[figsize, figsize])
else:
    plotter.show()

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

# We start by projecting a discontinuous function into a second order DG
# space Note that we use the `cell_tags` from the previous section to
# restrict the integration domain on the RHS.
dx = ufl.Measure("dx", subdomain_data=cell_tags)
V = FunctionSpace(mesh, ("DG", 2))
uh = Function(V)
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
x = ufl.SpatialCoordinate(mesh)
a = ufl.inner(u, v) * dx
L = ufl.inner(x[0], v) * dx(1) + ufl.inner(0.01, v) * dx(0)
problem = LinearProblem(a, L, u=uh, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
problem.solve()

# To get a topology that has a 1-1 correspondence with the degrees of
# freedom in the function space, we call
# `dolfinx.plot.create_vtk_topology`. We obtain the geometry for
# the dofs owned on this process by tabulation of the dof coordinates.
topology, cell_types = dolfinx.plot.create_vtk_topology(V)
num_dofs_local = uh.function_space.dofmap.index_map.size_local
geometry = uh.function_space.tabulate_dof_coordinates()[:num_dofs_local]

# We discard the complex values if using PETSc in complex mode
values = uh.vector.array.real if np.iscomplexobj(uh.vector.array) else uh.vector.array

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


mesh = UnitSquareMesh(MPI.COMM_WORLD, 6, 6, CellType.triangle)
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
num_dofs_local = uh.function_space.dofmap.index_map.size_local
geometry = uh.function_space.tabulate_dof_coordinates()[:num_dofs_local]
values = np.zeros((V.dofmap.index_map.size_local, 3), dtype=np.float64)
values[:, :mesh.geometry.dim] = uh.vector.array.real.reshape(V.dofmap.index_map.size_local, V.dofmap.index_map_bs)

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


mesh = UnitCubeMesh(MPI.COMM_WORLD, 4, 4, 4, CellType.hexahedron)
V = VectorFunctionSpace(mesh, ("DG", 2))
uh = Function(V)
uh.interpolate(vel)

num_cells = mesh.topology.index_map(mesh.topology.dim).size_local
cell_entities = np.arange(num_cells, dtype=np.int32)

topology, cell_types = dolfinx.plot.create_vtk_topology(V, cell_entities)
num_dofs_local = uh.function_space.dofmap.index_map.size_local
geometry = uh.function_space.tabulate_dof_coordinates()[:num_dofs_local]
values = np.zeros((V.dofmap.index_map.size_local, 3), dtype=np.float64)
values[:, :mesh.geometry.dim] = uh.vector.array.real.reshape(V.dofmap.index_map.size_local, V.dofmap.index_map_bs)

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
