# Copyright (C) 2021 Jørgen S. Dokken
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# =========================================
# Using pyvista for visualization
# =========================================


import dolfinx
import dolfinx.io
import dolfinx.plotting
import numpy as np
import ufl
from mpi4py import MPI

import pyvista


def activate_virtual_framebuffer():
    '''
    See: https://github.com/pyvista/pyvista/issues/155

    Activates a virtual (headless) framebuffer for rendering 3D
    scenes via VTK.

    Most critically, this function is useful when this code is being run
    in a Dockerized notebook, or over a server without X forwarding.

    * Requires the following packages:
      * `sudo apt-get install libgl1-mesa-dev xvfb`
    '''

    import os
    import subprocess
    pyvista.OFFSCREEN = True
    os.environ['DISPLAY'] = ':99.0'

    commands = ['Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &',
                'sleep 3',
                'exec "$@"']

    for command in commands:
        subprocess.call(command, shell=True)


activate_virtual_framebuffer()
off_screen = True
transparent = False
figsize = 800


# Plotting a 3D dolfinx.Function with pyvista
# ===========================================


def int_u(x):
    return x[0] + 3 * x[1] + 5 * x[2]


mesh = dolfinx.UnitCubeMesh(MPI.COMM_WORLD, 3, 5, 7, cell_type=dolfinx.cpp.mesh.CellType.hexahedron)


V = dolfinx.FunctionSpace(mesh, ("CG", 1))
u = dolfinx.Function(V)
u.interpolate(int_u)
vertex_values = u.compute_point_values()

pyvista.rcParams["background"] = [0.5, 0.5, 0.5]

# Extract mesh data from dolfin-X and create a pyvista UnstructuredGrid
num_cells = mesh.topology.index_map(mesh.topology.dim).size_local
cell_entities = np.arange(num_cells, dtype=np.int32)
pyvista_cells, cell_types = dolfinx.plotting.pyvista_topology_from_mesh(mesh, mesh.topology.dim, cell_entities)
grid = pyvista.UnstructuredGrid(pyvista_cells, cell_types, mesh.geometry.x)

# Create plotter for mesh and point cloud
plotter = pyvista.Plotter(off_screen=off_screen)

# Create point cloud of vertices, and add the vertex values to the cloud
vertices = pyvista.PolyData(grid.points)
vertices.point_arrays["u"] = vertex_values
vertices.set_active_scalars("u")

# Plot the mesh with the finite element function visualized as the point cloud
sargs = dict(height=0.6, width=0.1, vertical=True, position_x=0.825,
             position_y=0.2, fmt="%1.2e",
             title_font_size=40, color="black", label_font_size=25)

plotter.add_text("Visualization of a mesh and \n corresponding dof values",
                 position="upper_edge", font_size=20, color="black")
plotter.add_mesh(grid, style="wireframe", line_width=2,
                 render_points_as_spheres=True, color="black")
plotter.add_mesh(vertices, point_size=10.0, render_points_as_spheres=True, scalar_bar_args=sargs)
plotter.set_position([1.5, 0.5, 4])

# Save as png if we are using a container with no rendering
if off_screen:
    plotter.screenshot("3D_wireframe_with_nodes.png", transparent_background=transparent,
                       window_size=[figsize, figsize])
else:
    plotter.show()

plotter = pyvista.Plotter(off_screen=off_screen)
# Add values from the vertices of the mesh to the grid
sargs = dict(height=0.1, width=0.8, vertical=False, position_x=0.1,
             position_y=0.05, fmt="%1.2e",
             title_font_size=40, color="black", label_font_size=25)
grid.point_arrays["u"] = vertex_values
# Adjust camera to show the entire mesh
plotter.set_position([-2, -2, 2.1])
plotter.set_focus([1, 1, -0.01])
plotter.set_viewup([0, 0, 1])
plotter.add_text("Visualization of function values\n over the surface of a mesh",
                 position="upper_edge", font_size=20, color="black")
plotter.add_mesh(grid, show_edges=True, scalars="u", scalar_bar_args=sargs)

if off_screen:
    plotter.screenshot("3D_function.png", transparent_background=transparent, window_size=[figsize, figsize])
else:
    plotter.show()

# Plotting a 2D dolfinx.Function with pyvista using warp by scalar
# ================================================================


def int_2D(x):
    return np.sin(np.pi * x[0]) * np.sin(2 * x[1] * np.pi)


# Create mesh and function
mesh = dolfinx.UnitSquareMesh(MPI.COMM_WORLD, 12, 12, cell_type=dolfinx.cpp.mesh.CellType.quadrilateral)
V = dolfinx.FunctionSpace(mesh, ("CG", 1))
u = dolfinx.Function(V)
u.interpolate(int_2D)

# Create pyvista mesh and warp grid by scalar u
num_cells = mesh.topology.index_map(mesh.topology.dim).size_local
cells = np.arange(num_cells, dtype=np.int32)
pyvista_cells, cell_types = dolfinx.plotting.pyvista_topology_from_mesh(mesh, mesh.topology.dim, cells)
grid = pyvista.UnstructuredGrid(pyvista_cells, cell_types, mesh.geometry.x)
grid.point_arrays["u"] = u.compute_point_values()
grid.set_active_scalars("u")
warped = grid.warp_by_scalar()

# Plot warped mesh
sargs = dict(height=0.8, width=0.1, vertical=True, position_x=0.05,
             position_y=0.05, fmt="%1.2e",
             title_font_size=40, color="black", label_font_size=25)
plotter = pyvista.Plotter(off_screen=off_screen)
plotter.add_text("Visualization of warped function",
                 position="upper_edge", font_size=20, color="black")
plotter.add_mesh(warped, show_edges=True, scalar_bar_args=sargs)
plotter.set_position([-3, 2.6, 0.3])
plotter.set_focus([3, -1, -0.15])
plotter.set_viewup([0, 0, 1])
if off_screen:
    plotter.screenshot("2D_function_warp.png", transparent_background=transparent, window_size=[figsize, figsize])
else:
    plotter.show()


plotter = pyvista.Plotter(off_screen=off_screen)

# Plotting a 2D MeshTags and using subplots
# =========================================


def left(x):
    # Mark sphere with radius < sqrt(2)
    return np.array((x.T[0] - 0.5)**2 + (x.T[1] - 0.5)**2 < 0.2**2, dtype=np.int32)


# Create a cell-tag for all cells, with either value 0 or 1
num_cells = mesh.topology.index_map(mesh.topology.dim).size_local
midpoints = dolfinx.cpp.mesh.midpoints(mesh, mesh.topology.dim, list(np.arange(num_cells, dtype=np.int32)))
cell_tags = dolfinx.MeshTags(mesh, mesh.topology.dim, np.arange(num_cells), left(midpoints))

# Create 2D plot of cell markers
grid.cell_arrays["Marker"] = cell_tags.values
grid.set_active_scalars("Marker")
subplotter = pyvista.Plotter(off_screen=off_screen, shape=(1, 2))
subplotter.subplot(0, 0)
subplotter.add_text("Mesh with markers", font_size=24, color="black", position="upper_edge")
subplotter.add_mesh(grid, show_edges=True, show_scalar_bar=False)
subplotter.view_xy()

pyvista_cells, cell_types = dolfinx.plotting.pyvista_topology_from_mesh(
    mesh, mesh.topology.dim, cell_tags.indices[cell_tags.values == 1])

# Plot only a subset of a mesh with a given cell marker
sub_grid = pyvista.UnstructuredGrid(pyvista_cells, cell_types, mesh.geometry.x)
subplotter.subplot(0, 1)
subplotter.add_text("Subset of mesh", font_size=24, color="black", position="upper_edge")
subplotter.add_mesh(sub_grid, show_edges=True, edge_color="black")

if off_screen:
    subplotter.screenshot("2D_markers.png", transparent_background=transparent,
                          window_size=[2 * figsize, figsize])
else:
    subplotter.show()


def left(x):
    # Mark sphere with radius < sqrt(2)
    return np.array((x.T[0] - 0.5)**2 + (x.T[1] - 0.5)**2 < 0.2**2, dtype=np.int32)


# Create a cell-tag for all cells, with either value 0 or 1
num_cells = mesh.topology.index_map(mesh.topology.dim).size_local
midpoints = dolfinx.cpp.mesh.midpoints(mesh, mesh.topology.dim, list(np.arange(num_cells, dtype=np.int32)))
cell_tags = dolfinx.MeshTags(mesh, mesh.topology.dim, np.arange(num_cells), left(midpoints))


# Project a DG function
dx = ufl.Measure("dx", subdomain_data=cell_tags)
V = dolfinx.FunctionSpace(mesh, ("DG", 2))
uh = dolfinx.Function(V)
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
x = ufl.SpatialCoordinate(mesh)
a = ufl.inner(u, v) * dx
L = ufl.inner(x[0], v) * dx(1) + ufl.inner(0.01, v) * dx(0)
problem = dolfinx.fem.LinearProblem(a, L, u=uh, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
problem.solve()


topology, cell_types = dolfinx.plotting.pyvista_topology_from_function_space(uh)
geometry = uh.function_space.tabulate_dof_coordinates()
values = uh.vector.array

# Plot only a subset of a mesh with a given cell marker
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
grid.point_arrays["DG"] = values
grid.set_active_scalars("DG")

vertices = pyvista.PolyData(grid.points)
vertices.point_arrays["DG"] = values
vertices.set_active_scalars("DG")

num_cells = mesh.topology.index_map(mesh.topology.dim).size_local
cell_entities = np.arange(num_cells, dtype=np.int32)
pyvista_cells, cell_types = dolfinx.plotting.pyvista_topology_from_mesh(mesh, mesh.topology.dim, cell_entities)
org_grid = pyvista.UnstructuredGrid(pyvista_cells, cell_types, mesh.geometry.x)
plotter = pyvista.Plotter(off_screen=off_screen)
plotter.add_text("Visualization of second order \nDiscontinous Galerkin elements",
                 position="upper_edge", font_size=25, color="black")
sargs = dict(height=0.1, width=0.8, vertical=False, position_x=0.1, position_y=0, color="black")
plotter.add_mesh(grid, show_edges=False, scalar_bar_args=sargs)
plotter.add_mesh(org_grid, show_edges=True, color="black", style="wireframe")
plotter.add_mesh(vertices, point_size=15, render_points_as_spheres=True)
plotter.view_xy()
if off_screen:
    plotter.screenshot("DG.png", transparent_background=transparent, window_size=[1500, 1700])
else:
    plotter.show()
