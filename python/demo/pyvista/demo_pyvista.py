# Copyright (C) 2020 Jørgen S. Dokken
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
import numba
import numpy as np
import ufl
from mpi4py import MPI
from numba.typed import List
from petsc4py import PETSc

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
pyvista_mesh_data = dolfinx.plotting.mesh_to_pyvista(mesh, mesh.topology.dim)
grid = pyvista.UnstructuredGrid(*pyvista_mesh_data)

# Create plotter for mesh and point cloud
plotter = pyvista.Plotter(off_screen=True)

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

# Use this if you use docker
plotter.screenshot("3D_wireframe_with_nodes.png", transparent_background=True, window_size=[900, 900])
# Otherwise
# plotter.show()

plotter.clear()
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
plotter.screenshot("3D_function.png", transparent_background=True, window_size=[900, 900])
plotter.clear()

# Plotting a 2D dolfinx.Function with pyvista using warp by scalar
# ================================================================


def int_2D(x):
    return np.sin(np.pi * x[0]) * np.sin(2 * x[1] * np.pi)


# Create mesh and function
mesh = dolfinx.UnitSquareMesh(MPI.COMM_WORLD, 27, 17, cell_type=dolfinx.cpp.mesh.CellType.triangle)
V = dolfinx.FunctionSpace(mesh, ("CG", 1))
u = dolfinx.Function(V)
u.interpolate(int_2D)

# Create pyvista mesh and warp grid by scalar u
grid = pyvista.UnstructuredGrid(*dolfinx.plotting.mesh_to_pyvista(mesh, mesh.topology.dim))
grid.point_arrays["u"] = u.compute_point_values()
grid.set_active_scalars("u")
warped = grid.warp_by_scalar()

# Plot warped mesh
sargs = dict(height=0.8, width=0.1, vertical=True, position_x=0.05,
             position_y=0.05, fmt="%1.2e",
             title_font_size=40, color="black", label_font_size=25)
plotter.add_text("Visualization of warped function",
                 position="upper_edge", font_size=20, color="black")
plotter.add_mesh(warped, show_edges=True, scalar_bar_args=sargs)
plotter.set_position([-3, 2.6, 0.3])
plotter.set_focus([3, -1, -0.15])
plotter.set_viewup([0, 0, 1])
plotter.screenshot("2D_function_warp.png", transparent_background=True, window_size=[900, 900])
plotter.clear()


# Plotting a 2D MeshTags and using subplots
# =========================================

def left(x):
    # Mark sphere with radius < sqrt(2)
    return np.array((x.T[0] - 0.5)**2 + (x.T[1] - 0.5)**2 < 0.2**2, dtype=np.int32)


# Create a cell-tag for all cells, with either value 0 or 1
num_cells = mesh.topology.index_map(mesh.topology.dim).size_local
midpoints = dolfinx.cpp.mesh.midpoints(mesh, mesh.topology.dim, np.arange(num_cells))
cell_tags = dolfinx.MeshTags(mesh, mesh.topology.dim, np.arange(num_cells), left(midpoints))

# Create 2D plot of cell markers
grid.cell_arrays["Marker"] = cell_tags.values
grid.set_active_scalars("Marker")
subplotter = pyvista.Plotter(off_screen=True, shape=(1, 2))
subplotter.subplot(0, 0)
subplotter.add_text("Mesh with markers", font_size=24, color="black", position="upper_edge")
subplotter.add_mesh(grid, show_edges=True, show_scalar_bar=False)
subplotter.view_xy()


@numba.njit
def isin(value, array):
    """
    Numba helper to check if a value is in an array
    """
    is_in = False
    for item in array:
        if value == item:
            is_in = True
            break
    return is_in


@numba.njit
def extract_sub_topology(cells, cell_types, indices, new_cells, new_cell_types):
    """
    Extract the sub topology required for pyvista, given the indices of the cells
    that should be extracted.
    """
    i, cell_counter, marked_counter = 0, 0, 0
    while i < len(cells):
        num_nodes = cells[i]
        if isin(cell_counter, indices):
            new_cells.append(num_nodes)
            for node in cells[i + 1: i + num_nodes + 1]:
                new_cells.append(node)
            new_cell_types[marked_counter] = cell_types[cell_counter]
            marked_counter += 1
        i += num_nodes + 1
        cell_counter += 1


# Create topology for cells marked with 1
cell_subset = List.empty_list(numba.types.int32)
cell_indices = cell_tags.indices[cell_tags.values == 1]
cell_types_subset = np.zeros(len(cell_indices))
extract_sub_topology(grid.cells, grid.celltypes, cell_indices, cell_subset, cell_types_subset)

# Plot only a subset of a mesh with a given cell marker
sub_grid = pyvista.UnstructuredGrid(np.array(cell_subset, dtype=np.int32), cell_types_subset, grid.points)
subplotter.subplot(0, 1)
subplotter.add_text("Subset of mesh", font_size=24, color="black", position="upper_edge")
subplotter.add_mesh(sub_grid, show_edges=True, edge_color="black")
subplotter.screenshot("2D_markers.png", transparent_background=True, window_size=[1500, 750])
subplotter.clear()


def left(x):
    # Mark sphere with radius < sqrt(2)
    return np.array((x.T[0] - 0.5)**2 + (x.T[1] - 0.5)**2 < 0.2**2, dtype=np.int32)


# Create a cell-tag for all cells, with either value 0 or 1
num_cells = mesh.topology.index_map(mesh.topology.dim).size_local
midpoints = dolfinx.cpp.mesh.midpoints(mesh, mesh.topology.dim, np.arange(num_cells))
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
u = dolfinx.Function(V)
b = dolfinx.fem.assemble_vector(L)
bcs = []
dolfinx.fem.assemble.apply_lifting(b, [a], [bcs])
b.ghostUpdate(addv=PETSc.InsertMode.ADD,
              mode=PETSc.ScatterMode.REVERSE)
dolfinx.fem.assemble.set_bc(b, bcs)
A = dolfinx.fem.assemble_matrix(a, bcs=bcs)
A.assemble()
solver_proj = PETSc.KSP().create(MPI.COMM_WORLD)
solver_proj.setType("preonly")
solver_proj.setTolerances(rtol=1.0e-14)
solver_proj.getPC().setType("lu")
solver_proj.getPC().setFactorSolverType("mumps")
solver_proj.setOperators(A)
solver_proj.solve(b, uh.vector)
uh.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                      mode=PETSc.ScatterMode.FORWARD)


geometry, topology, cell_types = dolfinx.plotting.create_pyvista_mesh_from_function_space(uh)
values = uh.vector.array

# Plot only a subset of a mesh with a given cell marker
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
grid.point_arrays["DG"] = values
grid.set_active_scalars("DG")

vertices = pyvista.PolyData(grid.points)
vertices.point_arrays["DG"] = values
vertices.set_active_scalars("DG")

org_grid = pyvista.UnstructuredGrid(*dolfinx.plotting.mesh_to_pyvista(mesh, mesh.topology.dim))
plotter = pyvista.Plotter(off_screen=True)
plotter.add_text("Visualization of second order \nDiscontinous Galerkin elements",
                 position="upper_edge", font_size=25, color="black")
sargs = dict(height=0.1, width=0.8, vertical=False, position_x=0.1, position_y=0, color="black")
plotter.add_mesh(grid, show_edges=False, scalar_bar_args=sargs)
plotter.add_mesh(org_grid, show_edges=True, color="black", style="wireframe")
plotter.add_mesh(vertices, point_size=15, render_points_as_spheres=True)
plotter.screenshot("DG.png", transparent_background=True, window_size=[1500, 1700])
