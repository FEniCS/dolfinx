# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.6
# ---

# Copyright (C) 2021-2022 Jørgen S. Dokken and Garth N. Wells
#
# This file is part of DOLFINx (<https://www.fenicsproject.org>)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# # Visualization with pyvista

# +
import numpy as np

import dolfinx.plot as plot
from dolfinx.fem import Function, FunctionSpace, VectorFunctionSpace
from dolfinx.mesh import (CellType, compute_midpoints, create_unit_cube,
                          create_unit_square, meshtags)

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
# -

# ## Plotting a Function using warp by scalar


def plot_scalar():

    # We start by creating a unit square mesh and interpolating a function
    # into a first order Lagrange space
    msh = create_unit_square(MPI.COMM_WORLD, 12, 12, cell_type=CellType.quadrilateral)
    V = FunctionSpace(msh, ("Lagrange", 1))
    u = Function(V, dtype=np.float64)
    u.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(2 * x[1] * np.pi))

    # As we want to visualize the function u, we have to create a grid to
    # attached the DoF values to We do this by creating a topology and
    # geometry based on the function space V
    cells, types, x = plot.create_vtk_mesh(V)
    grid = pyvista.UnstructuredGrid(cells, types, x)
    grid.point_data["u"] = u.x.array

    # We set the function "u" as the active scalar for the mesh, and warp
    # the mesh in z-direction by its values
    grid.set_active_scalars("u")
    warped = grid.warp_by_scalar()

    # We create a plotting window consisting of to plots, one of the scalar
    # values, and one where the mesh is warped by these values
    subplotter = pyvista.Plotter(shape=(1, 2))
    subplotter.subplot(0, 0)
    subplotter.add_text("Scalar contour field", font_size=14, color="black", position="upper_edge")
    subplotter.add_mesh(grid, show_edges=True, show_scalar_bar=True)
    subplotter.view_xy()

    subplotter.subplot(0, 1)
    subplotter.add_text("Warped function", position="upper_edge", font_size=14, color="black")
    sargs = dict(height=0.8, width=0.1, vertical=True, position_x=0.05,
                 position_y=0.05, fmt="%1.2e", title_font_size=40, color="black", label_font_size=25)
    subplotter.set_position([-3, 2.6, 0.3])
    subplotter.set_focus([3, -1, -0.15])
    subplotter.set_viewup([0, 0, 1])
    subplotter.add_mesh(warped, show_edges=True, scalar_bar_args=sargs)
    if pyvista.OFF_SCREEN:
        subplotter.screenshot("2D_function_warp.png", transparent_background=transparent,
                              window_size=[figsize, figsize])
    else:
        subplotter.show()


# ## Mesh tags and using subplots

def plot_meshtags():

    msh = create_unit_square(MPI.COMM_WORLD, 25, 25, cell_type=CellType.quadrilateral)

    # We continue using the mesh from the previous section, and find all
    # cells satisfying the condition below

    def in_circle(x):
        """True for points inside circle with radius 2"""
        return np.array((x.T[0] - 0.5)**2 + (x.T[1] - 0.5)**2 < 0.2**2, dtype=np.int32)

    # Create mesh tags for all cells. If midpoint is inside the
    # circle, it gets value 1, otherwise 0.
    num_cells = msh.topology.index_map(msh.topology.dim).size_local
    midpoints = compute_midpoints(msh, msh.topology.dim, list(np.arange(num_cells, dtype=np.int32)))
    cell_tags = meshtags(msh, msh.topology.dim, np.arange(num_cells), in_circle(midpoints))

    cells, types, x = plot.create_vtk_mesh(msh)
    grid = pyvista.UnstructuredGrid(cells, types, x)

    # As the mesh tags contain a value for every cell in the
    # geometry, we can attach it directly to the grid
    grid.cell_data["Marker"] = cell_tags.values
    grid.set_active_scalars("Marker")

    # We create a plotter consisting of two windows, and add a plot of the
    # mesh tags to the first window.
    subplotter = pyvista.Plotter(shape=(1, 2))
    subplotter.subplot(0, 0)
    subplotter.add_text("Mesh with markers", font_size=14, color="black", position="upper_edge")
    subplotter.add_mesh(grid, show_edges=True, show_scalar_bar=False)
    subplotter.view_xy()

    # We can also visualize subsets of data, by creating a smaller topology,
    # only consisting of those entities that has value one in the
    # mesh tags
    cells, types, x = plot.create_vtk_mesh(msh, entities=cell_tags.find(1))

    # We add this grid to the second plotter
    sub_grid = pyvista.UnstructuredGrid(cells, types, x)
    subplotter.subplot(0, 1)
    subplotter.add_text("Subset of mesh", font_size=14, color="black", position="upper_edge")
    subplotter.add_mesh(sub_grid, show_edges=True, edge_color="black")

    if pyvista.OFF_SCREEN:
        subplotter.screenshot("2D_markers.png", transparent_background=transparent,
                              window_size=[2 * figsize, figsize])
    else:
        subplotter.show()


# ## Higher-order Functions
#
# In the previous sections we have considered degree 1 Lagrange spaces.
# We can also plot higher degree functions.

def plot_higher_order():

    msh = create_unit_square(MPI.COMM_WORLD, 12, 12, cell_type=CellType.quadrilateral)

    # We continue using the mesh from the previous section, and find all
    # cells satisfying the condition below

    def in_circle(x):
        """Mark sphere with radius < sqrt(2)"""
        return np.array((x.T[0] - 0.5)**2 + (x.T[1] - 0.5)**2 < 0.2**2, dtype=np.int32)

    # Create mesh tags for all cells. If midpoint is inside the
    # circle, it gets value 1, otherwise 0.
    num_cells = msh.topology.index_map(msh.topology.dim).size_local
    midpoints = compute_midpoints(msh, msh.topology.dim, list(np.arange(num_cells, dtype=np.int32)))
    cell_tags = meshtags(msh, msh.topology.dim, np.arange(num_cells), in_circle(midpoints))

    # We start by interpolating a discontinuous function into a second order
    # discontinuous Lagrange space. Note that we use the `cell_tags` from
    # the previous section to get the cells for each of the regions
    V = FunctionSpace(msh, ("Discontinuous Lagrange", 2))
    u = Function(V, dtype=np.float64)
    u.interpolate(lambda x: x[0], cell_tags.find(0))
    u.interpolate(lambda x: x[1] + 1, cell_tags.find(1))
    u.x.scatter_forward()

    # To get a topology that has a 1-1 correspondence with the
    # degrees-of-freedom in the function space, we call
    # `dolfinx.plot.create_vtk_mesh`.
    cells, types, x = plot.create_vtk_mesh(V)

    # Create a pyvista mesh from the topology and geometry, and attach
    # the coefficients of the degrees of freedom
    grid = pyvista.UnstructuredGrid(cells, types, x)
    grid.point_data["u"] = u.x.array
    grid.set_active_scalars("u")

    # We would also like to visualize the underlying mesh and obtain
    # that as we have done previously
    num_cells = msh.topology.index_map(msh.topology.dim).size_local
    cell_entities = np.arange(num_cells, dtype=np.int32)
    cells, types, x = plot.create_vtk_mesh(msh, entities=cell_entities)
    org_grid = pyvista.UnstructuredGrid(cells, types, x)

    # We visualize the data
    plotter = pyvista.Plotter()
    plotter.add_text("Second-order (P2) discontinuous elements",
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


# ## Vector-element functions
#
# In this section we will consider how to plot vector-element functions,
# e.g. Raviart-Thomas or Nédélec elements.

def plot_nedelec():

    msh = create_unit_cube(MPI.COMM_WORLD, 4, 3, 5, cell_type=CellType.tetrahedron)

    # We create a pyvista plotter
    plotter = pyvista.Plotter()
    plotter.add_text("Mesh and corresponding vectors",
                     position="upper_edge", font_size=14, color="black")

    # Next, we create a pyvista.UnstructuredGrid based on the mesh
    pyvista_cells, cell_types, x = plot.create_vtk_mesh(msh)
    grid = pyvista.UnstructuredGrid(pyvista_cells, cell_types, x)

    # Add this grid (as a wireframe) to the plotter
    plotter.add_mesh(grid, style="wireframe", line_width=2, color="black")

    # Create a function space consisting of first order Nédélec (first kind)
    # elements and interpolate a vector-valued expression
    V = FunctionSpace(msh, ("N1curl", 2))
    u = Function(V, dtype=np.float64)
    u.interpolate(lambda x: (x[2]**2, np.zeros(x.shape[1]), -x[0] * x[2]))

    # Exact visualisation of the Nédélec spaces requires a Lagrange or
    # discontinuous Lagrange finite element functions. Therefore, we
    # interpolate the Nédélec function into a first-order discontinuous
    # Lagrange space.
    V0 = VectorFunctionSpace(msh, ("Discontinuous Lagrange", 2))
    u0 = Function(V0, dtype=np.float64)
    u0.interpolate(u)

    # Create a second grid, whose geometry and topology is based on the
    # output function space
    cells, cell_types, x = plot.create_vtk_mesh(V0)
    grid = pyvista.UnstructuredGrid(cells, cell_types, x)

    # Create point cloud of vertices, and add the vertex values to the cloud
    grid.point_data["u"] = u0.x.array.reshape(x.shape[0], V0.dofmap.index_map_bs)
    glyphs = grid.glyph(orient="u", factor=0.1)

    # We add in the glyphs corresponding to the plotter
    plotter.add_mesh(glyphs)

    # Save as png if we are using a container with no rendering
    if pyvista.OFF_SCREEN:
        plotter.screenshot("3D_wireframe_with_vectors.png", transparent_background=transparent,
                           window_size=[figsize, figsize])
    else:
        plotter.show()

# ## Plotting streamlines
#
# In this section we illustrate how to visualize streamlines in 3D


def plot_streamlines():

    msh = create_unit_cube(MPI.COMM_WORLD, 4, 4, 4, CellType.hexahedron)
    V = VectorFunctionSpace(msh, ("Discontinuous Lagrange", 2))
    u = Function(V, dtype=np.float64)
    u.interpolate(lambda x: np.vstack((-(x[1] - 0.5), x[0] - 0.5, np.zeros(x.shape[1]))))

    cells, types, x = plot.create_vtk_mesh(V)
    num_dofs = x.shape[0]
    values = np.zeros((num_dofs, 3), dtype=np.float64)
    values[:, :msh.geometry.dim] = u.x.array.reshape(num_dofs, V.dofmap.index_map_bs)

    # Create a point cloud of glyphs
    grid = pyvista.UnstructuredGrid(cells, types, x)
    grid["vectors"] = values
    grid.set_active_vectors("vectors")
    glyphs = grid.glyph(orient="vectors", factor=0.1)
    streamlines = grid.streamlines(vectors="vectors", return_source=False, source_radius=1, n_points=150)

    # Create Create plotter
    plotter = pyvista.Plotter()
    plotter.add_text("Streamlines.", position="upper_edge", font_size=20, color="black")
    plotter.add_mesh(grid, style="wireframe")
    plotter.add_mesh(glyphs)
    plotter.add_mesh(streamlines.tube(radius=0.001))
    plotter.view_xy()
    if pyvista.OFF_SCREEN:
        plotter.screenshot(f"streamlines_{MPI.COMM_WORLD.rank}.png",
                           transparent_background=transparent, window_size=[figsize, figsize])
    else:
        plotter.show()


plot_scalar()
plot_meshtags()
plot_higher_order()
plot_nedelec()
plot_streamlines()
