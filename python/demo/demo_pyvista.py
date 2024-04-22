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
# # Visualization with PyVista
#
# [PyVista](https://pyvista.org/) can be used with DOLFINx for
# interactive visualisation.
#
# To start, the required modules are imported and some PyVista
# parameters set.

from mpi4py import MPI

# +
import numpy as np

import dolfinx.plot as plot
from dolfinx.fem import Function, functionspace
from dolfinx.mesh import CellType, compute_midpoints, create_unit_cube, create_unit_square, meshtags

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
# -

# ## Plotting a finite element Function using warp by scalar


def plot_scalar():
    # We start by creating a unit square mesh and interpolating a
    # function into a degree 1 Lagrange space
    msh = create_unit_square(MPI.COMM_WORLD, 12, 12, cell_type=CellType.quadrilateral)
    V = functionspace(msh, ("Lagrange", 1))
    u = Function(V, dtype=np.float64)
    u.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(2 * x[1] * np.pi))

    # To visualize the function u, we create a VTK-compatible grid to
    # values of u to
    cells, types, x = plot.vtk_mesh(V)
    grid = pyvista.UnstructuredGrid(cells, types, x)
    grid.point_data["u"] = u.x.array

    # The function "u" is set as the active scalar for the mesh, and
    # warp in z-direction is set
    grid.set_active_scalars("u")
    warped = grid.warp_by_scalar()

    # A plotting window is created with two sub-plots, one of the scalar
    # values and the other of the mesh is warped by the scalar values in
    # z-direction
    subplotter = pyvista.Plotter(shape=(1, 2))
    subplotter.subplot(0, 0)
    subplotter.add_text("Scalar contour field", font_size=14, color="black", position="upper_edge")
    subplotter.add_mesh(grid, show_edges=True, show_scalar_bar=True)
    subplotter.view_xy()

    subplotter.subplot(0, 1)
    subplotter.add_text("Warped function", position="upper_edge", font_size=14, color="black")
    sargs = dict(
        height=0.8,
        width=0.1,
        vertical=True,
        position_x=0.05,
        position_y=0.05,
        fmt="%1.2e",
        title_font_size=40,
        color="black",
        label_font_size=25,
    )
    subplotter.set_position([-3, 2.6, 0.3])
    subplotter.set_focus([3, -1, -0.15])
    subplotter.set_viewup([0, 0, 1])
    subplotter.add_mesh(warped, show_edges=True, scalar_bar_args=sargs)
    if pyvista.OFF_SCREEN:
        subplotter.screenshot(
            "2D_function_warp.png",
            transparent_background=transparent,
            window_size=[figsize, figsize],
        )
    else:
        subplotter.show()


# ## Mesh tags and using subplots


def plot_meshtags():
    # Create a mesh
    msh = create_unit_square(MPI.COMM_WORLD, 25, 25, cell_type=CellType.quadrilateral)

    # Create a geometric indicator function
    def in_circle(x):
        return np.array((x.T[0] - 0.5) ** 2 + (x.T[1] - 0.5) ** 2 < 0.2**2, dtype=np.int32)

    # Create cell tags - if midpoint is inside circle, it gets value 1,
    # otherwise 0
    num_cells = msh.topology.index_map(msh.topology.dim).size_local
    midpoints = compute_midpoints(msh, msh.topology.dim, np.arange(num_cells, dtype=np.int32))
    cell_tags = meshtags(msh, msh.topology.dim, np.arange(num_cells), in_circle(midpoints))

    # Create VTK mesh
    cells, types, x = plot.vtk_mesh(msh)
    grid = pyvista.UnstructuredGrid(cells, types, x)

    # Attach the cells tag data to the pyvita grid
    grid.cell_data["Marker"] = cell_tags.values
    grid.set_active_scalars("Marker")

    # Create a plotter with two subplots, and add mesh tag plot to the
    # first sub-window
    subplotter = pyvista.Plotter(shape=(1, 2))
    subplotter.subplot(0, 0)
    subplotter.add_text("Mesh with markers", font_size=14, color="black", position="upper_edge")
    subplotter.add_mesh(grid, show_edges=True, show_scalar_bar=False)
    subplotter.view_xy()

    # We can visualize subsets of data, by creating a smaller topology
    # (set of cells). Here we create VTK mesh data for only cells with
    # that tag '1'.
    cells, types, x = plot.vtk_mesh(msh, entities=cell_tags.find(1))

    # Add this grid to the second plotter window
    sub_grid = pyvista.UnstructuredGrid(cells, types, x)
    subplotter.subplot(0, 1)
    subplotter.add_text("Subset of mesh", font_size=14, color="black", position="upper_edge")
    subplotter.add_mesh(sub_grid, show_edges=True, edge_color="black")

    if pyvista.OFF_SCREEN:
        subplotter.screenshot(
            "2D_markers.png", transparent_background=transparent, window_size=[2 * figsize, figsize]
        )
    else:
        subplotter.show()


# ## Higher-order Functions
#
# Higher-order finite element function can also be plotted.


def plot_higher_order():
    # Create a mesh
    msh = create_unit_square(MPI.COMM_WORLD, 12, 12, cell_type=CellType.quadrilateral)

    # Define a geometric indicator function
    def in_circle(x):
        return np.array((x.T[0] - 0.5) ** 2 + (x.T[1] - 0.5) ** 2 < 0.2**2, dtype=np.int32)

    # Create mesh tags for all cells. If midpoint is inside the circle,
    # it gets value 1, otherwise 0.
    num_cells = msh.topology.index_map(msh.topology.dim).size_local
    midpoints = compute_midpoints(msh, msh.topology.dim, np.arange(num_cells, dtype=np.int32))
    cell_tags = meshtags(msh, msh.topology.dim, np.arange(num_cells), in_circle(midpoints))

    # We start by interpolating a discontinuous function (discontinuous
    # between cells with different mesh tag values) into a degree 2
    # discontinuous Lagrange space.
    V = functionspace(msh, ("Discontinuous Lagrange", 2))
    u = Function(V, dtype=msh.geometry.x.dtype)
    u.interpolate(lambda x: x[0], cell_tags.find(0))
    u.interpolate(lambda x: x[1] + 1, cell_tags.find(1))
    u.x.scatter_forward()

    # Create a topology that has a 1-1 correspondence with the
    # degrees-of-freedom in the function space V
    cells, types, x = plot.vtk_mesh(V)

    # Create a pyvista mesh and attach the values of u
    grid = pyvista.UnstructuredGrid(cells, types, x)
    grid.point_data["u"] = u.x.array
    grid.set_active_scalars("u")

    # We would also like to visualize the underlying mesh and obtain
    # that as we have done previously
    num_cells = msh.topology.index_map(msh.topology.dim).size_local
    cell_entities = np.arange(num_cells, dtype=np.int32)
    cells, types, x = plot.vtk_mesh(msh, entities=cell_entities)
    org_grid = pyvista.UnstructuredGrid(cells, types, x)

    # We visualize the data
    plotter = pyvista.Plotter()
    plotter.add_text(
        "Second-order (P2) discontinuous elements",
        position="upper_edge",
        font_size=14,
        color="black",
    )
    sargs = dict(height=0.1, width=0.8, vertical=False, position_x=0.1, position_y=0, color="black")
    plotter.add_mesh(grid, show_edges=False, scalar_bar_args=sargs, line_width=0)
    plotter.add_mesh(org_grid, color="white", style="wireframe", line_width=5)
    plotter.add_mesh(
        grid.copy(), style="points", point_size=15, render_points_as_spheres=True, line_width=0
    )
    plotter.view_xy()
    if pyvista.OFF_SCREEN:
        plotter.screenshot(
            f"DG_{MPI.COMM_WORLD.rank}.png",
            transparent_background=transparent,
            window_size=[figsize, figsize],
        )
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
    plotter.add_text(
        "Mesh and corresponding vectors", position="upper_edge", font_size=14, color="black"
    )

    # Next, we create a pyvista.UnstructuredGrid based on the mesh
    pyvista_cells, cell_types, x = plot.vtk_mesh(msh)
    grid = pyvista.UnstructuredGrid(pyvista_cells, cell_types, x)

    # Add this grid (as a wireframe) to the plotter
    plotter.add_mesh(grid, style="wireframe", line_width=2, color="black")

    # Create a function space consisting of first order Nédélec (first kind)
    # elements and interpolate a vector-valued expression
    V = functionspace(msh, ("N1curl", 2))
    u = Function(V, dtype=np.float64)
    u.interpolate(lambda x: (x[2] ** 2, np.zeros(x.shape[1]), -x[0] * x[2]))

    # Exact visualisation of the Nédélec spaces requires a Lagrange or
    # discontinuous Lagrange finite element functions. Therefore, we
    # interpolate the Nédélec function into a first-order discontinuous
    # Lagrange space.
    gdim = msh.geometry.dim
    V0 = functionspace(msh, ("Discontinuous Lagrange", 2, (gdim,)))
    u0 = Function(V0, dtype=np.float64)
    u0.interpolate(u)

    # Create a second grid, whose geometry and topology is based on the
    # output function space
    cells, cell_types, x = plot.vtk_mesh(V0)
    grid = pyvista.UnstructuredGrid(cells, cell_types, x)

    # Create point cloud of vertices, and add the vertex values to the cloud
    grid.point_data["u"] = u0.x.array.reshape(x.shape[0], V0.dofmap.index_map_bs)
    glyphs = grid.glyph(orient="u", factor=0.1)

    # We add in the glyphs corresponding to the plotter
    plotter.add_mesh(glyphs)

    # Save as png if we are using a container with no rendering
    if pyvista.OFF_SCREEN:
        plotter.screenshot(
            "3D_wireframe_with_vectors.png",
            transparent_background=transparent,
            window_size=[figsize, figsize],
        )
    else:
        plotter.show()


# ## Plotting streamlines
#
# In this section we illustrate how to visualize streamlines in 3D


def plot_streamlines():
    msh = create_unit_cube(MPI.COMM_WORLD, 4, 4, 4, CellType.hexahedron)
    gdim = msh.geometry.dim
    V = functionspace(msh, ("Discontinuous Lagrange", 2, (gdim,)))
    u = Function(V, dtype=np.float64)
    u.interpolate(lambda x: np.vstack((-(x[1] - 0.5), x[0] - 0.5, np.zeros(x.shape[1]))))

    cells, types, x = plot.vtk_mesh(V)
    num_dofs = x.shape[0]
    values = np.zeros((num_dofs, 3), dtype=np.float64)
    values[:, : msh.geometry.dim] = u.x.array.reshape(num_dofs, V.dofmap.index_map_bs)

    # Create a point cloud of glyphs
    grid = pyvista.UnstructuredGrid(cells, types, x)
    grid["vectors"] = values
    grid.set_active_vectors("vectors")
    glyphs = grid.glyph(orient="vectors", factor=0.1)
    streamlines = grid.streamlines(
        vectors="vectors", return_source=False, source_radius=1, n_points=150
    )

    # Create Create plotter
    plotter = pyvista.Plotter()
    plotter.add_text("Streamlines.", position="upper_edge", font_size=20, color="black")
    plotter.add_mesh(grid, style="wireframe")
    plotter.add_mesh(glyphs)
    plotter.add_mesh(streamlines.tube(radius=0.001))
    plotter.view_xy()
    if pyvista.OFF_SCREEN:
        plotter.screenshot(
            f"streamlines_{MPI.COMM_WORLD.rank}.png",
            transparent_background=transparent,
            window_size=[figsize, figsize],
        )
    else:
        plotter.show()


plot_scalar()
plot_meshtags()
plot_higher_order()
plot_nedelec()
plot_streamlines()
