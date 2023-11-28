# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.6
# ---

# # Interpolation and IO
#
# Copyright (C) 2022 Garth N. Wells
#
# This demo ({download}`demo_interpolation-io.py`) shows the
# interpolation of functions into vector-element $H(\mathrm{curl})$
# finite element spaces, and the interpolation of these special finite
# elements in discontinuous Lagrange spaces for artifact-free
# visualisation.

from mpi4py import MPI

# +
import numpy as np

from dolfinx import default_scalar_type, plot
from dolfinx.fem import Function, functionspace
from dolfinx.mesh import CellType, create_rectangle, locate_entities

# -

# Create a mesh. For later in the demo we need to ensure that a boundary
# between cells is located at $x_0=0.5$.

msh = create_rectangle(MPI.COMM_WORLD, ((0.0, 0.0), (1.0, 1.0)), (16, 16), CellType.triangle)

# Create a Nédélec function space and finite element Function

V = functionspace(msh, ("Nedelec 1st kind H(curl)", 1))
u = Function(V, dtype=default_scalar_type)

# Find cells with *all* vertices (0) $x_0 <= 0.5$ or (1) $x_0 >= 0.5$:

tdim = msh.topology.dim
cells0 = locate_entities(msh, tdim, lambda x: x[0] <= 0.5)
cells1 = locate_entities(msh, tdim, lambda x: x[0] >= 0.5)

# Interpolate in the Nédélec/H(curl) space a vector-valued expression
# `f`, where $f \cdot n$ is discontinuous at $x_0 = 0.5$ and  $f \cdot
# e$ is continuous.

u.interpolate(lambda x: np.vstack((x[0], x[1])), cells0)
u.interpolate(lambda x: np.vstack((x[0] + 1, x[1])), cells1)

# Create a vector-valued discontinuous Lagrange space and function, and
# interpolate the $H({\rm curl})$ function `u`

gdim = msh.geometry.dim
V0 = functionspace(msh, ("Discontinuous Lagrange", 1, (gdim,)))
u0 = Function(V0, dtype=default_scalar_type)
u0.interpolate(u)

# We save the interpolated function `u0` in VTX format. When visualising
# the field, at $x_0 = 0.5$ the $x_0$-component should appear
# discontinuous and the $x_1$-component should appear continuous.

try:
    from dolfinx.io import VTXWriter

    with VTXWriter(msh.comm, "output_nedelec.bp", u0, "bp4") as f:
        f.write(0.0)
except ImportError:
    print("ADIOS2 required for VTX output")


# Plot the functions

try:
    import pyvista

    cells, types, x = plot.vtk_mesh(V0)
    grid = pyvista.UnstructuredGrid(cells, types, x)
    values = np.zeros((x.shape[0], 3), dtype=np.float64)
    values[:, : msh.topology.dim] = u0.x.array.reshape(x.shape[0], msh.topology.dim).real
    grid.point_data["u"] = values

    pl = pyvista.Plotter(shape=(2, 2))

    pl.subplot(0, 0)
    pl.add_text("magnitude", font_size=12, color="black", position="upper_edge")
    pl.add_mesh(grid.copy(), show_edges=True)

    pl.subplot(0, 1)
    glyphs = grid.glyph(orient="u", factor=0.08)
    pl.add_text("vector glyphs", font_size=12, color="black", position="upper_edge")
    pl.add_mesh(glyphs, show_scalar_bar=False)
    pl.add_mesh(grid.copy(), style="wireframe", line_width=2, color="black")

    pl.subplot(1, 0)
    pl.add_text("x-component", font_size=12, color="black", position="upper_edge")
    pl.add_mesh(grid.copy(), component=0, show_edges=True)

    pl.subplot(1, 1)
    pl.add_text("y-component", font_size=12, color="black", position="upper_edge")
    pl.add_mesh(grid.copy(), component=1, show_edges=True)

    pl.view_xy()
    pl.link_views()

    # If pyvista environment variable is set to off-screen (static)
    # plotting save png
    if pyvista.OFF_SCREEN:
        pyvista.start_xvfb(wait=0.1)
        pl.screenshot("uh.png")
    else:
        pl.show()
except ModuleNotFoundError:
    print("pyvista is required to visualise the solution")
