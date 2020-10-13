# Copyright (C) 2020 Jørgen S. Dokken
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# =========================================
# Using pyvista for visualization
# =========================================


from petsc4py import PETSc
import ufl
from numba.typed import List
import numba
from IPython import embed
import dolfinx
import dolfinx.io
from mpi4py import MPI
import pyvista
import numpy as np


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

    import subprocess
    import os
    pyvista.OFFSCREEN = True
    os.environ['DISPLAY'] = ':99.0'

    commands = ['Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &',
                'sleep 3',
                'exec "$@"']

    for command in commands:
        subprocess.call(command, shell=True)


activate_virtual_framebuffer()


# # Plotting a 3D dolfinx.Function with pyvista
# # ===========================================

# def int_u(x):
#     return x[0] + 2 * x[1] + x[2]


# mesh = dolfinx.UnitCubeMesh(MPI.COMM_WORLD, 3, 5, 7, cell_type=dolfinx.cpp.mesh.CellType.hexahedron)


# V = dolfinx.FunctionSpace(mesh, ("CG", 1))
# u = dolfinx.Function(V)
# u.interpolate(int_u)
# vertex_values = u.compute_point_values()

# pyvista.rcParams["background"] = [0.5, 0.5, 0.5]

# # Extract mesh data from dolfin-X and create a pyvista UnstructuredGrid
# pyvista_mesh_data = dolfinx.io.mesh_to_pyvista(mesh, mesh.topology.dim)
# grid = pyvista.UnstructuredGrid(*pyvista_mesh_data)
# # Add values from the vertices of the mesh to the grid
# grid.point_arrays["u"] = vertex_values

# # Create plotter for mesh and point cloud
# plotter = pyvista.Plotter(off_screen=True)

# # Create point cloud of vertices, and add the vertex values to the cloud
# vertices = pyvista.PolyData(grid.points)
# vertices.point_arrays["u"] = vertex_values
# vertices.set_active_scalars("u")

# # Plot the mesh with the finite element function visualized as the point cloud
# plotter.add_mesh(grid, color="orange",
#                  render_points_as_spheres=True, show_edges=True, opacity=0.33)
# plotter.add_mesh(vertices, point_size=10.0, render_points_as_spheres=True)
# plotter.show(screenshot="3D_point_plot.png")

# # Create plotter for mesh and with function values rendered on cells
# # with adjusted camera angle
# plotter2 = pyvista.Plotter(off_screen=True)
# plotter2.add_mesh(grid, color="orange", show_edges=True, scalars="u")
# plotter2.show(screenshot="3D_plot.png", cpos=[-0.1, 0.2, 0.15])

# # Plotting a 2D dolfinx.Function with pyvista using warp by scalar
# # ================================================================


# def int_2D(x):
#     return np.sin(np.pi * x[0]) * np.sin(2 * x[1] * np.pi)


# mesh2D = dolfinx.UnitSquareMesh(MPI.COMM_WORLD, 12, 17, cell_type=dolfinx.cpp.mesh.CellType.triangle)

# grid2D = pyvista.UnstructuredGrid(*dolfinx.io.mesh_to_pyvista(mesh2D, mesh2D.topology.dim))

# V2 = dolfinx.FunctionSpace(mesh2D, ("CG", 1))
# u2 = dolfinx.Function(V2)
# u2.interpolate(int_2D)


# grid2D.point_arrays["u"] = u2.compute_point_values()
# grid2D.set_active_scalars("u")
# warped = grid2D.warp_by_scalar()
# warped.plot(show_edges=True, off_screen=True, screenshot="2D_warp.png", cpos=[1, 1, 1])


# # Plotting a 2D MeshTags
# # ======================

# def left(x):
#     # Mark sphere with radius < sqrt(2)
#     return np.array((x.T[0] - 0.5)**2 + (x.T[1] - 0.5)**2 < 0.2**2, dtype=np.int32)


# # Create a cell-tag for all cells, with either value 0 or 1
# num_cells = mesh2D.topology.index_map(mesh2D.topology.dim).size_local
# midpoints = dolfinx.cpp.mesh.midpoints(mesh2D, mesh2D.topology.dim, np.arange(num_cells))
# cell_tags = dolfinx.MeshTags(mesh, mesh.topology.dim, np.arange(num_cells), left(midpoints))

# grid2D.cell_arrays["Marker"] = cell_tags.values
# grid2D.set_active_scalars("Marker")
# grid2D.plot(show_edges=True, cpos="xy", screenshot="2D_MeshTag.png", off_screen=True)


# # Plotting a subset of a mesh using MeshTags
# # ==========================================


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


# @numba.njit
# def extract_sub_topology(cells, cell_types, indices, new_cells, new_cell_types):
#     """
#     Extract the sub topology required for pyvista, given the indices of the cells
#     that should be extracted.
#     """
#     i, cell_counter, marked_counter = 0, 0, 0
#     while i < len(cells):
#         num_nodes = cells[i]
#         if isin(cell_counter, indices):
#             new_cells.append(num_nodes)
#             for node in cells[i + 1:i + num_nodes + 1]:
#                 new_cells.append(node)
#             new_cell_types[marked_counter] = cell_types[cell_counter]
#             marked_counter += 1
#         i += num_nodes + 1
#         cell_counter += 1


# # Create topology for cells marked with 1
# cell_subset = List.empty_list(numba.types.int32)
# cell_indices = cell_tags.indices[cell_tags.values == 1]
# cell_types_subset = np.zeros(len(cell_indices))
# extract_sub_topology(grid2D.cells, grid2D.celltypes, cell_indices, cell_subset, cell_types_subset)

# # Plot only a subset of a mesh with a given cell marker
# sub_grid = pyvista.UnstructuredGrid(np.array(cell_subset, dtype=np.int32), cell_types_subset, grid2D.points)
# sub_grid.plot(show_edges=True, cpos="xy", screenshot="submesh.png", off_screen=True, render_points_as_spheres=True)


mesh = dolfinx.UnitSquareMesh(MPI.COMM_WORLD, 15, 15, cell_type=dolfinx.cpp.mesh.CellType.triangle)


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
# Solve

solver_proj.solve(b, uh.vector)
uh.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                      mode=PETSc.ScatterMode.FORWARD)


@numba.njit(cache=True)
def extend_geometry_and_topology(nodes, cell_topology):
    traversed_nodes = List.empty_list(numba.types.int32)
    for i, cell in enumerate(cell_topology):
        for j, vertex in enumerate(cell):
            if len(traversed_nodes) == 0 or not isin(vertex, traversed_nodes):
                traversed_nodes.append(vertex)
            else:
                nodes = np.vstack((nodes, nodes[vertex].reshape((1, -1))))
                cell_topology[i, j] = nodes.shape[0] - 1
    return nodes


def visualize_DG_function(u):
    mesh = u.function_space.mesh
    assert(u.function_space.ufl_element().degree() == mesh.ufl_domain().ufl_coordinate_element().degree())
    num_cells = mesh.topology.index_map(mesh.topology.dim).size_local
    nodes = mesh.geometry.x.copy()
    cell_topology = dolfinx.cpp.mesh.entities_to_geometry(
        mesh, mesh.topology.dim, np.arange(num_cells), False)
    dofmap = u.function_space.dofmap.list.array
    offsets = u.function_space.dofmap.list.offsets
    u_arr = u.x.array()
    num_dofs_per_cell = u.function_space.dofmap.dof_layout.num_dofs
    values = np.zeros(num_cells * num_dofs_per_cell, dtype=np.float64)
    for i in range(num_cells):
        dofs = dofmap[offsets[i]:offsets[i + 1]]
        values[num_dofs_per_cell * i:num_dofs_per_cell * (i + 1)] = u_arr[dofs]
    new_geometry = extend_geometry_and_topology(nodes, cell_topology)

    sorted_indices = np.argsort(cell_topology.reshape(-1))
    point_values = values[sorted_indices]

    _dolfin_to_vtk_cell = {dolfinx.cpp.mesh.CellType.point: 1, dolfinx.cpp.mesh.CellType.interval: 3,
                           dolfinx.cpp.mesh.CellType.triangle: 5, dolfinx.cpp.mesh.CellType.quadrilateral: 9,
                           dolfinx.cpp.mesh.CellType.tetrahedron: 10, dolfinx.cpp.mesh.CellType.hexahedron: 12}
    dolfin_cell_type = dolfinx.cpp.mesh.cell_entity_type(mesh.topology.cell_type, mesh.topology.dim)
    cell_types = np.full(num_cells, _dolfin_to_vtk_cell[dolfin_cell_type])
    # Permute cell topology to VTK
    perm_to_vtk = dolfinx.cpp.io.transpose_map(dolfinx.cpp.io.perm_vtk(dolfin_cell_type, cell_topology.shape[1]))
    topology = cell_topology[:, perm_to_vtk]
    flattened_topology = np.hstack([np.full((num_cells, 1), topology.shape[1]), topology]).reshape(-1)

    return new_geometry, flattened_topology, cell_types, point_values


@numba.njit(cache=True)
def create_dg_cell_topology(topology, perm_to_vtk, num_dofs_per_cell, dofmap, offsets):
    for i in range(num_cells):
        topology[i * (num_dofs_per_cell + 1)] = num_dofs_per_cell
        topology[i * (num_dofs_per_cell + 1) + 1: i * (num_dofs_per_cell + 1)
                 + num_dofs_per_cell + 1] = dofmap[offsets[i]:offsets[i + 1]][perm_to_vtk]


def visualize_DG_function_new(u):
    V = u.function_space
    geometry = V.tabulate_dof_coordinates()
    point_values = u.vector.array

    mesh = u.function_space.mesh
    num_cells = mesh.topology.index_map(mesh.topology.dim).size_local
    dofmap = u.function_space.dofmap.list.array
    offsets = u.function_space.dofmap.list.offsets
    num_dofs_per_cell = u.function_space.dofmap.dof_layout.num_dofs
    topology = np.zeros(num_cells * (num_dofs_per_cell + 1), dtype=np.int32)

    dolfin_cell_type = dolfinx.cpp.mesh.cell_entity_type(mesh.topology.cell_type, mesh.topology.dim)
    perm_to_vtk = dolfinx.cpp.io.transpose_map(dolfinx.cpp.io.perm_vtk(dolfin_cell_type, num_dofs_per_cell))
    create_dg_cell_topology(topology, np.array(perm_to_vtk, dtype=np.int32), num_dofs_per_cell, dofmap, offsets)

    _dolfin_to_vtk_cell = {dolfinx.cpp.mesh.CellType.point: 1, dolfinx.cpp.mesh.CellType.interval: 3,
                           dolfinx.cpp.mesh.CellType.triangle: 69, dolfinx.cpp.mesh.CellType.quadrilateral: 9,
                           dolfinx.cpp.mesh.CellType.tetrahedron: 10, dolfinx.cpp.mesh.CellType.hexahedron: 12}
    cell_types = np.full(num_cells, _dolfin_to_vtk_cell[dolfin_cell_type])
    return geometry, topology, cell_types, point_values


geometry, topology, cell_types, values = visualize_DG_function_new(uh)
# geometry, topology, cell_types, values = visualize_DG_function(uh)


# Plot only a subset of a mesh with a given cell marker
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
grid.point_arrays["DG"] = values
grid.set_active_scalars("DG")


# warped.plot(show_edges=True, off_screen=True, screenshot="DG.png", cpos=[0.001, -0.1, 0.01])
vertices = pyvista.PolyData(grid.points)
vertices.point_arrays["DG"] = values
vertices.set_active_scalars("DG")

org_grid = pyvista.UnstructuredGrid(*dolfinx.io.mesh_to_pyvista(mesh, mesh.topology.dim))
plotter = pyvista.Plotter(off_screen=True)
plotter.add_mesh(grid, show_edges=False)
plotter.add_mesh(org_grid, show_edges=True, color="black", style="wireframe")
plotter.add_mesh(vertices, point_size=7, render_points_as_spheres=True)
plotter.show(screenshot="DG.png", cpos="xy")  # [0.001, -0.1, 0.01])


# grid.plot(show_edges=True, cpos="xy", screenshot="DG.png",
#           off_screen=True, render_points_as_spheres=True, scalars="DG")
