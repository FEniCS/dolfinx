# Copyright (C) 2021 Jørgen S. Dokken and Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Support functions for plotting"""

import functools
import warnings

import numpy as np

from dolfinx import cpp as _cpp
from dolfinx import fem, mesh
from dolfinx.mesh import CellType

# NOTE: Edge visualization of higher order elements are sketchy, see:
# https://github.com/pyvista/pyvista/issues/947


# NOTE: These dictionaries and following function should be replaced by
# cpp.io.get_vtk_cell_type when plotting module has better support for
# arbitrary lagrangian elements
_first_order_vtk = {CellType.interval: 3,
                    CellType.triangle: 5,
                    CellType.quadrilateral: 9,
                    CellType.tetrahedron: 10,
                    CellType.hexahedron: 12}
_cell_degree_triangle = {3: 1, 6: 2, 10: 3, 15: 4, 21: 5, 28: 6, 36: 7, 45: 8, 55: 9}
_cell_degree_tetrahedron = {4: 1, 10: 2, 20: 3}
_cell_degree_hexahedron = {8: 1, 27: 2}


@functools.singledispatch
def create_vtk_mesh(mesh: mesh.Mesh, dim: int, entities=None):
    """Create vtk mesh topology data for mesh entities of a given
    dimension. The vertex indices in the returned topology array are the
    indices for the associated entry in the mesh geometry.

    """
    if entities is None:
        num_cells = mesh.topology.index_map(dim).size_local
        entities = np.arange(num_cells, dtype=np.int32)
    else:
        num_cells = len(entities)

    # Get the indices in the geometry array that correspong to the
    # topology vertices
    geometry_entities = _cpp.mesh.entities_to_geometry(mesh, dim, entities, False)

    # Array holding the cell type (shape) for each cell
    if mesh.topology.cell_type == CellType.prism:
        raise RuntimeError("Plotting of prism meshes not supported")
    e_type = _cpp.mesh.cell_entity_type(mesh.topology.cell_type, dim, 0)
    degree = mesh.geometry.cmap.degree
    if degree == 1:
        cell_types = np.full(num_cells, _first_order_vtk[e_type])
    else:
        warnings.warn("Plotting of higher order mesh topologies is experimental.")
        cell_types = np.full(num_cells, _cpp.io.get_vtk_cell_type(mesh, dim))

    # Get cell data and the DOLFINx -> VTK permutation array
    num_vertices_per_cell = geometry_entities.shape[1]
    map_vtk = np.argsort(_cpp.io.perm_vtk(e_type, num_vertices_per_cell))

    # Create mesh topology
    topology = np.zeros((num_cells, num_vertices_per_cell + 1), dtype=np.int32)
    topology[:, 0] = num_vertices_per_cell
    topology[:, 1:] = geometry_entities[:, map_vtk]

    return topology.reshape(1, -1)[0], cell_types, mesh.geometry.x


@create_vtk_mesh.register(fem.FunctionSpace)
def _(V: fem.FunctionSpace, entities=None):
    """Creates a VTK mesh topology (topology array and array of cell
    types) that is based on the degree-of-freedom coordinates. Note that
    this function supports Lagrange elements (continuous and
    discontinuous) only.

    """
    family = V.ufl_element().family()
    if not (family in ['Discontinuous Lagrange', "Lagrange", "DQ", "Q"]):
        raise RuntimeError("Can only create meshes from Continuous or Discontinuous function-spaces")
    if V.ufl_element().degree() == 0:
        raise RuntimeError("Cannot create topology from cellwise constants.")

    mesh = V.mesh
    if entities is None:
        num_cells = mesh.topology.index_map(mesh.topology.dim).size_local
        entities = np.arange(num_cells, dtype=np.int32)
    else:
        num_cells = entities.size

    dofmap = V.dofmap
    num_dofs_per_cell = V.dofmap.dof_layout.num_dofs
    degree = V.ufl_element().degree()
    cell_type = mesh.topology.cell_type
    perm = np.argsort(_cpp.io.perm_vtk(cell_type, num_dofs_per_cell))

    if degree == 1:
        cell_types = np.full(num_cells, _first_order_vtk[mesh.topology.cell_type])
    else:
        warnings.warn("Plotting of higher order functions is experimental.")
        cell_types = np.full(num_cells, _cpp.io.get_vtk_cell_type(mesh, mesh.topology.dim))

    topology = np.zeros((num_cells, num_dofs_per_cell + 1), dtype=np.int32)
    topology[:, 0] = num_dofs_per_cell
    dofmap_ = dofmap.list.array.reshape(dofmap.list.num_nodes, num_dofs_per_cell)

    topology[:, 1:] = dofmap_[:num_cells, perm]
    return topology.reshape(1, -1)[0], cell_types, V.tabulate_dof_coordinates()
