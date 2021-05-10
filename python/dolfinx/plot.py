# Copyright (C) 2021 Jørgen S. Dokken and Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Support functions for plotting"""

import functools
import warnings
import numpy as np

from dolfinx import cpp, fem

# Permutation for DOLFINx DG layout to VTK
# Note that third order tetrahedrons has a special ordering:
# https://gitlab.kitware.com/vtk/vtk/-/issues/17746
_perm_dg = {cpp.mesh.CellType.triangle: {1: [0, 1, 2], 2: [0, 2, 5, 1, 4, 3], 3: [0, 3, 9, 1, 2, 6, 8, 7, 4, 5],
                                         4: [0, 4, 14, 1, 2, 3, 8, 11, 13, 12, 9, 5, 6, 7, 10]},
            cpp.mesh.CellType.tetrahedron: {1: [0, 1, 2, 3], 2: [0, 2, 5, 9, 1, 4, 5, 6, 7, 8],
                                            3: [0, 3, 9, 19, 1, 2, 6, 8, 7, 4, 10, 16, 12, 17, 15, 18, 11, 14, 13, 5]}}
_perm_dq = {cpp.mesh.CellType.quadrilateral: {1: [0, 1, 3, 2], 2: [0, 2, 8, 6, 1, 5, 7, 3, 4],
                                              3: [0, 3, 15, 12, 1, 2, 7, 11, 13, 14, 4, 8, 5, 6, 9, 10]},
            cpp.mesh.CellType.hexahedron: {1: [0, 1, 3, 2, 4, 5, 7, 6],
                                           2: [0, 2, 8, 6, 18, 20, 26, 24, 1, 5, 7, 3, 19,
                                               23, 25, 21, 9, 11, 17, 15, 12, 14, 10, 16, 4, 22, 14]}}

# NOTE: Edge visualization of higher order elements are sketchy, see:
# https://github.com/pyvista/pyvista/issues/947


# NOTE: These dictionaries and following function should be replaced by
# cpp.io.get_vtk_cell_type when plotting module has better support for
# arbitrary lagrangian elements
_first_order_vtk = {cpp.mesh.CellType.interval: 3,
                    cpp.mesh.CellType.triangle: 5,
                    cpp.mesh.CellType.quadrilateral: 9,
                    cpp.mesh.CellType.tetrahedron: 10,
                    cpp.mesh.CellType.hexahedron: 12}
_cell_degree_triangle = {3: 1, 6: 2, 10: 3, 15: 4, 21: 5, 28: 6, 36: 7, 45: 8, 55: 9}
_cell_degree_tetrahedron = {4: 1, 10: 2, 20: 3}
_cell_degree_hexahedron = {8: 1, 27: 2}


def _element_degree(cell_type: cpp.mesh.CellType, num_nodes: int):
    """Determine the degree of a cell by the number of nodes"""
    if cell_type == cpp.mesh.CellType.triangle:
        return _cell_degree_triangle[num_nodes]
    elif cell_type == cpp.mesh.CellType.point:
        return 1
    elif cell_type == cpp.mesh.CellType.interval:
        return num_nodes - 1
    elif cell_type == cpp.mesh.CellType.tetrahedron:
        return _cell_degree_tetrahedron[num_nodes]
    elif cell_type == cpp.mesh.CellType.quadrilateral:
        return int(np.sqrt(num_nodes) - 1)
    elif cell_type == cpp.mesh.CellType.hexahedron:
        return _cell_degree_hexahedron[num_nodes]


@functools.singledispatch
def create_vtk_topology(mesh: cpp.mesh.Mesh, dim: int, entities=None):
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
    geometry_entities = cpp.mesh.entities_to_geometry(mesh, dim, entities, False)

    # Array holding the cell type (shape) for each cell
    e_type = cpp.mesh.cell_entity_type(mesh.topology.cell_type, dim)
    degree = _element_degree(e_type, geometry_entities.shape[1])
    if degree == 1:
        cell_types = np.full(num_cells, _first_order_vtk[e_type])
    else:
        warnings.warn("Plotting of higher order mesh topologies is experimental.")
        cell_types = np.full(num_cells, cpp.io.get_vtk_cell_type(mesh, dim))

    # Get cell data and the DOLFINx -> VTK permutation array
    num_vertices_per_cell = geometry_entities.shape[1]
    map_vtk = np.argsort(cpp.io.perm_vtk(e_type, num_vertices_per_cell))

    # Create mesh topology
    topology = np.zeros((num_cells, num_vertices_per_cell + 1), dtype=np.int32)
    topology[:, 0] = num_vertices_per_cell
    topology[:, 1:] = geometry_entities[:, map_vtk]
    return topology.reshape(1, -1)[0], cell_types


@create_vtk_topology.register(fem.FunctionSpace)
def _(V: fem.FunctionSpace, entities=None):
    """Creates a vtk mesh topology (topology array and array of cell
    types) that is based on degree of freedom coordinate. Note that this
    function supports Lagrange elements (continuous and discontinuous)
    only.

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
    if family == "Discontinuous Lagrange":
        perm = np.array(_perm_dg[cell_type][degree], dtype=np.int32)
    elif family == "DQ":
        perm = np.array(_perm_dq[cell_type][degree], dtype=np.int32)
    else:
        perm = np.argsort(cpp.io.perm_vtk(cell_type, num_dofs_per_cell))

    if degree == 1:
        cell_types = np.full(num_cells, _first_order_vtk[mesh.topology.cell_type])
    else:
        warnings.warn("Plotting of higher order functions is experimental.")
        cell_types = np.full(num_cells, cpp.io.get_vtk_cell_type(mesh, mesh.topology.dim))

    topology = np.zeros((num_cells, num_dofs_per_cell + 1), dtype=np.int32)
    topology[:, 0] = num_dofs_per_cell
    dofmap_ = dofmap.list.array.reshape(dofmap.list.num_nodes, num_dofs_per_cell)

    topology[:, 1:] = dofmap_[:num_cells, perm]
    return topology.reshape(1, -1)[0], cell_types
