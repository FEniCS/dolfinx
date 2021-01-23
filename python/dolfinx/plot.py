# Copyright (C) 2021 Jørgen S. Dokken and Garth N. Wells
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Support function for plotting"""

import functools

import numpy as np

from dolfinx import cpp, fem

# Permutation for Dolfinx DG layout to VTK
# Note that third order tetrahedrons has a special ordering: https://gitlab.kitware.com/vtk/vtk/-/issues/17746
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


def _transpose(map):
    """Transpose of the map. E.g., is `map = [1, 2, 3, 0]`, the
    transpose will be `[3 , 0, 1, 2 ]`.

    """
    transpose = np.zeros(len(map), dtype=np.int32)
    for i in range(len(map)):
        transpose[map[i]] = i
    return transpose


@functools.singledispatch
def create_pyvista_topology(mesh: cpp.mesh.Mesh, dim: int, entities=None):
    """Create pyvista mesh topology data for mesh entities of a given
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
    cell_types = np.full(num_cells, cpp.io.get_vtk_cell_type(mesh, dim))

    # Get cell data and the DOLFINX -> VTK permutation array
    num_vertices_per_cell = geometry_entities.shape[1]
    e_type = cpp.mesh.cell_entity_type(mesh.topology.cell_type, dim)
    map_vtk = _transpose(cpp.io.perm_vtk(e_type, num_vertices_per_cell))

    # Create mesh topology
    topology = np.zeros((num_cells, num_vertices_per_cell + 1), dtype=np.int32)
    topology[:, 0] = num_vertices_per_cell
    topology[:, 1:] = geometry_entities[:, map_vtk]
    return topology.reshape(1, -1)[0], cell_types


@create_pyvista_topology.register(fem.FunctionSpace)
def _(V: fem.FunctionSpace, entities=None):
    """Creates a pyvista mesh topology (topology array and array of cell
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

    cell_type = mesh.topology.cell_type
    if family == "Discontinuous Lagrange":
        perm = np.array(_perm_dg[cell_type][V.ufl_element().degree()], dtype=np.int32)
    elif family == "DQ":
        perm = np.array(_perm_dq[cell_type][V.ufl_element().degree()], dtype=np.int32)
    else:
        perm = _transpose(cpp.io.perm_vtk(cell_type, num_dofs_per_cell))

    cell_types = np.full(num_cells, cpp.io.get_vtk_cell_type(mesh, mesh.topology.dim))
    topology = np.zeros((num_cells, num_dofs_per_cell + 1), dtype=np.int32)
    topology[:, 0] = num_dofs_per_cell
    dofmap_ = dofmap.list.array.reshape(dofmap.list.num_nodes, num_dofs_per_cell)

    topology[:, 1:] = dofmap_[:num_cells, perm]
    return topology.reshape(1, -1)[0], cell_types
