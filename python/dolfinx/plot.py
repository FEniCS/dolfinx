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
from dolfinx import fem
from dolfinx import mesh as _mesh

# NOTE: This dictionary and following function should be revised when
# pyvista has better support for arbitrary lagrangian elements, see:
# https://github.com/pyvista/pyvista/issues/947
# Cell types can be found at
# https://vtk.org/doc/nightly/html/vtkCellType_8h_source.html
_first_order_vtk = {_mesh.CellType.interval: 3,
                    _mesh.CellType.triangle: 5,
                    _mesh.CellType.quadrilateral: 9,
                    _mesh.CellType.tetrahedron: 10,
                    _mesh.CellType.hexahedron: 12}


@functools.singledispatch
def create_vtk_mesh(mesh: _mesh.Mesh, dim: int, entities=None):
    """Create vtk mesh topology data for mesh entities of a given
    dimension. The vertex indices in the returned topology array are the
    indices for the associated entry in the mesh geometry.

    """
    cell_type = mesh.topology.cell_type
    degree = mesh.geometry.cmap.degree
    if cell_type == _mesh.CellType.prism:
        raise RuntimeError("Plotting of prism meshes not supported")

    # Use all cells local to process if not supplied
    if entities is None:
        num_cells = mesh.topology.index_map(dim).size_local
        entities = np.arange(num_cells, dtype=np.int32)
    else:
        num_cells = len(entities)

    if dim == mesh.topology.dim:
        vtk_topology = _cpp.io.extract_vtk_connectivity(mesh)[entities]
        num_nodes_per_cell = vtk_topology.shape[1]
    else:
        # NOTE: This linearizes higher order geometries
        geometry_entities = _cpp.mesh.entities_to_geometry(mesh, dim, entities, False)
        if degree > 1:
            warnings.warn("Linearizing topology for higher order sub entities.")
        e_type = _cpp.mesh.cell_entity_type(cell_type, dim, 0)

        # Get cell data and the DOLFINx -> VTK permutation array
        num_nodes_per_cell = geometry_entities.shape[1]
        map_vtk = np.argsort(_cpp.io.perm_vtk(e_type, num_nodes_per_cell))
        vtk_topology = geometry_entities[:, map_vtk]

    # Create mesh topology
    topology = np.empty((num_cells, num_nodes_per_cell + 1), dtype=np.int32)
    topology[:, 0] = num_nodes_per_cell
    topology[:, 1:] = vtk_topology

    # Array holding the cell type (shape) for each cell
    vtk_type = _first_order_vtk[cell_type] if degree == 1 else _cpp.io.get_vtk_cell_type(cell_type, mesh.topology.dim)
    cell_types = np.full(num_cells, vtk_type)

    return topology.reshape(-1), cell_types, mesh.geometry.x


@create_vtk_mesh.register(fem.FunctionSpace)
def _(V: fem.FunctionSpace, entities=None):
    """Creates a VTK mesh topology (topology array and array of cell
    types) that is based on the degree-of-freedom coordinates. Note that
    this function supports Lagrange elements (continuous and
    discontinuous) only.

    """
    family = V.ufl_element().family()
    degree = V.ufl_element().degree()
    if not (family in ['Discontinuous Lagrange', "Lagrange", "DQ", "Q"]):
        raise RuntimeError("Can only create meshes from Continuous or Discontinuous function-spaces")
    if degree == 0:
        raise RuntimeError("Cannot create topology from cellwise constants.")
    elif degree > 1:
        warnings.warn("Plotting of higher order Lagrange functions are experimental.")

    mesh = V.mesh
    if entities is None:
        num_cells = mesh.topology.index_map(mesh.topology.dim).size_local
        entities = np.arange(num_cells, dtype=np.int32)
    else:
        num_cells = entities.size

    dofmap = V.dofmap
    num_dofs_per_cell = V.dofmap.dof_layout.num_dofs
    cell_type = mesh.topology.cell_type
    perm = np.argsort(_cpp.io.perm_vtk(cell_type, num_dofs_per_cell))

    vtk_type = _first_order_vtk[cell_type] if degree == 1 else _cpp.io.get_vtk_cell_type(cell_type, mesh.topology.dim)
    cell_types = np.full(num_cells, vtk_type)

    topology = np.zeros((num_cells, num_dofs_per_cell + 1), dtype=np.int32)
    topology[:, 0] = num_dofs_per_cell
    dofmap_ = dofmap.list.array.reshape(dofmap.list.num_nodes, num_dofs_per_cell)

    topology[:, 1:] = dofmap_[:num_cells, perm]
    return topology.reshape(1, -1)[0], cell_types, V.tabulate_dof_coordinates()
