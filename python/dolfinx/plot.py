# Copyright (C) 2021-2022 Jørgen S. Dokken and Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Support functions for plotting"""

import functools
import typing
import warnings

import numpy as np

from dolfinx import cpp as _cpp
from dolfinx import fem, mesh

# NOTE: This dictionary and the below function that uses it should be
# revised when pyvista improves rendering of 'arbitrary' Lagrange
# elements, i.e. for the VTK cell types that define a shape but allow
# arbitrary degree geometry. See
# https://github.com/pyvista/pyvista/issues/947.
#
# Cell types can be found at
# https://vtk.org/doc/nightly/html/vtkCellType_8h_source.html
_first_order_vtk = {mesh.CellType.interval: 3,
                    mesh.CellType.triangle: 5,
                    mesh.CellType.quadrilateral: 9,
                    mesh.CellType.tetrahedron: 10,
                    mesh.CellType.hexahedron: 12}


@functools.singledispatch
def create_vtk_mesh(msh: mesh.Mesh, dim: typing.Optional[int] = None, entities=None):
    """Create vtk mesh topology data for mesh entities of a given
    dimension. The vertex indices in the returned topology array are the
    indices for the associated entry in the mesh geometry.

    """
    if dim is None:
        dim = msh.topology.dim

    tdim = msh.topology.dim

    cell_type = _cpp.mesh.cell_entity_type(msh.topology.cell_type, dim, 0)
    degree = msh.geometry.cmap.degree
    if cell_type == mesh.CellType.prism:
        raise RuntimeError("Plotting of prism meshes not supported")

    # Use all local cells if not supplied
    if entities is None:
        entities = range(msh.topology.index_map(dim).size_local)

    if dim == tdim:
        vtk_topology = _cpp.io.extract_vtk_connectivity(msh)[entities]
        num_nodes_per_cell = vtk_topology.shape[1]
    else:
        # NOTE: This linearizes higher order geometries
        geometry_entities = _cpp.mesh.entities_to_geometry(msh, dim, entities, False)
        if degree > 1:
            warnings.warn("Linearizing topology for higher order sub entities.")

        # Get cell data and the DOLFINx -> VTK permutation array
        num_nodes_per_cell = geometry_entities.shape[1]
        map_vtk = np.argsort(_cpp.io.perm_vtk(cell_type, num_nodes_per_cell))
        vtk_topology = geometry_entities[:, map_vtk]

    # Create mesh topology
    topology = np.empty((len(entities), num_nodes_per_cell + 1), dtype=np.int32)
    topology[:, 0] = num_nodes_per_cell
    topology[:, 1:] = vtk_topology

    # Array holding the cell type (shape) for each cell
    vtk_type = _first_order_vtk[cell_type] if degree == 1 else _cpp.io.get_vtk_cell_type(cell_type, tdim)
    cell_types = np.full(len(entities), vtk_type)

    return topology.reshape(-1), cell_types, msh.geometry.x


@create_vtk_mesh.register(fem.FunctionSpace)
def _(V: fem.FunctionSpace, entities=None):
    """Creates a VTK mesh topology (topology array and array of cell
    types) that is based on the degree-of-freedom coordinates. Note that
    this function supports Lagrange elements (continuous and
    discontinuous) only.

    """
    if not (V.ufl_element().family() in ['Discontinuous Lagrange', "Lagrange", "DQ", "Q", "DP", "P"]):
        raise RuntimeError("Can only create meshes from continuous or discontinuous Lagrange spaces")

    degree = V.ufl_element().degree()
    if degree == 0:
        raise RuntimeError("Cannot create topology from cellwise constants.")

    # Use all local cells if not supplied
    msh = V.mesh
    tdim = msh.topology.dim
    if entities is None:
        entities = range(msh.topology.index_map(tdim).size_local)

    dofmap = V.dofmap
    num_dofs_per_cell = V.dofmap.dof_layout.num_dofs
    cell_type = msh.topology.cell_type
    perm = np.argsort(_cpp.io.perm_vtk(cell_type, num_dofs_per_cell))

    vtk_type = _first_order_vtk[cell_type] if degree == 1 else _cpp.io.get_vtk_cell_type(cell_type, tdim)
    cell_types = np.full(len(entities), vtk_type)

    topology = np.zeros((len(entities), num_dofs_per_cell + 1), dtype=np.int32)
    topology[:, 0] = num_dofs_per_cell
    dofmap_ = dofmap.list.array.reshape(dofmap.list.num_nodes, num_dofs_per_cell)

    topology[:, 1:] = dofmap_[:len(entities), perm]
    return topology.reshape(1, -1)[0], cell_types, V.tabulate_dof_coordinates()
