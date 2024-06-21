# Copyright (C) 2021-2022 Jørgen S. Dokken and Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Support functions for plotting"""

import functools
import typing

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
_first_order_vtk = {
    mesh.CellType.interval: 3,
    mesh.CellType.triangle: 5,
    mesh.CellType.quadrilateral: 9,
    mesh.CellType.tetrahedron: 10,
    mesh.CellType.hexahedron: 12,
}


@functools.singledispatch
def vtk_mesh(msh: mesh.Mesh, dim: typing.Optional[int] = None, entities=None):
    """Create vtk mesh topology data for mesh entities of a given
    dimension. The vertex indices in the returned topology array are the
    indices for the associated entry in the mesh geometry.

    Args:
        mesh: Mesh to extract data from.
        dim: Topological dimension of entities to extract.
        entities: Entities to extract. Extract all if ``None``.

    Returns:
        Topology, type for each cell, and geometry in VTK-ready format.

    """
    if dim is None:
        dim = msh.topology.dim

    cell_type = _cpp.mesh.cell_entity_type(msh.topology.cell_type, dim, 0)
    if cell_type == mesh.CellType.prism:
        raise RuntimeError("Plotting of prism meshes not supported")

    # Use all local cells if not supplied
    if entities is None:
        entities = np.arange(msh.topology.index_map(dim).size_local, dtype=np.int32)

    msh.topology.create_connectivity(dim, msh.topology.dim)
    msh.topology.create_connectivity(msh.topology.dim, dim)
    geometry_entities = _cpp.mesh.entities_to_geometry(msh._cpp_object, dim, entities, False)

    num_nodes_per_cell = geometry_entities.shape[1]
    map_vtk = np.argsort(_cpp.io.perm_vtk(cell_type, num_nodes_per_cell))
    vtk_topology = geometry_entities[:, map_vtk]

    # Create mesh topology
    topology = np.empty((len(entities), num_nodes_per_cell + 1), dtype=np.int32)
    topology[:, 0] = num_nodes_per_cell
    topology[:, 1:] = vtk_topology

    # Array holding the cell type (shape) for each cell
    vtk_type = _cpp.io.get_vtk_cell_type(cell_type, dim)
    cell_types = np.full(len(entities), vtk_type)

    return topology.reshape(-1), cell_types, msh.geometry.x


@vtk_mesh.register(fem.FunctionSpace)
def _(V: fem.FunctionSpace, entities=None):
    """Creates a VTK mesh topology (topology array and array of cell
    types) that is based on the degree-of-freedom coordinates.

    This function supports visualisation when the degree of the finite
    element space is different from the geometric degree of the mesh.

    Note:
        This function supports Lagrange elements (continuous and
        discontinuous) only.

    Args:
        V: Mesh to extract data from.
        entities: Entities to extract. Extract all if ``None``.

    Returns:
        Topology, type for each cell, and geometry in VTK-ready format.

    """
    if V.ufl_element().family_name not in [
        "Discontinuous Lagrange",
        "Lagrange",
        "DQ",
        "Q",
        "DP",
        "P",
    ]:
        raise RuntimeError(
            "Can only create meshes from continuous or discontinuous Lagrange spaces"
        )

    degree = V.ufl_element().degree
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

    vtk_type = (
        _first_order_vtk[cell_type] if degree == 1 else _cpp.io.get_vtk_cell_type(cell_type, tdim)
    )
    cell_types = np.full(len(entities), vtk_type)

    topology = np.zeros((len(entities), num_dofs_per_cell + 1), dtype=np.int32)
    topology[:, 0] = num_dofs_per_cell
    dofmap_ = dofmap.list

    topology[:, 1:] = dofmap_[: len(entities), perm]
    return topology.reshape(1, -1)[0], cell_types, V.tabulate_dof_coordinates()
