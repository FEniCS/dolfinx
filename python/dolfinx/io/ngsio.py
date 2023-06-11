# Copyright (C) 2022 Jørgen S. Dokken
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Tools to extract data from Gmsh models"""
import typing

import numpy as np
import numpy.typing as npt

import basix
import basix.ufl
import ufl
from dolfinx import cpp as _cpp
from dolfinx.cpp.graph import AdjacencyList_int32
from dolfinx.mesh import (CellType, GhostMode, Mesh, create_cell_partitioner,
                          create_mesh, meshtags, meshtags_from_entities)

from mpi4py import MPI as _MPI

__all__ = ["cell_perm_array", "ufl_mesh"]

try:
    import ngsolve as ngs
    from ngsPETSc.plex import DMPlexMapping
    import netgen as ng
    _has_ngs = True
except ModuleNotFoundError:
    _has_ngs = False


# Map from Gmsh cell type identifier (integer) to DOLFINx cell type
# and degree http://gmsh.info//doc/texinfo/gmsh.html#MSH-file-format
_ngs_to_cells = {1: ("interval", 1), (2,3): ("triangle", 1),
                  (2,4): ("quadrilateral", 1), (3,4): ("tetrahedron", 1),
                  5: ("hexahedron", 1), 8: ("interval", 2),
                  9: ("triangle", 2), 10: ("quadrilateral", 2),
                  11: ("tetrahedron", 2), 12: ("hexahedron", 2),
                  15: ("point", 0), 21: ("triangle", 3),
                  26: ("interval", 3), 29: ("tetrahedron", 3),
                  36: ("quadrilateral", 3)}

def ufl_mesh(ngs_cell: int, gdim: int) -> ufl.Mesh:
    """Create a UFL mesh from a Gmsh cell identifier and the geometric dimension.

    See: http://gmsh.info//doc/texinfo/gmsh.html#MSH-file-format.

    Args:
        gmsh_cell: The Gmsh cell identifier
        gdim: The geometric dimension of the mesh

    Returns:
        A ufl Mesh using Lagrange elements (equispaced) of the
        corresponding DOLFINx cell
    """
    shape, degree = _ngs_to_cells[ngs_cell]
    print("Shape: {}, Degree: {}".format(shape,degree))
    cell = ufl.Cell(shape, geometric_dimension=gdim)

    element = basix.ufl.element(
        basix.ElementFamily.P, cell.cellname(), degree, basix.LagrangeVariant.equispaced, shape=(gdim, ),
        gdim=gdim)
    return ufl.Mesh(element)


def cell_perm_array(cell_type: CellType, num_nodes: int) -> typing.List[int]:
    """The permutation array for permuting Gmsh ordering to DOLFINx ordering.

    Args:
        cell_type: The DOLFINx cell type
        num_nodes: The number of nodes in the cell

    Returns:
        An array `p` such that `a_dolfinx[i] = a_gmsh[p[i]]`.
    """
    return _cpp.io.perm_gmsh(cell_type, num_nodes)


if _has_ngs:
    __all__ += ["model_to_mesh"]

    def model_to_mesh(geo: ng.libngpy._geom2d, comm: _MPI.Comm, hmax: float, gdim: int = 2,
                      partitioner: typing.Callable[
            [_MPI.Comm, int, int, AdjacencyList_int32], AdjacencyList_int32] =
            create_cell_partitioner(GhostMode.none), transform: typing.Any = None) -> typing.Tuple[
            Mesh, _cpp.mesh.MeshTags_int32, _cpp.mesh.MeshTags_int32]:
        """Given a NetGen model, take all physical entities of the highest
        topological dimension and create the corresponding DOLFINx mesh.
        
        This function only works in serial, at the moment.

        Args:
            comm: The MPI communicator to use for mesh creation
            hmax: The rank the Gmsh model is initialized on
            model: The Gmsh model
            gdim: Geometrical dimension of the mesh
            partitioner: Function that computes the parallel
                distribution of cells across MPI ranks

        Returns:
            A triplet (mesh, cell_tags, facet_tags) where cell_tags hold
            markers for the cells and facet tags holds markers for
            facets (if tags are found in the NetGen model).
        """
        # First we generate a mesh
        ngmesh = geo.GenerateMesh(maxh=hmax)
        # Applying any PETSc Transform
        if transform is not None:
            meshMap = DMPlexMapping(ngmesh)
            plex = meshMap.plex
            plex.view()
            transform.setDM(meshMap.plex)
            transform.setUp()
            newplex = transform.apply(meshMap.plex)
            newplex.view()
            meshMap = DMPlexMapping(newplex)
            ngmesh = meshMap.ngmesh
        # We extract topology and geometry
        if ngmesh.dim == 2:
            V = ngmesh.Coordinates()
            T = ngmesh.Elements2D().NumPy()["nodes"]
            T = np.array([list(np.trim_zeros(a, 'b')) for a in list(T)])-1
        elif ngmesh.dim == 3:
            V = ngmesh.Coordinates()
            T = ngmesh.Elements3D().NumPy()["nodes"]
            T = np.array([list(np.trim_zeros(a, 'b')) for a in list(T)])-1
        print("Dimension: {} Points: {}, Num: {}".format(gdim,len(T[0]),T.shape[0]))
        ufl_domain = ufl_mesh((gdim,T.shape[1]),gdim)
        cell_perm = cell_perm_array(_cpp.mesh.to_type(str(ufl_domain.ufl_cell())), T.shape[1])
        T = T[:, cell_perm]
        mesh = create_mesh(comm, T, V, ufl_domain, partitioner)


        # Create MeshTags for cells
        #local_entities, local_values = _cpp.io.distribute_entity_data(
        #    mesh._cpp_object, mesh.topology.dim, cells, cell_values)
        #mesh.topology.create_connectivity(mesh.topology.dim, 0)
        #adj = _cpp.graph.AdjacencyList_int32(local_entities)
        #ct = meshtags_from_entities(mesh, mesh.topology.dim, adj, local_values.astype(np.int32, copy=False))
        #ct.name = "Cell tags"

        ## Create MeshTags for facets
        #topology = mesh.topology
        #if has_facet_data:
        #    # Permute facets from MSH to DOLFINx ordering
        #    # FIXME: This does not work for prism meshes
        #    if topology.cell_types[0] == CellType.prism or topology.cell_types[0] == CellType.pyramid:
        #        raise RuntimeError(f"Unsupported cell type {topology.cell_types[0]}")

        #    facet_type = _cpp.mesh.cell_entity_type(_cpp.mesh.to_type(str(ufl_domain.ufl_cell())), topology.dim - 1, 0)
        #    gmsh_facet_perm = cell_perm_array(facet_type, num_facet_nodes)
        #    marked_facets = marked_facets[:, gmsh_facet_perm]

        #    local_entities, local_values = _cpp.io.distribute_entity_data(
        #        mesh._cpp_object, mesh.topology.dim - 1, marked_facets, facet_values)
        #    mesh.topology.create_connectivity(topology.dim - 1, topology.dim)
        #    adj = _cpp.graph.AdjacencyList_int32(local_entities)
        #    ft = meshtags_from_entities(mesh, topology.dim - 1, adj, local_values.astype(np.int32, copy=False))
        #    ft.name = "Facet tags"
        #else:
        #    ft = meshtags(mesh, topology.dim - 1, np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32))

        return mesh

