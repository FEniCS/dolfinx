# Copyright (C) 2022 Jørgen S. Dokken
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Tools to extract data from Gmsh models"""
import typing

import basix
import basix.ufl
import numpy as np
import ufl
from dolfinx.cpp.graph import AdjacencyList_int32
from dolfinx.mesh import GhostMode, Mesh, create_cell_partitioner, create_mesh
from mpi4py import MPI as _MPI

from dolfinx import cpp as _cpp

__all__ = ["ufl_mesh"]

try:
    import netgen as ng
    from ngsPETSc.plex import DMPlexMapping
    _has_ngs = True
except ModuleNotFoundError:
    _has_ngs = False


# Map from Gmsh cell type identifier (integer) to DOLFINx cell type
# and degree http://gmsh.info//doc/texinfo/gmsh.html#MSH-file-format
_ngs_to_cells = {(2, 3): ("triangle", 1),
                 (2, 4): ("quadrilateral", 1),
                 (3, 4): ("tetrahedron", 1)}


def ufl_mesh(ngs_cell: typing.Tuple[int, int], gdim: int) -> ufl.Mesh:
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
    cell = ufl.Cell(shape, geometric_dimension=gdim)

    element = basix.ufl.element(
        basix.ElementFamily.P, cell.cellname(), degree, basix.LagrangeVariant.equispaced, shape=(gdim, ),
        gdim=gdim)
    return ufl.Mesh(element)


if _has_ngs:
    __all__ += ["model_to_mesh"]

    def model_to_mesh(geo: ng.libngpy._geom2d, comm: _MPI.Comm, hmax: float, gdim: int = 2,
                      partitioner: typing.Callable[
            [_MPI.Comm, int, int, AdjacencyList_int32], AdjacencyList_int32] =
            create_cell_partitioner(GhostMode.none),
            transform: typing.Any = None, routine: typing.Any = None) -> Mesh:
        """Given a NetGen model, take all physical entities of the highest
        topological dimension and create the corresponding DOLFINx mesh.

        This function only works in serial, at the moment.

        Args:
            comm: The MPI communicator to use for mesh creation
            hmax: The maximum diameter of the elements in the triangulation
            model: The NetGen model
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
        # Apply any ngs routine post meshing
        if routine is not None:
            ngmesh, geo = routine(ngmesh, geo)
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
            T = np.array([list(np.trim_zeros(a, 'b')) for a in list(T)]) - 1
        elif ngmesh.dim == 3:
            V = ngmesh.Coordinates()
            T = ngmesh.Elements3D().NumPy()["nodes"]
            T = np.array([list(np.trim_zeros(a, 'b')) for a in list(T)]) - 1
        ufl_domain = ufl_mesh((gdim, T.shape[1]), gdim)
        cell_perm = _cpp.io.perm_gmsh(_cpp.mesh.to_type(
            str(ufl_domain.ufl_cell())), T.shape[1])
        T = T[:, cell_perm]
        mesh = create_mesh(comm, T, V, ufl_domain, partitioner)
        return mesh
