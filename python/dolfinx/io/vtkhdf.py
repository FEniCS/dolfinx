# Copyright (C) 2024-2025 Chris Richardson and Jørgen S. Dokken
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from pathlib import Path

from mpi4py import MPI as _MPI

import numpy as np
import numpy.typing as npt

import basix
import ufl
from dolfinx.cpp.io import (
    read_vtkhdf_mesh_float32,
    read_vtkhdf_mesh_float64,
    write_vtkhdf_data,
    write_vtkhdf_mesh,
)
from dolfinx.mesh import Mesh


def read_mesh(
    comm: _MPI.Comm,
    filename: str | Path,
    dtype: npt.DTypeLike = np.float64,
    gdim: int = 3,
    max_facet_to_cell_links: int = 2,
):
    """Read a mesh from a VTKHDF format file.

    Note:
        Changing `max_facet_to_cell_links` from the default value should
        only be required when working on branching manifolds. Changing this
        value on non-branching meshes will only result in a slower mesh
        partitioning and creation.

    Args:
        comm: An MPI communicator.
        filename: File to read from.
        dtype: Scalar type of mesh geometry (need not match dtype in
            file).
        gdim: Geometric dimension of the mesh.
        max_facet_to_cell_links: Maximum number of cells that can be
            linked to a facet.
    """
    if dtype == np.float64:
        mesh_cpp = read_vtkhdf_mesh_float64(comm, filename, gdim, max_facet_to_cell_links)
    elif dtype == np.float32:
        mesh_cpp = read_vtkhdf_mesh_float32(comm, filename, gdim, max_facet_to_cell_links)

    cell_types = mesh_cpp.topology.entity_types[-1]
    if len(cell_types) > 1:
        # FIXME: not yet defined for mixed topology
        domain = None
    else:
        cell_degree = mesh_cpp.geometry.cmap.degree
        variant = mesh_cpp.geometry.cmap.variant
        domain = ufl.Mesh(
            basix.ufl.element(
                "Lagrange", cell_types[0].name, cell_degree, variant, shape=(mesh_cpp.geometry.dim,)
            )
        )
    return Mesh(mesh_cpp, domain)


def write_mesh(filename: str | Path, mesh: Mesh):
    """Write a mesh to file in VTKHDF format

    Args:
        filename: File to write to.
        mesh: Mesh.
    """
    write_vtkhdf_mesh(filename, mesh._cpp_object)


def write_point_data(filename: str | Path, mesh: Mesh, data: npt.NDArray, time: float):
    """Write data at vertices of the mesh.

    Args:
        filename: File to write to.
        mesh: Mesh.
        data: Data at the points of the mesh, local to each process.
        time: Timestamp.
    """
    write_vtkhdf_data("Point", filename, [mesh.geometry.index_map()], data, time)


def write_cell_data(filename: str | Path, mesh: Mesh, data: npt.NDArray, time: float):
    """Write data at cells of the mesh.

    Args:
        filename: File to write to.
        mesh: Mesh.
        data: Data at the cells of the mesh, local to each process.
        time: Timestamp.
    """
    write_vtkhdf_data(
        "Cell", filename, [im for im in mesh.topology.index_maps(mesh.topology.dim)], data, time
    )
