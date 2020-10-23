# Copyright (C) 2020 Michal Habera
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import os

import numpy as np
import pytest
from dolfinx.cpp.mesh import CellType
from dolfinx.generation import UnitCubeMesh
from dolfinx.io import XDMFFile
from dolfinx.mesh import MeshTags, locate_entities
from dolfinx_utils.test.fixtures import tempdir
from mpi4py import MPI

assert (tempdir)

# Supported XDMF file encoding
if MPI.COMM_WORLD.size > 1:
    encodings = (XDMFFile.Encoding.HDF5, )
else:
    encodings = (XDMFFile.Encoding.ASCII, XDMFFile.Encoding.HDF5)

celltypes_3D = [CellType.tetrahedron, CellType.hexahedron]


@pytest.mark.parametrize("cell_type", celltypes_3D)
@pytest.mark.parametrize("encoding", encodings)
def test_3d(tempdir, cell_type, encoding):
    filename = os.path.join(tempdir, "meshtags_3d.xdmf")
    comm = MPI.COMM_WORLD
    mesh = UnitCubeMesh(comm, 4, 4, 4, cell_type)

    bottom_facets = locate_entities(mesh, 2, lambda x: np.isclose(x[1], 0.0))
    bottom_values = np.full(bottom_facets.shape, 1, dtype=np.intc)
    left_facets = locate_entities(mesh, 2, lambda x: np.isclose(x[0], 0.0))
    left_values = np.full(left_facets.shape, 2, dtype=np.intc)

    indices, pos = np.unique(np.hstack((bottom_facets, left_facets)), return_index=True)
    mt = MeshTags(mesh, 2, indices, np.hstack((bottom_values, left_values))[pos])
    mt.name = "facets"

    top_lines = locate_entities(mesh, 1, lambda x: np.isclose(x[2], 1.0))
    top_values = np.full(top_lines.shape, 3, dtype=np.intc)
    right_lines = locate_entities(mesh, 1, lambda x: np.isclose(x[0], 1.0))
    right_values = np.full(right_lines.shape, 4, dtype=np.intc)

    indices, pos = np.unique(np.hstack((top_lines, right_lines)), return_index=True)
    mt_lines = MeshTags(mesh, 1, indices, np.hstack((top_values, right_values))[pos])
    mt_lines.name = "lines"

    with XDMFFile(comm, filename, "w", encoding=encoding) as file:
        mesh.topology.create_connectivity_all()
        file.write_mesh(mesh)
        file.write_meshtags(mt)
        file.write_meshtags(mt_lines)
        file.write_information("units", "mm")

    with XDMFFile(comm, filename, "r", encoding=encoding) as file:
        mesh_in = file.read_mesh()
        mesh_in.topology.create_connectivity_all()
        mt_in = file.read_meshtags(mesh_in, "facets")
        mt_lines_in = file.read_meshtags(mesh_in, "lines")
        units = file.read_information("units")
        assert units == "mm"
        assert mt_in.name == "facets"
        assert mt_lines_in.name == "lines"

    with XDMFFile(comm, os.path.join(tempdir, "meshtags_3d_out.xdmf"), "w", encoding=encoding) as file:
        file.write_mesh(mesh_in)
        file.write_meshtags(mt_lines_in)
        file.write_meshtags(mt_in)

    # Check number of owned and marked entities
    lines_local = comm.allreduce((mt_lines.indices < mesh.topology.index_map(1).size_local).sum(), op=MPI.SUM)
    lines_local_in = comm.allreduce(
        (mt_lines_in.indices < mesh_in.topology.index_map(1).size_local).sum(), op=MPI.SUM)

    assert lines_local == lines_local_in
