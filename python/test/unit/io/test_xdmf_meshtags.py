# Copyright (C) 2020 Michal Habera
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import os

import mpi4py
import numpy
import pytest
from dolfinx_utils.test.fixtures import tempdir

from dolfinx.cpp.mesh import CellType
from dolfinx.generation import UnitCubeMesh
from dolfinx.io import XDMFFile
from dolfinx.mesh import MeshTags, locate_entities_geometrical

assert (tempdir)

# Supported XDMF file encoding
if mpi4py.MPI.COMM_WORLD.size > 1:
    encodings = (XDMFFile.Encoding.HDF5, )
else:
    encodings = (XDMFFile.Encoding.ASCII, XDMFFile.Encoding.HDF5)

celltypes_3D = [CellType.tetrahedron, CellType.hexahedron]


@pytest.mark.parametrize("cell_type", celltypes_3D)
@pytest.mark.parametrize("encoding", encodings)
def test_3d(tempdir, cell_type, encoding):
    filename = os.path.join(tempdir, "meshtags_3d.xdmf")
    comm = mpi4py.MPI.COMM_WORLD
    mesh = UnitCubeMesh(comm, 4, 4, 4, cell_type)

    bottom_facets = locate_entities_geometrical(mesh, 2, lambda x: numpy.isclose(x[1], 0.0))
    left_facets = locate_entities_geometrical(mesh, 2, lambda x: numpy.isclose(x[0], 0.0))

    bottom_values = numpy.full(bottom_facets.shape, 1, dtype=numpy.intc)
    left_values = numpy.full(left_facets.shape, 2, dtype=numpy.intc)

    mt = MeshTags(mesh, 2, numpy.hstack((bottom_facets, left_facets)), numpy.hstack((bottom_values, left_values)))
    mt.name = "facets"

    top_lines = locate_entities_geometrical(mesh, 1, lambda x: numpy.isclose(x[2], 1.0))
    right_lines = locate_entities_geometrical(mesh, 1, lambda x: numpy.isclose(x[0], 1.0))

    top_values = numpy.full(top_lines.shape, 3, dtype=numpy.intc)
    right_values = numpy.full(right_lines.shape, 4, dtype=numpy.intc)

    mt_lines = MeshTags(mesh, 1, numpy.hstack((top_lines, right_lines)), numpy.hstack((top_values, right_values)))
    mt_lines.name = "lines"

    with XDMFFile(comm, filename, "w", encoding=encoding) as file:
        mesh.topology.create_connectivity_all()
        file.write_mesh(mesh)
        file.write_meshtags(mt)
        file.write_meshtags(mt_lines)

    with XDMFFile(comm, filename, "r", encoding=encoding) as file:
        mesh_in = file.read_mesh()
        mesh_in.topology.create_connectivity_all()
        mt_in = file.read_meshtags(mesh_in, "facets")
        mt_lines_in = file.read_meshtags(mesh_in, "lines")

        assert mt_in.name == "facets"
        assert mt_lines_in.name == "lines"

    with XDMFFile(comm, os.path.join(tempdir, "meshtags_3d_out.xdmf"), "w", encoding=encoding) as file:
        file.write_mesh(mesh_in)
        file.write_meshtags(mt_lines_in)
        file.write_meshtags(mt_in)

    # Check number of owned and marked entities
    lines_local = comm.allreduce((mt_lines.indices < mesh.topology.index_map(1).size_local).sum(), op=mpi4py.MPI.SUM)
    lines_local_in = comm.allreduce(
        (mt_lines_in.indices < mesh_in.topology.index_map(1).size_local).sum(), op=mpi4py.MPI.SUM)

    assert lines_local == lines_local_in
