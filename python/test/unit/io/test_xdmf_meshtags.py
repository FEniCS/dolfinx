# Copyright (C) 2020 Michal Habera
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import numpy
import os
import dolfinx
from dolfinx.generation import UnitCubeMesh, UnitSquareMesh
from dolfinx import MPI
from dolfinx.io import XDMFFile
from dolfinx.cpp.mesh import CellType
from dolfinx.mesh import locate_entities_geometrical, MeshTags
import pytest
from dolfinx_utils.test.fixtures import tempdir

# Supported XDMF file encoding
if MPI.size(MPI.comm_world) > 1:
    encodings = (XDMFFile.Encoding.HDF5, )
else:
    encodings = (XDMFFile.Encoding.HDF5, XDMFFile.Encoding.ASCII)
    encodings = (XDMFFile.Encoding.HDF5, )

celltypes_3D = [CellType.tetrahedron]#, CellType.hexahedron]
celltypes_2D = [CellType.triangle]


@pytest.mark.parametrize("cell_type", celltypes_2D)
@pytest.mark.parametrize("encoding", encodings)
def test_3d(tempdir, cell_type, encoding):
    filename = os.path.join(tempdir, "meshtags_2d.xdmf")
    mesh = UnitSquareMesh(MPI.comm_world, 10, 10, cell_type)

    rank = MPI.rank(MPI.comm_world)

    bottom_facets = locate_entities_geometrical(mesh, 1, lambda x: numpy.isclose(x[1], 0.0))
    left_facets = locate_entities_geometrical(mesh, 1, lambda x: numpy.isclose(x[0], 0.0))

    # print(rank, bottom_facets)
    # print(rank, mesh.topology.index_map(2).size_local)
    # exit()

    bottom_values = numpy.full(bottom_facets.shape, 1, dtype=numpy.intc)
    left_values = numpy.full(left_facets.shape, 2, dtype=numpy.intc)

    mt = MeshTags(mesh, 1, bottom_facets, bottom_values)
    mt.append(left_facets, left_values)
    mt.name = "mt_facets"

    num_cells = mesh.topology.index_map(mesh.topology.dim).size_local
    partition = numpy.arange(0, num_cells)

    mt_part = MeshTags(mesh, 2, partition, numpy.full(partition.shape, rank, dtype=numpy.intc))
    mt_part.name = "part"

    # top_lines = locate_entities_geometrical(mesh, 1, lambda x: numpy.isclose(x[2], 1.0))
    # right_lines = locate_entities_geometrical(mesh, 1, lambda x: numpy.isclose(x[0], 1.0))

    # top_values = numpy.full(top_lines.shape, 3, dtype=numpy.intc)
    # right_values = numpy.full(right_lines.shape, 4, dtype=numpy.intc)

    # mt_lines = MeshTags(mesh, 1, top_lines, top_values)
    # mt_lines.append_unique(right_lines, right_values)
    # mt_lines.name = "mt_lines"

    with XDMFFile(MPI.comm_world, filename, "w", encoding=encoding) as file:
        file.write_mesh(mesh)
        file.write_meshtags(mt)
        file.write_meshtags(mt_part)

    mesh.mpi_comm().barrier()

    # with XDMFFile(mesh.mpi_comm(), filename, "a", encoding=encoding) as file:
    #     file.write_meshtags(mt_lines)

    with XDMFFile(MPI.comm_world, filename, "r", encoding=encoding) as file:
        mesh_in = file.read_mesh()
        # print(MPI.rank(MPI.comm_world), numpy.max(mesh_in.geometry.flags),
        #       mesh_in.geometry.index_map().size_local + mesh_in.geometry.index_map().num_ghosts, len(mesh_in.geometry.flags))
        # print(rank, mesh_in.geometry.global_indices())
        # print(rank, mesh_in.geometry.flags)
        # exit()
        mt_in = file.read_meshtags(mesh_in, "mt_facets")
        print(mt_in.indices)

    num_cells = mesh_in.topology.index_map(mesh_in.topology.dim).size_local
    partition = numpy.arange(0, num_cells)

    mt_in_part = MeshTags(mesh_in, 2, partition, numpy.full(partition.shape, rank, dtype=numpy.intc))
    mt_in_part.name = "part_in"

    with XDMFFile(MPI.comm_world, os.path.join(tempdir, "out_meshtags_2d.xdmf"), "w", encoding=encoding) as file:
        file.write_mesh(mesh_in)
        file.write_meshtags(mt_in_part)
        file.write_meshtags(mt_in)

    # assert numpy.allclose(mt_lines_in.indices, mt_lines.indices)
