# Copyright (C) 2020 Michal Habera, Jørgen S. Dokken
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from pathlib import Path
from xml.etree import ElementTree as ET

from mpi4py import MPI

import numpy as np
import pytest

from dolfinx import default_real_type
from dolfinx.io import XDMFFile
from dolfinx.mesh import CellType, create_unit_cube, locate_entities, meshtags

# Supported XDMF file encoding
if MPI.COMM_WORLD.size > 1:
    encodings = [XDMFFile.Encoding.HDF5]
else:
    encodings = [XDMFFile.Encoding.ASCII, XDMFFile.Encoding.HDF5]

celltypes_3D = [CellType.tetrahedron, CellType.hexahedron]


@pytest.mark.skipif(default_real_type != np.float64, reason="float32 not supported yet")
@pytest.mark.parametrize("cell_type", celltypes_3D)
@pytest.mark.parametrize("encoding", encodings)
def test_3d(tempdir, cell_type, encoding):
    filename = Path(tempdir, "meshtags_3d.xdmf")
    comm = MPI.COMM_WORLD
    mesh = create_unit_cube(comm, 4, 4, 4, cell_type)
    mesh.topology.create_entities(2)

    bottom_facets = locate_entities(mesh, 2, lambda x: np.isclose(x[1], 0.0))
    bottom_values = np.full(bottom_facets.shape, 1, dtype=np.int32)
    left_facets = locate_entities(mesh, 2, lambda x: np.isclose(x[0], 0.0))
    left_values = np.full(left_facets.shape, 2, dtype=np.int32)

    indices, pos = np.unique(np.hstack((bottom_facets, left_facets)), return_index=True)
    mt = meshtags(mesh, 2, indices, np.hstack((bottom_values, left_values))[pos])
    mt.name = "facets"

    top_lines = locate_entities(mesh, 1, lambda x: np.isclose(x[2], 1.0))
    top_values = np.full(top_lines.shape, 3, dtype=np.int32)
    right_lines = locate_entities(mesh, 1, lambda x: np.isclose(x[0], 1.0))
    right_values = np.full(right_lines.shape, 4, dtype=np.int32)

    indices, pos = np.unique(np.hstack((top_lines, right_lines)), return_index=True)
    mt_lines = meshtags(mesh, 1, indices, np.hstack((top_values, right_values))[pos])
    mt_lines.name = "lines"

    mesh.topology.create_connectivity(1, 3)
    mesh.topology.create_connectivity(2, 3)

    with XDMFFile(comm, filename, "w", encoding=encoding) as file:
        file.write_mesh(mesh)
        file.write_meshtags(mt, mesh.geometry)
        file.write_meshtags(mt_lines, mesh.geometry)
        file.write_information("units", "mm")

    with XDMFFile(comm, filename, "r", encoding=encoding) as file:
        mesh_in = file.read_mesh()
        tdim = mesh_in.topology.dim
        mesh_in.topology.create_connectivity(tdim - 1, tdim)
        mesh_in.topology.create_connectivity(1, tdim)

        mt_in = file.read_meshtags(mesh_in, "facets")
        mt_lines_in = file.read_meshtags(mesh_in, "lines")
        units = file.read_information("units")
        assert units == "mm"
        assert mt_in.name == "facets"
        assert mt_lines_in.name == "lines"

    with XDMFFile(comm, Path(tempdir, "meshtags_3d_out.xdmf"), "w", encoding=encoding) as file:
        file.write_mesh(mesh_in)
        file.write_meshtags(mt_lines_in, mesh_in.geometry)
        file.write_meshtags(mt_in, mesh_in.geometry)

    # Check number of owned and marked entities
    lines_local = comm.allreduce(
        (mt_lines.indices < mesh.topology.index_map(1).size_local).sum(), op=MPI.SUM
    )
    lines_local_in = comm.allreduce(
        (mt_lines_in.indices < mesh_in.topology.index_map(1).size_local).sum(), op=MPI.SUM
    )

    assert lines_local == lines_local_in

    # Check that only owned data is written to file
    facets_local = comm.allreduce(
        (mt.indices < mesh.topology.index_map(2).size_local).sum(), op=MPI.SUM
    )
    parser = ET.XMLParser()
    tree = ET.parse(Path(tempdir, "meshtags_3d_out.xdmf"), parser)
    num_lines = int(tree.findall(".//Grid[@Name='lines']/Topology")[0].get("NumberOfElements"))
    num_facets = int(tree.findall(".//Grid[@Name='facets']/Topology")[0].get("NumberOfElements"))
    assert num_lines == lines_local
    assert num_facets == facets_local


@pytest.mark.parametrize("cell_type", celltypes_3D)
@pytest.mark.parametrize("encoding", encodings)
def test_read_named_meshtags(tempdir, cell_type, encoding):
    domain_value = 1
    material_value = 2

    filename = Path(tempdir, "named_meshtags.xdmf")
    comm = MPI.COMM_WORLD
    mesh = create_unit_cube(comm, 4, 4, 4, cell_type)

    indices = np.arange(mesh.topology.index_map(3).size_local)
    domain_values = np.full(indices.shape, domain_value, dtype=np.int32)
    mt_domains = meshtags(mesh, 3, indices, domain_values)
    mt_domains.name = "domain"

    material_values = np.full(indices.shape, material_value, dtype=np.int32)
    mt_materials = meshtags(mesh, 3, indices, material_values)
    mt_materials.name = "material"

    with XDMFFile(comm, filename, "w", encoding=encoding) as file:
        file.write_mesh(mesh)
        file.write_meshtags(mt_domains, mesh.geometry)
        file.write_meshtags(mt_materials, mesh.geometry)

    with XDMFFile(comm, filename, "r", encoding=encoding) as file:
        mesh_in = file.read_mesh()
        tdim = mesh_in.topology.dim
        mesh_in.topology.create_connectivity(1, tdim)

        mt_first_in = file.read_meshtags(mesh_in, "material")
        assert all(v == material_value for v in mt_first_in.values)

        mt_domains_in = file.read_meshtags(mesh_in, "domain")
        assert all(v == domain_value for v in mt_domains_in.values)

        mt_materials_in = file.read_meshtags(mesh_in, "material")
        assert all(v == material_value for v in mt_materials_in.values)
