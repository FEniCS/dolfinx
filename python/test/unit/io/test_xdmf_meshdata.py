# Copyright (C) 2012-2020 Garth N. Wells, Jørgen S. Dokken
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import os
from xml.etree import ElementTree

import numpy as np
import pytest
from dolfinx import MeshTags, UnitCubeMesh, UnitIntervalMesh, UnitSquareMesh
from dolfinx.cpp.mesh import CellType
from dolfinx.io import XDMFFile
from dolfinx_utils.test.fixtures import tempdir
from mpi4py import MPI

assert (tempdir)

# Supported XDMF file encoding
if MPI.COMM_WORLD.size > 1:
    encodings = (XDMFFile.Encoding.HDF5, )
else:
    encodings = (XDMFFile.Encoding.HDF5, XDMFFile.Encoding.ASCII)
    encodings = (XDMFFile.Encoding.HDF5, )

celltypes_2D = [CellType.triangle, CellType.quadrilateral]
celltypes_3D = [CellType.tetrahedron, CellType.hexahedron]


def mesh_factory(tdim, n):
    if tdim == 1:
        return UnitIntervalMesh(MPI.COMM_WORLD, n)
    elif tdim == 2:
        return UnitSquareMesh(MPI.COMM_WORLD, n, n)
    elif tdim == 3:
        return UnitCubeMesh(MPI.COMM_WORLD, n, n, n)


@pytest.fixture
def worker_id(request):
    """Return worker ID when using pytest-xdist to run tests in parallel"""
    if hasattr(request.config, 'slaveinput'):
        return request.config.slaveinput['slaveid']
    else:
        return 'master'


@pytest.mark.parametrize("tdim", [2, 3])
@pytest.mark.parametrize("n", [6])
def test_read_mesh_data(tempdir, tdim, n):
    filename = os.path.join(tempdir, "mesh.xdmf")
    mesh = mesh_factory(tdim, n)
    encoding = XDMFFile.Encoding.HDF5
    with XDMFFile(mesh.mpi_comm(), filename, "w", encoding) as file:
        file.write_mesh(mesh)

    with XDMFFile(MPI.COMM_WORLD, filename, "r") as file:
        cell_type = file.read_cell_type()
        cells = file.read_topology_data()
        x = file.read_geometry_data()

    assert cell_type[0] == mesh.topology.cell_type
    assert cell_type[1] == 1
    assert mesh.topology.index_map(tdim).size_global == mesh.mpi_comm().allreduce(cells.shape[0], op=MPI.SUM)
    assert mesh.geometry.index_map().size_global == mesh.mpi_comm().allreduce(x.shape[0], op=MPI.SUM)


def test_write_mesh_data(tempdir):
    comm_world = MPI.COMM_WORLD
    comm_self = MPI.COMM_SELF
    filename_serial = os.path.join(tempdir, "mesh_serial.xdmf")
    filename_parallel = os.path.join(tempdir, "mesh_parallel.xdmf")

    if comm_world.rank == 0:
        mesh = UnitCubeMesh(comm_self, 2, 2, 2)
        mesh.name = "mesh"
        num_cells = mesh.topology.index_map(mesh.topology.dim).size_local
        indices = np.arange(num_cells, dtype=np.int32)
        values = np.ones(num_cells, dtype=np.int32)
        mt = MeshTags(mesh, mesh.topology.dim, indices, values)
        mt.name = "cells"
        with XDMFFile(comm_self, filename_serial, "w") as xdmf:
            xdmf.write_mesh(mesh)
            xdmf.write_meshtags(mt)
    comm_world.barrier()

    with XDMFFile(comm_world, filename_serial, "r") as xdmf:
        mesh2 = xdmf.read_mesh(name="mesh")
        mt2 = xdmf.read_meshtags(mesh2, name="cells")

    with XDMFFile(comm_world, filename_parallel, "w") as xdmf:
        xdmf.write_mesh(mesh2)
        xdmf.write_meshtags(mt2)

    parser = ElementTree.XMLParser()
    tree = ElementTree.parse(filename_parallel, parser)
    root = tree.getroot()
    domain = list(root)[0]
    meshes = list(domain)
    num_cells = []
    for mesh in meshes:
        elements = list(mesh)
        for element in elements:
            if element.tag == "Topology":
                num_cells.append(element.get("NumberOfElements"))
    assert(num_cells[0] == num_cells[1])
