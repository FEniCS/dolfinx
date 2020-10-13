# Copyright (C) 2012-2020 Garth N. Wells and Jørgen S. Dokken
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import os

import numpy as np
import pytest
from dolfinx import UnitCubeMesh, UnitIntervalMesh, UnitSquareMesh, cpp
from dolfinx.cpp.io import perm_gmsh
from dolfinx.cpp.mesh import CellType
from dolfinx.io import XDMFFile, ufl_mesh_from_gmsh
from dolfinx.mesh import create_mesh
from dolfinx_utils.test.fixtures import tempdir
from mpi4py import MPI

assert (tempdir)

# Supported XDMF file encoding
if MPI.COMM_WORLD.size > 1:
    encodings = (XDMFFile.Encoding.HDF5, )
else:
    encodings = (XDMFFile.Encoding.HDF5, XDMFFile.Encoding.ASCII)

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


@pytest.mark.parametrize("encoding", encodings)
def test_save_and_load_1d_mesh(tempdir, encoding):
    filename = os.path.join(tempdir, "mesh.xdmf")
    mesh = UnitIntervalMesh(MPI.COMM_WORLD, 32)
    with XDMFFile(mesh.mpi_comm(), filename, "w", encoding=encoding) as file:
        file.write_mesh(mesh)
    with XDMFFile(MPI.COMM_WORLD, filename, "r", encoding=encoding) as file:
        mesh2 = file.read_mesh()
    assert mesh.topology.index_map(0).size_global == mesh2.topology.index_map(0).size_global
    assert mesh.topology.index_map(mesh.topology.dim).size_global == mesh2.topology.index_map(
        mesh.topology.dim).size_global


@pytest.mark.parametrize("cell_type", celltypes_2D)
@pytest.mark.parametrize("encoding", encodings)
def test_save_and_load_2d_mesh(tempdir, encoding, cell_type):
    filename = os.path.join(tempdir, "mesh.xdmf")
    mesh = UnitSquareMesh(MPI.COMM_WORLD, 12, 12, cell_type)
    mesh.name = "square"

    with XDMFFile(mesh.mpi_comm(), filename, "w", encoding=encoding) as file:
        file.write_mesh(mesh)

    with XDMFFile(MPI.COMM_WORLD, filename, "r", encoding=encoding) as file:
        mesh2 = file.read_mesh(name="square")

    assert mesh2.name == mesh.name
    assert mesh.topology.index_map(0).size_global == mesh2.topology.index_map(0).size_global
    assert mesh.topology.index_map(mesh.topology.dim).size_global == mesh2.topology.index_map(
        mesh.topology.dim).size_global


@pytest.mark.parametrize("cell_type", celltypes_3D)
@pytest.mark.parametrize("encoding", encodings)
def test_save_and_load_3d_mesh(tempdir, encoding, cell_type):
    filename = os.path.join(tempdir, "mesh.xdmf")
    mesh = UnitCubeMesh(MPI.COMM_WORLD, 12, 12, 8, cell_type)
    with XDMFFile(mesh.mpi_comm(), filename, "w", encoding=encoding) as file:
        file.write_mesh(mesh)

    with XDMFFile(MPI.COMM_WORLD, filename, "r", encoding=encoding) as file:
        mesh2 = file.read_mesh()

    assert mesh.topology.index_map(0).size_global == mesh2.topology.index_map(0).size_global
    assert mesh.topology.index_map(mesh.topology.dim).size_global == mesh2.topology.index_map(
        mesh.topology.dim).size_global


@pytest.mark.parametrize("encoding", encodings)
def test_read_write_p2_mesh(tempdir, encoding):
    if MPI.COMM_WORLD.rank == 0:
        gmsh = pytest.importorskip("gmsh")
        gmsh.initialize()
        gmsh.model.occ.addSphere(0, 0, 0, 1, tag=1)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.3)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.4)
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.setOrder(2)

        idx, points, _ = gmsh.model.mesh.getNodes()
        points = points.reshape(-1, 3)
        idx -= 1
        srt = np.argsort(idx)
        assert np.all(idx[srt] == np.arange(len(idx)))
        x = points[srt]

        element_types, element_tags, node_tags = gmsh.model.mesh.getElements(dim=3)
        name, dim, order, num_nodes, local_coords, num_first_order_nodes = gmsh.model.mesh.getElementProperties(
            element_types[0])
        cells = node_tags[0].reshape(-1, num_nodes) - 1
        num_nodes, gmsh_cell_id = MPI.COMM_WORLD.bcast(
            [cells.shape[1], gmsh.model.mesh.getElementType("tetrahedron", 2)], root=0)
        gmsh.finalize()

    else:
        num_nodes, gmsh_cell_id = MPI.COMM_WORLD.bcast([None, None], root=0)
        cells, x = np.empty([0, num_nodes]), np.empty([0, 3])

    domain = ufl_mesh_from_gmsh(gmsh_cell_id, 3)
    cell_type = cpp.mesh.to_type(str(domain.ufl_cell()))
    cells = cells[:, perm_gmsh(cell_type, cells.shape[1])]

    mesh = create_mesh(MPI.COMM_WORLD, cells, x, domain)

    filename = os.path.join(tempdir, "tet10_mesh.xdmf")
    with XDMFFile(mesh.mpi_comm(), filename, "w", encoding=encoding) as xdmf:
        xdmf.write_mesh(mesh)
    with XDMFFile(mesh.mpi_comm(), filename, "r", encoding=encoding) as xdmf:
        mesh2 = xdmf.read_mesh()

    assert mesh.topology.index_map(0).size_global == mesh2.topology.index_map(0).size_global
    assert mesh.topology.index_map(mesh.topology.dim).size_global == mesh2.topology.index_map(
        mesh.topology.dim).size_global
