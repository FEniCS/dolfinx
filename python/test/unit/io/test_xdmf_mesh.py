# Copyright (C) 2012-2020 Garth N. Wells and Jørgen S. Dokken
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from pathlib import Path

from mpi4py import MPI

import numpy as np
import pytest

from dolfinx import cpp as _cpp
from dolfinx import default_real_type
from dolfinx.io import XDMFFile
from dolfinx.io.gmshio import cell_perm_array, ufl_mesh
from dolfinx.mesh import (
    CellType,
    GhostMode,
    create_mesh,
    create_submesh,
    create_unit_cube,
    create_unit_interval,
    create_unit_square,
    locate_entities,
)

# Supported XDMF file encoding
if MPI.COMM_WORLD.size > 1:
    encodings = [XDMFFile.Encoding.HDF5]
else:
    encodings = [XDMFFile.Encoding.HDF5, XDMFFile.Encoding.ASCII]

celltypes_2D = [CellType.triangle, CellType.quadrilateral]
celltypes_3D = [CellType.tetrahedron, CellType.hexahedron]


def mesh_factory(tdim, n, ghost_mode=GhostMode.shared_facet, dtype=default_real_type):
    if tdim == 1:
        return create_unit_interval(MPI.COMM_WORLD, n, ghost_mode=ghost_mode, dtype=dtype)
    elif tdim == 2:
        return create_unit_square(MPI.COMM_WORLD, n, n, ghost_mode=ghost_mode, dtype=dtype)
    elif tdim == 3:
        return create_unit_cube(MPI.COMM_WORLD, n, n, n, ghost_mode=ghost_mode, dtype=dtype)


@pytest.mark.skipif(default_real_type != np.float64, reason="float32 not supported yet")
@pytest.mark.parametrize("encoding", encodings)
def test_save_and_load_1d_mesh(tempdir, encoding):
    filename = Path(tempdir, "mesh.xdmf")
    mesh = create_unit_interval(MPI.COMM_WORLD, 32)
    with XDMFFile(mesh.comm, filename, "w", encoding=encoding) as file:
        file.write_mesh(mesh)
    with XDMFFile(MPI.COMM_WORLD, filename, "r", encoding=encoding) as file:
        mesh2 = file.read_mesh()
    assert mesh.topology.index_map(0).size_global == mesh2.topology.index_map(0).size_global
    assert (
        mesh.topology.index_map(mesh.topology.dim).size_global
        == mesh2.topology.index_map(mesh.topology.dim).size_global
    )


@pytest.mark.skipif(default_real_type != np.float64, reason="float32 not supported yet")
@pytest.mark.parametrize("cell_type", celltypes_2D)
@pytest.mark.parametrize("encoding", encodings)
def test_save_and_load_2d_mesh(tempdir, encoding, cell_type):
    filename = Path(tempdir, "mesh.xdmf")
    mesh = create_unit_square(MPI.COMM_WORLD, 12, 12, cell_type)
    mesh.name = "square"

    with XDMFFile(mesh.comm, filename, "w", encoding=encoding) as file:
        file.write_mesh(mesh)

    with XDMFFile(MPI.COMM_WORLD, filename, "r", encoding=encoding) as file:
        mesh2 = file.read_mesh(name="square")

    assert mesh2.name == mesh.name
    topology, topology2 = mesh.topology, mesh2.topology
    assert topology.index_map(0).size_global == topology2.index_map(0).size_global
    assert (
        topology.index_map(topology.dim).size_global
        == topology2.index_map(topology.dim).size_global
    )


@pytest.mark.skipif(default_real_type != np.float64, reason="float32 not supported yet")
@pytest.mark.parametrize("cell_type", celltypes_3D)
@pytest.mark.parametrize("encoding", encodings)
def test_save_and_load_3d_mesh(tempdir, encoding, cell_type):
    filename = Path(tempdir, "mesh.xdmf")
    mesh = create_unit_cube(MPI.COMM_WORLD, 12, 12, 8, cell_type)
    with XDMFFile(mesh.comm, filename, "w", encoding=encoding) as file:
        file.write_mesh(mesh)

    with XDMFFile(MPI.COMM_WORLD, filename, "r", encoding=encoding) as file:
        mesh2 = file.read_mesh()

    topology, topology2 = mesh.topology, mesh2.topology
    assert topology.index_map(0).size_global == topology2.index_map(0).size_global
    assert (
        topology.index_map(topology.dim).size_global
        == topology2.index_map(topology.dim).size_global
    )


@pytest.mark.skipif(default_real_type != np.float64, reason="float32 not supported yet")
@pytest.mark.parametrize("encoding", encodings)
def test_read_write_p2_mesh(tempdir, encoding):
    try:
        import gmsh
    except ImportError:
        pytest.skip()
    if MPI.COMM_WORLD.rank == 0:
        gmsh.initialize()
        model = gmsh.model()
        model.add("test_read_write_p2_mesh")
        model.setCurrent("test_read_write_p2_mesh")

        model.occ.addSphere(0, 0, 0, 1, tag=1)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.3)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.4)
        model.occ.synchronize()
        model.mesh.generate(3)
        model.mesh.setOrder(2)

        idx, points, _ = model.mesh.getNodes()
        points = points.reshape(-1, 3)
        idx -= 1
        srt = np.argsort(idx)
        assert np.all(idx[srt] == np.arange(len(idx)))
        x = points[srt]

        element_types, element_tags, node_tags = model.mesh.getElements(dim=3)
        (
            name,
            dim,
            order,
            num_nodes,
            local_coords,
            num_first_order_nodes,
        ) = model.mesh.getElementProperties(element_types[0])
        cells = node_tags[0].reshape(-1, num_nodes) - 1
        num_nodes, gmsh_cell_id = MPI.COMM_WORLD.bcast(
            [cells.shape[1], model.mesh.getElementType("tetrahedron", 2)], root=0
        )
        gmsh.finalize()
    else:
        num_nodes, gmsh_cell_id = MPI.COMM_WORLD.bcast([None, None], root=0)
        cells, x = np.empty([0, num_nodes]), np.empty([0, 3])

    domain = ufl_mesh(gmsh_cell_id, 3, dtype=default_real_type)
    cell_type = _cpp.mesh.to_type(str(domain.ufl_cell()))
    cells = cells[:, cell_perm_array(cell_type, cells.shape[1])].copy()

    mesh = create_mesh(MPI.COMM_WORLD, cells, x, domain)

    filename = Path(tempdir, "tet10_mesh.xdmf")
    with XDMFFile(mesh.comm, filename, "w", encoding=encoding) as xdmf:
        xdmf.write_mesh(mesh)
    with XDMFFile(mesh.comm, filename, "r", encoding=encoding) as xdmf:
        mesh2 = xdmf.read_mesh()

    topology, topology2 = mesh.topology, mesh2.topology
    assert topology.index_map(0).size_global == topology2.index_map(0).size_global
    assert (
        topology.index_map(topology.dim).size_global
        == topology2.index_map(topology.dim).size_global
    )


@pytest.mark.parametrize("d", [2, 3])
@pytest.mark.parametrize("n", [2, 5])
@pytest.mark.parametrize("codim", [0, 1])
@pytest.mark.parametrize("ghost_mode", [GhostMode.none, GhostMode.shared_facet])
@pytest.mark.parametrize("encoding", encodings)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_submesh(tempdir, d, n, codim, ghost_mode, encoding, dtype):
    mesh = mesh_factory(d, n, ghost_mode, dtype=dtype)
    edim = d - codim
    entities = locate_entities(mesh, edim, lambda x: x[0] > 0.4999)
    submesh = create_submesh(mesh, edim, entities)[0]

    # Check writing the mesh doesn't cause a segmentation fault
    filename = Path(tempdir, "submesh.xdmf")
    with XDMFFile(mesh.comm, filename, "w", encoding=encoding) as xdmf:
        xdmf.write_mesh(submesh)
