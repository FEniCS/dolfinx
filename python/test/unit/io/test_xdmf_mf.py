# Copyright (C) 2012-2019 Garth N. Wells
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import os

import numpy as np
import pytest

from dolfinx import (MPI, MeshFunction, UnitCubeMesh, UnitIntervalMesh,
                     UnitSquareMesh, cpp)
from dolfinx.cpp.mesh import CellType
from dolfinx.io import XDMFFileNew
from dolfinx_utils.test.fixtures import tempdir

assert (tempdir)

# Supported XDMF file encoding
if MPI.size(MPI.comm_world) > 1:
    encodings = (XDMFFileNew.Encoding.HDF5, )
else:
    encodings = (XDMFFileNew.Encoding.HDF5, XDMFFileNew.Encoding.ASCII)
    encodings = (XDMFFileNew.Encoding.HDF5, )

celltypes_2D = [CellType.triangle, CellType.quadrilateral]
celltypes_3D = [CellType.tetrahedron, CellType.hexahedron]


def mesh_factory(tdim, n):
    if tdim == 1:
        return UnitIntervalMesh(MPI.comm_world, n, new_style=True)
    elif tdim == 2:
        return UnitSquareMesh(MPI.comm_world, n, n, new_style=True)
    elif tdim == 3:
        return UnitCubeMesh(MPI.comm_world, n, n, n, new_style=True)


@pytest.fixture
def worker_id(request):
    """Return worker ID when using pytest-xdist to run tests in parallel"""
    if hasattr(request.config, 'slaveinput'):
        return request.config.slaveinput['slaveid']
    else:
        return 'master'


# --- MeshFunctions
# encodings = (XDMFFileNew.Encoding.ASCII,)
# celltypes_2D = [CellType.triangle, CellType.quadrilateral]
# celltypes_2D = [CellType.quadrilateral]
# celltypes_2D = [CellType.quadrilateral]
data_types = (('int', int), )


@pytest.mark.parametrize("cell_type", celltypes_2D)
@pytest.mark.parametrize("encoding", encodings)
@pytest.mark.parametrize("data_type", data_types)
def test_save_2D_cell_function(tempdir, encoding, data_type, cell_type):
    dtype_str, dtype = data_type
    filename = os.path.join(tempdir, "mf_2D_{}.xdmf".format(dtype_str))
    filename_msh = os.path.join(tempdir, "mf_2D_{}-mesh.xdmf".format(dtype_str))
    mesh = UnitSquareMesh(MPI.comm_world, 22, 11, cell_type, new_style=True)

    # print(mesh.topology.connectivity(2,0))
    # print(mesh.geometry.dofmap())
    # print(mesh.geometry.x)
    # return

    mf = MeshFunction(dtype_str, mesh, mesh.topology.dim, 0)
    mf.name = "cells"

    tdim = mesh.topology.dim
    map = mesh.topology.index_map(tdim)
    num_cells = map.size_local + map.num_ghosts
    mf.values[:] = np.arange(num_cells, dtype=dtype) + map.local_range[0]
    x = mesh.geometry.x
    x_dofmap = mesh.geometry.dofmap()
    num_vertices = cpp.mesh.cell_num_vertices(cell_type)
    for c in range(num_cells):
        dofs = x_dofmap.links(c)
        v = [dofs[i] for i in range(num_vertices)]
        x_r = np.linalg.norm(np.sum(x[v], axis=0) / len(v))
        if (x_r < 1):
            mf.values[c] = -1
        else:
            mf.values[c] = 1

    # NOTE: We need to write the mesh and mesh function to handle
    # re-odering of indices
    # Write mesh and mesh function
    with XDMFFileNew(mesh.mpi_comm(), filename_msh, encoding=encoding) as file:
        file.write(mesh)
    with XDMFFileNew(mesh.mpi_comm(), filename, encoding=encoding) as file:
        file.write(mf)

    with XDMFFileNew(mesh.mpi_comm(), filename_msh) as xdmf:
        mesh2 = xdmf.read_mesh()
    with XDMFFileNew(mesh.mpi_comm(), filename) as xdmf:
        read_function = getattr(xdmf, "read_mf_" + dtype_str)
        mf_in = read_function(mesh2, "cells")

    map = mesh2.topology.index_map(tdim)
    num_cells = map.size_local + map.num_ghosts
    x = mesh2.geometry.x
    x_dofmap = mesh2.geometry.dofmap()
    for c in range(num_cells):
        dofs = x_dofmap.links(c)
        v = [dofs[i] for i in range(num_vertices)]
        x_r = np.linalg.norm(np.sum(x[v], axis=0) / len(v))
        if (x_r < 1.0):
            assert mf_in.values[c] == -1
        else:
            assert mf_in.values[c] == 1


@pytest.mark.parametrize("cell_type", celltypes_3D)
@pytest.mark.parametrize("encoding", encodings)
@pytest.mark.parametrize("data_type", data_types)
def test_save_3D_cell_function(tempdir, encoding, data_type, cell_type):
    dtype_str, dtype = data_type
    filename = os.path.join(tempdir, "mf_2D_{}.xdmf".format(dtype_str))
    filename_msh = os.path.join(tempdir, "mf_2D_{}-mesh.xdmf".format(dtype_str))
    mesh = UnitCubeMesh(MPI.comm_world, 7, 11, 5, cell_type, new_style=True)
    mf = MeshFunction(dtype_str, mesh, mesh.topology.dim, 0)
    mf.name = "cells"

    tdim = mesh.topology.dim
    map = mesh.topology.index_map(tdim)
    num_cells = map.size_local + map.num_ghosts
    mf.values[:] = np.arange(num_cells, dtype=dtype) + map.local_range[0]
    x = mesh.geometry.x
    x_dofmap = mesh.geometry.dofmap()
    num_vertices = cpp.mesh.cell_num_vertices(cell_type)
    for c in range(num_cells):
        dofs = x_dofmap.links(c)
        v = [dofs[i] for i in range(num_vertices)]
        x_r = np.linalg.norm(np.sum(x[v], axis=0) / len(v))
        if (x_r < 1):
            mf.values[c] = -1
        else:
            mf.values[c] = 1

    # NOTE: We need to write the mesh and mesh function to handle
    # re-odering of indices
    # Write mesh and mesh function
    with XDMFFileNew(mesh.mpi_comm(), filename_msh, encoding=encoding) as file:
        file.write(mesh)
    with XDMFFileNew(mesh.mpi_comm(), filename, encoding=encoding) as file:
        file.write(mf)

    with XDMFFileNew(mesh.mpi_comm(), filename_msh) as xdmf:
        mesh2 = xdmf.read_mesh()
    with XDMFFileNew(mesh.mpi_comm(), filename) as xdmf:
        read_function = getattr(xdmf, "read_mf_" + dtype_str)
        mf_in = read_function(mesh2, "cells")

    map = mesh2.topology.index_map(tdim)
    num_cells = map.size_local + map.num_ghosts
    x = mesh2.geometry.x
    x_dofmap = mesh2.geometry.dofmap()
    for c in range(num_cells):
        dofs = x_dofmap.links(c)
        v = [dofs[i] for i in range(num_vertices)]
        x_r = np.linalg.norm(np.sum(x[v], axis=0) / len(v))
        if (x_r < 1.0):
            assert mf_in.values[c] == -1
        else:
            assert mf_in.values[c] == 1


# encodings = (XDMFFileNew.Encoding.ASCII,)
# celltypes_2D = [CellType.quadrilateral]


@pytest.mark.parametrize("cell_type", celltypes_2D)
@pytest.mark.parametrize("encoding", encodings)
@pytest.mark.parametrize("data_type", data_types)
def test_save_2D_facet_function(tempdir, encoding, data_type, cell_type):
    dtype_str, dtype = data_type
    mesh = UnitSquareMesh(MPI.comm_world, 7, 8, cell_type, new_style=True)
    tdim = mesh.topology.dim
    mf = MeshFunction(dtype_str, mesh, tdim - 1, 0)
    mf.name = "facets"

    # TODO: Add test that is robust with respect to number, i.e.
    # computing something based on coordinate
    map = mesh.topology.index_map(tdim - 1)
    global_indices = map.global_indices(True)
    mf.values[:] = global_indices[:]

    # filename = os.path.join(tempdir, "mf_facet_2D_%s.xdmf" % dtype_str)
    # filename_msh = os.path.join(tempdir, "mf_facet_2D_%s-mesh.xdmf" % dtype_str)
    filename = os.path.join("mf_facet_2D_%s.xdmf" % dtype_str)
    filename_msh = os.path.join("mf_facet_2D_%s-mesh.xdmf" % dtype_str)
    mesh.create_connectivity(tdim - 1, tdim)

    # NOTE: We need to write the mesh and mesh function to handle
    # re-odering of indices
    # Write mesh and mesh function
    with XDMFFileNew(mesh.mpi_comm(), filename_msh, encoding=encoding) as file:
        file.write(mesh)
    with XDMFFileNew(mesh.mpi_comm(), filename, encoding=encoding) as xdmf:
        xdmf.write(mf)

    # with XDMFFileNew(mesh.mpi_comm(), filename_msh, encoding=encoding) as xdmf:
    #     mesh2 = xdmf.read_mesh()

    # print(mesh2.topology.connectivity(2,0))
    # print(mesh2.geometry.dofmap())
    # print(mesh2.geometry.x)
    # return

    # mesh2.create_connectivity(tdim - 1, tdim)
    # with XDMFFileNew(mesh.mpi_comm(), filename, encoding=encoding) as xdmf:
    #     read_function = getattr(xdmf, "read_mf_" + dtype_str)
    #     mf_in = read_function(mesh2, "facets")

    # print(mf_in.values)
    # diff = mf_in.values - mf.values
    # assert np.all(diff == 0)
