# Copyright (C) 2012-2020 Garth N. Wells
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import os

import numpy as np
import pytest

from dolfinx import (MPI, MeshFunction, MeshValueCollection, UnitCubeMesh,
                     UnitIntervalMesh, UnitSquareMesh, cpp)
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


data_types = (('int', int), )
celltypes_2D = [CellType.triangle]
# encodings = (XDMFFileNew.Encoding.ASCII, )

# @pytest.mark.parametrize("cell_type", celltypes_3D)
@pytest.mark.parametrize("cell_type", celltypes_2D)
@pytest.mark.parametrize("encoding", encodings)
@pytest.mark.parametrize("data_type", data_types)
def test_save_mesh_value_collection(tempdir, encoding, data_type, cell_type):
    dtype_str, dtype = data_type
    mesh = UnitSquareMesh(MPI.comm_world, 4, 4, cell_type, new_style=True)
    tdim = mesh.topology.dim
    meshfn = MeshFunction(dtype_str, mesh, mesh.topology.dim, False)
    meshfn.name = "volume_marker"
    mp = cpp.mesh.midpoints(mesh, tdim, range(mesh.num_entities(tdim)))
    for i in range(mesh.num_cells()):
        if mp[i, 1] > 0.1:
            meshfn.values[i] = 1
        if mp[i, 1] > 0.9:
            meshfn.values[i] = 2

    for mvc_dim in range(0, tdim + 1):
        mvc = MeshValueCollection(dtype_str, mesh, mvc_dim)
        tag = "dim_{}_marker".format(mvc_dim)
        mvc.name = tag
        mesh.create_connectivity(mvc_dim, tdim)
        mp = cpp.mesh.midpoints(mesh, mvc_dim, range(mesh.num_entities(mvc_dim)))
        for e in range(mesh.num_entities(mvc_dim)):
            if (mp[e, 0] > 0.5):
                mvc.set_value(e, dtype(1))

        # filename = os.path.join(tempdir, "mvc_{}.xdmf".format(mvc_dim))
        filename = os.path.join("tmp-{}.xdmf".format(mvc_dim))

        with XDMFFileNew(mesh.mpi_comm(), filename, encoding=encoding) as xdmf:
            xdmf.write(meshfn)
            xdmf.write(mvc)

        with XDMFFileNew(mesh.mpi_comm(), filename) as xdmf:
            read_function = getattr(xdmf, "read_mvc_" + dtype_str)
            mvc = read_function(mesh, tag)
        # print(mvc.values())


celltypes_3D = [CellType.tetrahedron]


@pytest.mark.parametrize("cell_type", celltypes_3D)
@pytest.mark.parametrize("encoding", encodings)
@pytest.mark.parametrize("data_type", data_types)
def test_append_and_load_mesh_value_collections(tempdir, encoding, data_type, cell_type):
    dtype_str, dtype = data_type
    mesh = UnitCubeMesh(MPI.comm_world, 1, 1, 1, cell_type, new_style=True)
    mesh.create_connectivity_all()

    mvc_v = MeshValueCollection(dtype_str, mesh, 0)
    mvc_v.name = "vertices"
    mvc_e = MeshValueCollection(dtype_str, mesh, 1)
    mvc_e.name = "edges"
    mvc_f = MeshValueCollection(dtype_str, mesh, 2)
    mvc_f.name = "facets"
    mvc_c = MeshValueCollection(dtype_str, mesh, 3)
    mvc_c.name = "cells"

    mvcs = [mvc_v, mvc_e, mvc_f, mvc_c]

    filename = os.path.join(tempdir, "appended_mvcs.xdmf")

    with XDMFFileNew(mesh.mpi_comm(), filename) as xdmf:
        for mvc in mvcs:
            # global_indices = mesh.topology.global_indices(mvc.dim)
            map = mesh.topology.index_map(mvc.dim)
            global_indices = map.global_indices(True)

            for ent in range(mesh.num_entities(mvc.dim)):
                assert (mvc.set_value(ent, global_indices[ent]))
            xdmf.write(mvc)

    mvc_v_in = MeshValueCollection(dtype_str, mesh, 0)
    mvc_e_in = MeshValueCollection(dtype_str, mesh, 1)
    mvc_f_in = MeshValueCollection(dtype_str, mesh, 2)
    mvc_c_in = MeshValueCollection(dtype_str, mesh, 3)

    with XDMFFileNew(mesh.mpi_comm(), filename) as xdmf:
        read_function = getattr(xdmf, "read_mvc_" + dtype_str)
        mvc_v_in = read_function(mesh, "vertices")
        mvc_e_in = read_function(mesh, "edges")
        mvc_f_in = read_function(mesh, "facets")
        mvc_c_in = read_function(mesh, "cells")

    mvcs_in = [mvc_v_in, mvc_e_in, mvc_f_in, mvc_c_in]

    for (mvc, mvc_in) in zip(mvcs, mvcs_in):
        mf = MeshFunction(dtype_str, mesh, mvc, 0)
        mf_in = MeshFunction(dtype_str, mesh, mvc_in, 0)

        diff = mf_in.values - mf.values
        print(np.all(diff == 0))
        # assert np.all(diff == 0)
