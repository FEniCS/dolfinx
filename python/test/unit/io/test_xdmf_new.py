# Copyright (C) 2012-2019 Garth N. Wells
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import os

import pytest
from dolfinx_utils.test.fixtures import tempdir

from dolfinx import UnitSquareMesh, MPI
from dolfinx.cpp.mesh import CellType
from dolfinx.io import XDMFFileNew, XDMFFile

assert (tempdir)


@pytest.fixture
def worker_id(request):
    """Return worker ID when using pytest-xdist to run tests in parallel"""
    if hasattr(request.config, 'slaveinput'):
        return request.config.slaveinput['slaveid']
    else:
        return 'master'


def test_save_and_load_mesh(tempdir):
    # filename = os.path.join(tempdir, "mesh.xdmf")
    filename = os.path.join("mesh.xdmf")
    cell_type = CellType.triangle

    mesh = UnitSquareMesh(MPI.comm_world, 2, 2, cell_type, new_style=True)
    encoding = XDMFFileNew.Encoding.HDF5
    with XDMFFileNew(mesh.mpi_comm(), filename, encoding=encoding) as file:
        file.write(mesh)

    with XDMFFileNew(MPI.comm_world, filename) as file:
        mesh2 = file.read_mesh()
    assert mesh.topology.index_map(0).size_global == mesh2.topology.index_map(
        0).size_global
    dim = mesh.topology.dim
    assert mesh.topology.index_map(
        dim).size_global == mesh2.topology.index_map(dim).size_global
