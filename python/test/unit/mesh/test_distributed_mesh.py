# Copyright (C) 2019 Igor A. Baratta
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import os

from dolfin import MPI, UnitSquareMesh
from dolfin.cpp.mesh import GhostMode
from dolfin.io import XDMFFile
from dolfin_utils.test.fixtures import tempdir


assert (tempdir)


def xtest_distributed_mesh_2d(tempdir):
    """Read and partition mesh using only a subset of the
       avaialable processes"""

    comm = MPI.comm_world
    encoding = XDMFFile.Encoding.HDF5
    filename = os.path.join(tempdir, "mesh2d.xdmf")

    # Create mesh, and partition using all available processes
    mesh0 = UnitSquareMesh(comm, 4, 4)
    dim = mesh0.topology.dim
    assert mesh0.num_entities_global(0) == 25
    assert mesh0.num_entities_global(dim) == 32

    # Save mesh using XDMFFile
    with XDMFFile(comm, filename, encoding) as xdmf:
        xdmf.write(mesh0)

    # Use all available processes for reading and partitioning the mesh
    with XDMFFile(MPI.comm_world, filename) as xdmf:
        mesh1 = xdmf.read_mesh(MPI.comm_world, GhostMode.none, 1.0)

    assert mesh1.num_entities_global(0) == mesh0.num_entities_global(0)
    assert mesh1.num_entities_global(dim) == mesh0.num_entities_global(dim)
    assert MPI.max(comm, mesh1.hmax()) == MPI.max(comm, mesh0.hmax())
    assert MPI.min(comm, mesh1.hmin()) == MPI.min(comm, mesh0.hmin())

    # Use only half of the available proocesses for reading and partitioning
    # the mesh. And then redistribute the respective local meshes to all
    # processes.
    with XDMFFile(MPI.comm_world, filename) as xdmf:
        mesh2 = xdmf.read_mesh(MPI.comm_world, GhostMode.none, 0.5)

    assert mesh2.num_entities_global(0) == mesh0.num_entities_global(0)
    assert mesh2.num_entities_global(dim) == mesh0.num_entities_global(dim)
    assert MPI.max(comm, mesh2.hmax()) == MPI.max(comm, mesh0.hmax())
    assert MPI.min(comm, mesh2.hmin()) == MPI.min(comm, mesh0.hmin())
