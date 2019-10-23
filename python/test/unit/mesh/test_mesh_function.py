# Copyright (C) 2019 Michal Habera
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import numpy
import numpy.random
import pytest

from dolfin import MPI, MeshFunction, UnitCubeMesh
from dolfin.cpp.mesh import GhostMode


dtypes = (
    ('int', numpy.intc),
    ('double', numpy.float)
)


@pytest.fixture
def mesh():
    return UnitCubeMesh(MPI.comm_world, 3, 3, 3)


@pytest.mark.parametrize("dtype", dtypes)
def test_data_types(dtype, mesh):
    dtype_str, dtype = dtype
    mf = MeshFunction(dtype_str, mesh, 0, 0)
    assert isinstance(mf.values[0], dtype)


@pytest.mark.parametrize("dtype", dtypes)
def test_numpy_access(dtype, mesh):
    dtype_str, dtype = dtype
    mf = MeshFunction(dtype_str, mesh, 0, 0)
    values = mf.values
    values[:] = numpy.random.rand(len(values))
    assert numpy.all(values == mf.values)


@pytest.mark.xfail(MPI.size(MPI.comm_world) == 1,
                   reason="Shared ghost modes fail in serial.")
@pytest.mark.parametrize("ghost_mode",
                         [GhostMode.none, GhostMode.shared_vertex, GhostMode.shared_facet])
@pytest.mark.parametrize("dim",
                         [0, pytest.param(1, marks=pytest.mark.skip),
                          pytest.param(2, marks=pytest.mark.skip), 3])
def test_update_ghosts(mesh, ghost_mode, dim):
    comm = MPI.comm_world
    mesh = UnitCubeMesh(comm, 5, 5, 5, ghost_mode=ghost_mode)
    mf = MeshFunction('double', mesh, dim, 0)
    mf.values[:] = comm.rank
    mf.update_ghosts()

    size_owned = mesh.topology.ghost_offset(dim)
    size_local = mesh.topology.size(dim)
    num_ghosts = size_local - size_owned
    entity_owner = mesh.topology.entity_owner(dim)

    assert(num_ghosts == len(entity_owner))
    assert(numpy.all(mf.values[size_owned:] == entity_owner))
