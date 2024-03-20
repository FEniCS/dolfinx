# Copyright (C) 2021 Chris Richardson and Igor Baratta
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for the KrylovSolver interface"""

from mpi4py import MPI

import numpy as np
import pytest

from basix.ufl import element
from dolfinx import la
from dolfinx.fem import Function, functionspace
from dolfinx.mesh import create_unit_square


@pytest.mark.parametrize(
    "e", [element("Lagrange", "triangle", 1), element("Lagrange", "triangle", 1, shape=(2,))]
)
def test_scatter_forward(e):
    mesh = create_unit_square(MPI.COMM_WORLD, 5, 5)
    V = functionspace(mesh, e)
    u = Function(V)
    bs = V.dofmap.bs

    u.interpolate(lambda x: [x[i] for i in range(bs)])

    # Forward scatter should have no effect
    w0 = u.x.array.copy()
    u.x.scatter_forward()
    assert np.allclose(w0, u.x.array)

    # Fill local array with the mpi rank
    u.x.array.fill(MPI.COMM_WORLD.rank)
    w0 = u.x.array.copy()
    u.x.scatter_forward()

    # Now the ghosts should have the value of the rank of the owning
    # process
    ghost_owners = u.function_space.dofmap.index_map.owners
    ghost_owners = np.repeat(ghost_owners, bs)
    local_size = u.function_space.dofmap.index_map.size_local * bs
    assert np.allclose(u.x.array[local_size:], ghost_owners)


@pytest.mark.parametrize(
    "e", [element("Lagrange", "triangle", 1), element("Lagrange", "triangle", 1, shape=(2,))]
)
def test_scatter_reverse(e):
    comm = MPI.COMM_WORLD
    mesh = create_unit_square(MPI.COMM_WORLD, 5, 5)
    V = functionspace(mesh, e)
    u = Function(V)
    bs = V.dofmap.bs

    u.interpolate(lambda x: [x[i] for i in range(bs)])

    # Reverse scatter (insert) should have no effect
    w0 = u.x.array.copy()
    u.x.scatter_reverse(la.InsertMode.insert)
    assert np.allclose(w0, u.x.array)

    # Fill with MPI rank, and sum all entries in the vector (including
    # ghosts)
    u.x.array.fill(comm.rank)
    all_count0 = MPI.COMM_WORLD.allreduce(u.x.array.sum(), op=MPI.SUM)

    # Reverse scatter (add)
    u.x.scatter_reverse(la.InsertMode.add)
    num_ghosts = V.dofmap.index_map.num_ghosts
    ghost_count = MPI.COMM_WORLD.allreduce(num_ghosts * comm.rank, op=MPI.SUM)

    # New count should have gone up by the number of ghosts times their
    # rank on all processes
    all_count1 = MPI.COMM_WORLD.allreduce(u.x.array.sum(), op=MPI.SUM)
    assert all_count1 == (all_count0 + bs * ghost_count)


@pytest.mark.parametrize(
    "dtype",
    [
        np.float32,
        np.float64,
        np.complex64,
        np.complex128,
        np.int8,
        np.int32,
        np.uint32,
        np.int64,
        np.uint64,
    ],
)
def test_vector_from_index_map_scatter_forward(dtype):
    comm = MPI.COMM_WORLD
    mesh = create_unit_square(comm, 5, 5)

    for d in range(mesh.topology.dim):
        mesh.topology.create_entities(d)
        im = mesh.topology.index_map(d)
        vector = la.vector(im, dtype=dtype)
        vector.array[: im.size_local] = np.arange(*im.local_range, dtype=dtype)
        vector.scatter_forward()
        global_idxs = im.local_to_global(np.arange(im.size_local + im.num_ghosts, dtype=np.int32))
        global_idxs = np.asarray(global_idxs, dtype)

        assert np.all(vector.array == global_idxs)
