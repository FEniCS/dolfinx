# Copyright (C) 2021 Chris Richardson and Igor Baratta
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for the KrylovSolver interface"""


import numpy as np
import pytest

import ufl
from dolfinx import cpp as _cpp
from dolfinx.fem import Function, FunctionSpace
from dolfinx.mesh import create_unit_square

from mpi4py import MPI


@pytest.mark.parametrize("element", [ufl.FiniteElement("Lagrange", "triangle", 1),
                                     ufl.VectorElement("Lagrange", "triangle", 1)])
def test_scatter_forward(element):

    mesh = create_unit_square(MPI.COMM_WORLD, 5, 5)
    V = FunctionSpace(mesh, element)
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


@pytest.mark.parametrize("element", [ufl.FiniteElement("Lagrange", "triangle", 1),
                                     ufl.VectorElement("Lagrange", "triangle", 1)])
def test_scatter_reverse(element):

    comm = MPI.COMM_WORLD
    mesh = create_unit_square(MPI.COMM_WORLD, 5, 5)
    V = FunctionSpace(mesh, element)
    u = Function(V)
    bs = V.dofmap.bs

    u.interpolate(lambda x: [x[i] for i in range(bs)])

    # Reverse scatter (insert) should have no effect
    w0 = u.x.array.copy()
    u.x.scatter_reverse(_cpp.la.ScatterMode.insert)
    assert np.allclose(w0, u.x.array)

    # Fill with MPI rank, and sum all entries in the vector (including
    # ghosts)
    u.x.array.fill(comm.rank)
    all_count0 = MPI.COMM_WORLD.allreduce(u.x.array.sum(), op=MPI.SUM)

    # Reverse scatter (add)
    u.x.scatter_reverse(_cpp.la.ScatterMode.add)
    num_ghosts = V.dofmap.index_map.num_ghosts
    ghost_count = MPI.COMM_WORLD.allreduce(num_ghosts * comm.rank, op=MPI.SUM)

    # New count should have gone up by the number of ghosts times their
    # rank on all processes
    all_count1 = MPI.COMM_WORLD.allreduce(u.x.array.sum(), op=MPI.SUM)
    assert all_count1 == (all_count0 + bs * ghost_count)
