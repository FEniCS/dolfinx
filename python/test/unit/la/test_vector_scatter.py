# -*- coding: utf-8 -*-
# Copyright (C) 2020 Chris Richardson
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for the KrylovSolver interface"""


import numpy as np

from dolfinx import (Function, FunctionSpace, UnitSquareMesh, cpp)
from mpi4py import MPI


def test_scatter_forward():

    mesh = UnitSquareMesh(MPI.COMM_WORLD, 3, 3)
    V = FunctionSpace(mesh, ("Lagrange", 1))
    u = Function(V)
    u.interpolate(lambda x: (x[0]))

    # Forward scatter should have no effect
    w0 = np.array(u.x.array)
    cpp.la.scatter_forward(u.x)
    w1 = np.array(u.x.array)
    assert np.allclose(w0, w1)


def test_scatter_reverse():

    mesh = UnitSquareMesh(MPI.COMM_WORLD, 3, 3)
    V = FunctionSpace(mesh, ("Lagrange", 1))
    u = Function(V)

    u.interpolate(lambda x: (x[0]))

    # Reverse scatter (insert) should have no effect
    w0 = np.array(u.x.array)
    cpp.la.scatter_reverse(u.x, cpp.common.ScatterMode.insert)
    w1 = np.array(u.x.array)
    assert np.allclose(w0, w1)

    # Fill with ones, and count all entries on all processes
    u.interpolate(lambda x: (np.ones(x.shape[1])))
    all_count0 = MPI.COMM_WORLD.allreduce(sum(u.x.array), op=MPI.SUM)

    # Reverse scatter (add)
    cpp.la.scatter_reverse(u.x, cpp.common.ScatterMode.add)
    ghost_count = MPI.COMM_WORLD.allreduce(V.dofmap.index_map.num_ghosts, op=MPI.SUM)
    # New count should have gone up by the number of ghosts on all processes
    all_count1 = MPI.COMM_WORLD.allreduce(sum(u.x.array), op=MPI.SUM)
    assert all_count1 == (all_count0 + ghost_count)
