# Copyright (C) 2013 Mikael Mortensen
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import numpy as np

from dolfin import MPI, PeriodicBoundaryComputation, SubDomain, UnitSquareMesh
from dolfin_utils.test.fixtures import fixture
from dolfin_utils.test.skips import skip_in_parallel


@fixture
def periodic_boundary():
    class PeriodicBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return x[0] < np.finfo(float).eps

        def map(self, x, y):
            y[0] = x[0] - 1.0
            y[1] = x[1]

    return PeriodicBoundary()


@fixture
def mesh():
    return UnitSquareMesh(MPI.comm_world, 4, 4)


@skip_in_parallel
def test_ComputePeriodicPairs(periodic_boundary, mesh):
    # Verify that correct number of periodic pairs are computed
    vertices = PeriodicBoundaryComputation.compute_periodic_pairs(
        mesh, periodic_boundary, 0)
    edges = PeriodicBoundaryComputation.compute_periodic_pairs(
        mesh, periodic_boundary, 1)
    assert len(vertices) == 5
    assert len(edges) == 4


@skip_in_parallel
def test_MastersSlaves(periodic_boundary, mesh):
    # Verify that correct number of masters and slaves are marked
    mf = PeriodicBoundaryComputation.masters_slaves(mesh, periodic_boundary, 0)
    assert len(np.where(mf.array() == 1)[0]) == 5
    assert len(np.where(mf.array() == 2)[0]) == 5

    mf = PeriodicBoundaryComputation.masters_slaves(mesh, periodic_boundary, 1)
    assert len(np.where(mf.array() == 1)[0]) == 4
    assert len(np.where(mf.array() == 2)[0]) == 4
