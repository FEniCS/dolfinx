#!/usr/bin/env py.test

"""Unit tests for PeriodicBoundaryComputation"""

# Copyright (C) 2013 Mikael Mortensen
#
# This file is part of DOLFIN.
#
# DOLFIN is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DOLFIN is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
#
# First added:  2013-04-12
# Last changed: 2014-05-30

import pytest
import numpy as np
from dolfin import *

class PeriodicBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] < DOLFIN_EPS
    def map(self, x, y):
        y[0] = x[0] - 1.0
        y[1] = x[1]

# Create instance of PeriodicBoundaryComputation
periodic_boundary = PeriodicBoundary()
pbc = PeriodicBoundaryComputation()
mesh = UnitSquareMesh(4, 4)

@pytest.mark.skipif(MPI.size(mpi_comm_world()) > 1, 
            reason="Skipping unit test(s) not working in parallel")
class TestPeriodicBoundaryComputations:

    def test_ComputePeriodicPairs(self):

        # Verify that correct number of periodic pairs are computed
        vertices = pbc.compute_periodic_pairs(mesh, periodic_boundary, 0)
        edges    = pbc.compute_periodic_pairs(mesh, periodic_boundary, 1)
        assert len(vertices) == 5
        assert len(edges) == 4

    def test_MastersSlaves(self):

        # Verify that correct number of masters and slaves are marked
        mf = pbc.masters_slaves(mesh, periodic_boundary, 0)
        assert len(np.where(mf.array() == 1)[0]) == 5
        assert len(np.where(mf.array() == 2)[0]) == 5

        mf = pbc.masters_slaves(mesh, periodic_boundary, 1)
        assert len(np.where(mf.array() == 1)[0]) == 4
        assert len(np.where(mf.array() == 2)[0]) == 4
