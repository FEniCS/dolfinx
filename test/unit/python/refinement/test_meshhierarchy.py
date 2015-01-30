#!/usr/bin/env py.test

""" Unit tests for mesh hierarchies """

# Copyright (C) 2015 Chris Richardson
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
# First added:  2015-01-30

import pytest
from dolfin import *

def test_hierarchy_2D():
    mesh = UnitSquareMesh(4, 6)
    markers = CellFunction("bool", mesh, True)
    mh = MeshHierarchy(mesh)
    assert mh.size() == 1
    mh = mh.refine(markers)
    assert mh.size() == 2
    assert mh[0].id() == mesh.id()
    assert mh.coarsest().id() == mesh.id()

def test_hierarchy_coarsening_2D():
    mesh = UnitSquareMesh(4, 6)
    mh = MeshHierarchy(mesh)
    for j in range(3):
        markers = CellFunction("bool", mh.finest(), False)
        for c in cells(mh.finest()):
            if (c.midpoint().y() > 0.5):
                markers[c] = True
        mh = mh.refine(markers)

    mh = mh.unrefine()

    markers = CellFunction("bool", mh.finest(), False)
    for c in cells(mh.finest()):
        if (c.midpoint().x() > 0.5):
            markers[c] = True
    mh = mh.coarsen(markers)
    assert mh.size() == 3
    vol2 = 0.0
    for c in cells(mh.finest()):
        vol2 += c.volume()
    vol2 = MPI.sum(mesh.mpi_comm(), vol2)
    assert round(1.0 - vol2, 7) == 0.0
