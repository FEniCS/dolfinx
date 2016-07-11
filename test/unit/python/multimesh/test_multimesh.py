#!/usr/bin/env py.test

"""Unit tests for the MultiMeshFunction class"""

# Copyright (C) 2016 Jorgen S. Dokken
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
#
# First added:  2016-06-11
# Last changed: 2016-06-11

import pytest
from dolfin import *
import ufl 

from dolfin_utils.test import fixture    


@fixture
def multimesh():
    mesh_0 = RectangleMesh(Point(-1.5, -0.75), Point(1.5, 0.75), 40, 20)
    mesh_1 = RectangleMesh(Point(0.5, -1.5),  Point(2,1.5),25,40)
    multimesh = MultiMesh()
    multimesh.add(mesh_0)
    multimesh.add(mesh_1)
    multimesh.build()
    return multimesh

@fixture
def V(multimesh):
    return MultiMeshFunctionSpace(multimesh, 'CG', 1)

@fixture
def v(V):
    return MultiMeshFunction(V)

def test_measure_mul(v, multimesh):
    assert isinstance(v*dX, ufl.form.Form) 


def test_funcspace_storage(v):
    assert isinstance(MultiMeshFunction(v.function_space()), MultiMeshFunction)
