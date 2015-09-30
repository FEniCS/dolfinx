#!/usr/bin/env py.test

"""Unit tests for the DiscreteOperator class"""

# Copyright (C) 2015 Garth N. Wells
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

from __future__ import print_function
import pytest
import numpy as np
from dolfin import *

from dolfin_utils.test import *


def test_gradient():
    """Test discrete gradient computation (typically used for curl-curl
    AMG preconditioners"""

    def compute_discrete_gradient(mesh):
        V = FunctionSpace(mesh, "Lagrange", 1)
        W = FunctionSpace(mesh, "Nedelec 1st kind H(curl)", 1)

        G = DiscreteOperators.build_gradient(W, V)
        num_edges = mesh.size_global(1)
        assert G.size(0) == num_edges
        assert G.size(1) == mesh.size_global(0)
        assert round(G.norm("frobenius") - sqrt(2.0*num_edges), 8) == 0.0

    meshes = [UnitSquareMesh(11, 6), UnitCubeMesh(4, 3, 7)]
    for mesh in meshes:
        compute_discrete_gradient(mesh)


def test_incompatible_spaces():
    "Test that error is thrown when function spaces are not compatible"

    mesh = UnitSquareMesh(13, 7)
    V = FunctionSpace(mesh, "Lagrange", 1)
    W = FunctionSpace(mesh, "Nedelec 1st kind H(curl)", 1)
    with pytest.raises(RuntimeError):
        G = DiscreteOperators.build_gradient(V, W)
    with pytest.raises(RuntimeError):
        G = DiscreteOperators.build_gradient(V, V)
    with pytest.raises(RuntimeError):
        G = DiscreteOperators.build_gradient(W, W)

    V = FunctionSpace(mesh, "Lagrange", 2)
    with pytest.raises(RuntimeError):
        G = DiscreteOperators.build_gradient(W, V)
