"""Unit tests for the DiscreteOperator class"""

# Copyright (C) 2015 Garth N. Wells
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

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
        num_edges = mesh.num_entities_global(1)
        assert G.size(0) == num_edges
        assert G.size(1) == mesh.num_entities_global(0)
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
