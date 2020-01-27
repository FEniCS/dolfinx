# Copyright (C) 2019 Chris Richardson
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for the Constant class"""

import numpy as np
import pytest

from dolfinx import MPI, Constant, UnitCubeMesh


def test_scalar_constant():
    mesh = UnitCubeMesh(MPI.comm_world, 2, 2, 2)
    c = Constant(mesh, 1.0)
    assert c.value.shape == ()
    assert c.value == 1.0
    c.value += 1.0
    assert c.value == 2.0
    c.value = 3.0
    assert c.value == 3.0


def test_reshape():
    mesh = UnitCubeMesh(MPI.comm_world, 2, 2, 2)
    c = Constant(mesh, 1.0)
    with pytest.raises(ValueError):
        c.value.resize(100)


def test_wrong_dim():
    mesh = UnitCubeMesh(MPI.comm_world, 2, 2, 2)
    c = Constant(mesh, [1.0, 2.0])
    assert c.value.shape == (2,)
    with pytest.raises(ValueError):
        c.value = [1.0, 2.0, 3.0]


def test_vector_constant():
    mesh = UnitCubeMesh(MPI.comm_world, 2, 2, 2)
    c0 = Constant(mesh, [1.0, 2.0])
    c1 = Constant(mesh, np.array([1.0, 2.0]))
    assert (c0.value.all() == c1.value.all())
    c0.value += 1.0
    assert c0.value.all() == np.array([2.0, 3.0]).all()
    c0.value -= [1.0, 2.0]
    assert c0.value[0] == c0.value[1]


def test_tensor_constant():
    mesh = UnitCubeMesh(MPI.comm_world, 2, 2, 2)
    data = [[1.0, 2.0, 1.0], [1.0, 2.0, 1.0], [1.0, 2.0, 1.0]]
    c0 = Constant(mesh, data)
    assert c0.value.shape == (3, 3)
    assert c0.value.all() == np.asarray(data).all()
    c0.value *= 2.0
    assert c0.value.all() == (2.0 * np.asarray(data)).all()
