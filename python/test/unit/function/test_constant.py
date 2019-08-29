# Copyright (C) 2019 Chris Richardson
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for the Constant class"""

import numpy as np
from dolfin import MPI, UnitCubeMesh
from dolfin.function import Constant


def test_scalar_constant():
    mesh = UnitCubeMesh(MPI.comm_world, 2, 2, 2)

    c = Constant(mesh, 1.0)
    assert (c.value == 1.0)

    c.value += 1.0
    assert (c.value == 2.0)

    c.value = 3.0
    assert (c.value == 3.0)

    assert(np.array(c._cpp_object) == 3.0)


def test_vector_constant():
    mesh = UnitCubeMesh(MPI.comm_world, 2, 2, 2)

    c1 = Constant(mesh, [1.0, 2.0])

    c2 = Constant(mesh, np.array([1.0, 2.0]))
    assert (c1.value.all() == c2.value.all())

    c1.value += 1.0
    assert (c1.value.all() == np.array([2.0, 3.0]).all())

    c1.value -= [1.0, 2.0]
    assert (c1.value[0] == c1.value[1])
