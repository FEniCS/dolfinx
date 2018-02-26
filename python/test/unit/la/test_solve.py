"""Unit tests for the solve interface"""

# Copyright (C) 2011 Garth N. Wells
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from dolfin import *

def test_normalize_average():
    size = 200
    value = 2.0
    x = Vector(MPI.comm_world, size)
    x[:] = value
    factor = normalize(x, "average")
    assert factor == value
    assert x.sum() == 0.0

def test_normalize_l2():
    size = 200
    value = 2.0
    x = Vector(MPI.comm_world, size)
    x[:] = value
    factor = normalize(x, "l2")
    assert round(factor - sqrt(size*value*value), 7) == 0
    assert round(x.norm("l2") - 1.0, 7) == 0
