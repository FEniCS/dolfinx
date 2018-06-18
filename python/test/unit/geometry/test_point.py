"""Unit tests for Point interface"""

# Copyright (C) 2017 Jan Blechta
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import pytest
import numpy as np

from dolfin import Point, DOLFIN_EPS, DOLFIN_PI


def test_point_getitem():
    p = Point(1, 2, 3)
    assert p[0] == 1.0
    assert p[1] == 2.0
    assert p[2] == 3.0
    with pytest.raises(IndexError):
        p[3]
    assert np.all(p[:] == np.array((1.0, 2.0, 3.0)))


def test_point_setitem():
    p = Point()

    p[0] = 6.0
    assert p[0] == 6.0

    p[1] = 16.0
    p[1] += 600.0
    assert np.isclose(p[1], 616.0)

    p[2] = 111.0
    p[2] *= 12.0
    p[2] /= 2
    assert np.isclose(p[2], 666.0)

    with pytest.raises(IndexError):
        p[3] = 6666.0

    p[:] = (0, 0, 0)
    assert np.all(p[:] == 0)

    p[:] = (1, 2, 3)
    assert np.all(p[:] == (1, 2, 3))

    p[:] += np.array((1, 2, 3))
    assert np.all(p[:] == (2, 4, 6))

    p[:] /= 2
    assert np.all(p[:] == (1, 2, 3))

    p[:] *= np.array((2., 2., 2.))
    assert np.all(p[:] == (2, 4, 6))


def test_point_array():
    p = Point(1, 2, 3)
    assert np.all(p.array() == (1, 2, 3))

    # Point.array() is a copy, no in-place modification
    p.array()[:] += 1000.0
    assert np.all(p.array() == (1, 2, 3))


def test_point_equality():
    p = Point(1.23, 2, DOLFIN_PI)
    q = Point(1.23, 2, DOLFIN_PI)
    r = Point(1.23 + DOLFIN_EPS, 2, DOLFIN_PI)
    assert p == q
    assert p != r
