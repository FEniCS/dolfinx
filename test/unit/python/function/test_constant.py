#!/usr/bin/env py.test

"""Unit tests for the function library"""

# Copyright (C) 2007 Anders Logg
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
# First added:  2007-05-24
# Last changed: 2011-01-28

import pytest
from numpy import array
from dolfin import *


def test_name_argument():
    u = Constant(1.0)
    v = Constant(1.0, name="v")
    assert u.name() == "f_%d" % u.count()
    assert v.name() == "v"
    assert str(v) == "v"


def testConstantInit():
    c0 = Constant(1.)
    c1 = Constant([2, 3], interval)
    c2 = Constant([[2, 3], [3, 4]], triangle)
    c3 = Constant(array([2, 3]), tetrahedron)

    # FIXME:
    assert c0.cell() is None
    assert c1.cell() == interval
    assert c2.cell() == triangle
    assert c3.cell() == tetrahedron

    assert c0.ufl_shape == ()
    assert c1.ufl_shape == (2,)
    assert c2.ufl_shape == (2, 2)
    assert c3.ufl_shape == (2,)


def testGrad():
    import ufl
    zero = ufl.constantvalue.Zero((2, 3))
    c0 = Constant(1.)
    c3 = Constant(array([2, 3]), tetrahedron)

    def gradient(c):
        return grad(c)
    with pytest.raises(UFLException):
        grad(c0)
    assert zero == gradient(c3)


def test_compute_vertex_values():
    from numpy import zeros, all, array

    mesh = UnitCubeMesh(8, 8, 8)

    e0 = Constant(1)
    e1 = Constant((1, 2, 3))

    # e0_values = zeros(mesh.num_vertices(),dtype='d')
    # e1_values = zeros(mesh.num_vertices()*3,dtype='d')

    e0_values = e0.compute_vertex_values(mesh)
    e1_values = e1.compute_vertex_values(mesh)

    assert all(e0_values == 1)
    assert all(e1_values[:mesh.num_vertices()] == 1)
    assert all(e1_values[mesh.num_vertices():mesh.num_vertices()*2] == 2)
    assert all(e1_values[mesh.num_vertices()*2:mesh.num_vertices()*3] == 3)


def test_values():
    import numpy as np

    c0 = Constant(1.)
    c0_vals = c0.values()
    assert np.all(c0_vals == np.array([1.], dtype=np.double))

    c1 = Constant((1., 2.))
    c1_vals = c1.values()
    assert np.all(c1_vals == np.array([1., 2.], dtype=np.double))

    c2 = Constant((1., 2., 3.))
    c2_vals = c2.values()
    assert np.all(c2_vals == np.array([1., 2., 3.], dtype=np.double))


def test_str():
    c0 = Constant(1.)
    c0.str(False)
    c0.str(True)

    c1 = Constant((1., 2., 3.))
    c1.str(False)
    c1.str(True)
