"""Unit tests for the function library"""

# Copyright (C) 2007 Anders Logg
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import pytest
from numpy import array
from dolfin import (Constant, interval, triangle, tetrahedron, quadrilateral, hexahedron,
                    UnitCubeMesh, grad, MPI, CellType)
from ufl import UFLException


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
    c4 = Constant([[2, 3], [3, 4]], quadrilateral)
    c5 = Constant(array([2, 3]), hexahedron)

    # FIXME:
    assert c0.cell() is None
    assert c1.cell() == interval
    assert c2.cell() == triangle
    assert c3.cell() == tetrahedron
    assert c4.cell() == quadrilateral
    assert c5.cell() == hexahedron

    assert c0.ufl_shape == ()
    assert c1.ufl_shape == (2,)
    assert c2.ufl_shape == (2, 2)
    assert c3.ufl_shape == (2,)
    assert c4.ufl_shape == (2, 2)
    assert c5.ufl_shape == (2,)


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


@pytest.mark.parametrize('mesh_factory', [(UnitCubeMesh, (MPI.comm_world, 3, 3, 3)),
                                          (UnitCubeMesh, (MPI.comm_world, 3, 3, 3, CellType.Type.hexahedron))])
def test_compute_point_values(mesh_factory):
    from numpy import all

    func, args = mesh_factory
    mesh = func(*args)

    e0 = Constant(1)
    e1 = Constant((1, 2, 3))

    e0_values = e0.compute_point_values(mesh)
    e1_values = e1.compute_point_values(mesh)

    assert all(e0_values == 1)
    assert all(e1_values == [1, 2, 3])


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


def test_assign():
    c0 = Constant(1.)
    assert c0.values() == (1,)
    c0.assign(Constant(3))
    assert c0.values() == (3,)

    c1 = Constant([1, 2])
    assert (c1.values() == (1, 2)).all()
    c1.assign(Constant([3, 4]))
    assert (c1.values() == (3, 4)).all()
