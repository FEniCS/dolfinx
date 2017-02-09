#!/usr/bin/env py.test

"""Unit tests for MeshFunctions"""

# Copyright (C) 2011 Garth N. Wells
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
# First added:  2011-03-10
# Last changed: 2011-03-10

import pytest
import numpy.random
from dolfin import *
from six.moves import xrange as range
from dolfin_utils.test import fixture, skip_in_parallel


@pytest.fixture(scope="module", params=range(5))
def name(request):
    names = ["Cell", "Vertex", "Edge", "Face", "Facet"]
    return names[request.param]


@pytest.fixture(scope="module", params=range(4))
def tp(request):
    tps = ['int', 'size_t', 'bool', 'double']
    return tps[request.param]


@fixture
def mesh():
    return UnitCubeMesh(3, 3, 3)


@fixture
def funcs(mesh):
    names = ["Cell", "Vertex", "Edge", "Face", "Facet"]
    tps = ['int', 'size_t', 'bool', 'double']
    funcs = {}
    for tp in tps:
        for name in names:
            funcs[(tp, name)] = eval("%sFunction('%s', mesh)" % (name, tp))
    return funcs


@fixture
def cube():
    return UnitCubeMesh(8, 8, 8)


@fixture
def f(cube):
    return MeshFunction('int', cube, 0)


def test_size(tp, name, funcs, mesh):
    if name is "Vertex":
        a = len(funcs[(tp, name)])
        b = mesh.num_vertices()
        assert a == b
    else:
        a = len(funcs[(tp, name)])
        b = getattr(mesh, "num_%ss" % name.lower())()
        assert a == b


def test_access_type(tp, name, funcs):
    type_dict = dict(int=int, size_t=int, double=float, bool=bool)
    assert isinstance(funcs[(tp, name)][0], type_dict[tp])


def test_numpy_access(funcs, tp, name):
    values = funcs[(tp, name)].array()
    values[:] = numpy.random.rand(len(values))
    assert all(values[i] == funcs[(tp, name)][i] for i in range(len(values)))


def test_iterate(tp, name, funcs):
    for index, value in enumerate(funcs[(tp, name)]):
        pass
    assert index == len(funcs[(tp, name)])-1
    with pytest.raises(IndexError):
        funcs[(tp, name)].__getitem__(len(funcs[(tp, name)]))


def test_setvalues(tp, funcs, name):
    if tp != 'bool':
        with pytest.raises(TypeError):
            funcs[(tp, name)].__setitem__(len(funcs[(tp, name)])-1, "jada")


def test_Create(cube):
    """Create MeshFunctions."""
    v = MeshFunction("size_t", cube)

    v = MeshFunction("size_t", cube, 0)
    assert v.size() == cube.num_vertices()

    v = MeshFunction("size_t", cube, 1)
    assert v.size() == cube.num_edges()

    v = MeshFunction("size_t", cube, 2)
    assert v.size() == cube.num_facets()

    v = MeshFunction("size_t", cube, 3)
    assert v.size() == cube.num_cells()


def test_CreateAssign(cube):
    """Create MeshFunctions with value."""
    i = 10
    v = MeshFunction("size_t", cube, 0, i)
    assert v.size() == cube.num_vertices()
    assert v[0] == i

    v = MeshFunction("size_t", cube, 1, i)
    assert v.size() == cube.num_edges()
    assert v[0] == i

    v = MeshFunction("size_t", cube, 2, i)
    assert v.size() == cube.num_facets()
    assert v[0] == i

    v = MeshFunction("size_t", cube, 3, i)
    assert v.size() == cube.num_cells()
    assert v[0] == i


def test_Assign(f, cube):
    f = f
    f[3] = 10
    v = Vertex(cube, 3)
    assert f[v] == 10


@skip_in_parallel
def test_meshfunction_where_equal():
    mesh = UnitSquareMesh(2, 2)

    cf = CellFunctionSizet(mesh)
    cf.set_all(1)
    cf[0] = 3
    cf[3] = 3
    assert list(cf.where_equal(3)) == [0, 3]
    assert list(cf.where_equal(1)) == [1, 2, 4, 5, 6, 7]

    ff = FacetFunctionSizet(mesh)
    ff.set_all(0)
    ff[0] = 1
    ff[2] = 3
    ff[3] = 3
    assert list(ff.where_equal(1)) == [0]
    assert list(ff.where_equal(3)) == [2, 3]
    assert list(ff.where_equal(0)) == [1] + list(range(4, ff.size()))

    vf = VertexFunctionSizet(mesh)
    vf.set_all(3)
    vf[1] = 1
    vf[2] = 1
    assert list(vf.where_equal(1)) == [1, 2]
    assert list(vf.where_equal(3)) == [0] + list(range(3, vf.size()))
