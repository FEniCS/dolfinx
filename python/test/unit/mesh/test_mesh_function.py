# Copyright (C) 2011 Garth N. Wells
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import numpy.random
import pytest

from dolfin import MPI, MeshFunction, UnitCubeMesh, UnitSquareMesh, Vertex
from dolfin_utils.test.fixtures import fixture
from dolfin_utils.test.skips import skip_in_parallel


@pytest.fixture(params=range(4))
def name(request):
    names = [3, 0, 1, 2]
    return names[request.param]


@pytest.fixture(params=range(4))
def tp(request):
    tps = ['int', 'size_t', 'bool', 'double']
    return tps[request.param]


@fixture
def mesh():
    return UnitCubeMesh(MPI.comm_world, 3, 3, 3)


@fixture
def funcs(mesh):
    names = [3, 0, 1, 2]
    tps = ['int', 'size_t', 'bool', 'double']
    funcs = {}
    for tp in tps:
        for name in names:
            funcs[(tp, name)] = eval("MeshFunction('%s', mesh, %d, 0)" %
                                     (tp, name))
    return funcs


@fixture
def cube():
    return UnitCubeMesh(MPI.comm_world, 8, 8, 8)


@fixture
def f(cube):
    return MeshFunction('int', cube, 0, 0)


def test_size(tp, name, funcs, mesh):
    if name == 0:
        a = len(funcs[(tp, name)])
        b = mesh.num_entities(0)
        assert a == b
    else:
        a = len(funcs[(tp, name)])
        b = mesh.num_entities(name)
        assert a == b


def test_access_type(tp, name, funcs):
    type_dict = dict(int=int, size_t=int, double=float, bool=bool)
    assert isinstance(funcs[(tp, name)][0], type_dict[tp])


def test_numpy_access(funcs, tp, name):
    values = funcs[(tp, name)].array()
    values[:] = numpy.random.rand(len(values))
    assert all(values[i] == funcs[(tp, name)][i] for i in range(len(values)))


def test_setvalues(tp, funcs, name):
    if tp != 'bool':
        with pytest.raises(TypeError):
            funcs[(tp, name)].__setitem__(len(funcs[(tp, name)]) - 1, "jada")


def test_Create(cube):
    """Create MeshFunctions."""

    v = MeshFunction("size_t", cube, 0, 0)
    assert v.size() == cube.num_entities(0)

    v = MeshFunction("size_t", cube, 1, 0)
    assert v.size() == cube.num_entities(1)

    v = MeshFunction("size_t", cube, 2, 0)
    assert v.size() == cube.num_entities(2)

    v = MeshFunction("size_t", cube, 3, 0)
    assert v.size() == cube.num_cells()


def test_CreateAssign(cube):
    """Create MeshFunctions with value."""
    i = 10
    v = MeshFunction("size_t", cube, 0, i)
    assert v.size() == cube.num_entities(0)
    assert v[0] == i

    v = MeshFunction("size_t", cube, 1, i)
    assert v.size() == cube.num_entities(1)
    assert v[0] == i

    v = MeshFunction("size_t", cube, 2, i)
    assert v.size() == cube.num_entities(2)
    assert v[0] == i

    v = MeshFunction("size_t", cube, 3, i)
    assert v.size() == cube.num_cells()
    assert v[0] == i


def test_Assign(f, cube):
    f = f
    f[3] = 10
    v = Vertex(cube, 3)
    assert f[v] == 10
