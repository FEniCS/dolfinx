# Copyright (C) 2011 Garth N. Wells
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import pytest

from dolfin import MPI, Edges, UnitCubeMesh, UnitSquareMesh
from dolfin_utils.test.fixtures import fixture
from dolfin_utils.test.skips import skip_in_parallel


@fixture
def cube():
    return UnitCubeMesh(MPI.comm_world, 5, 5, 5)


@fixture
def square():
    return UnitSquareMesh(MPI.comm_world, 5, 5)


@pytest.fixture(scope='module', params=range(2))
def meshes(cube, square, request):
    mesh = [cube, square]
    return mesh[request.param]
