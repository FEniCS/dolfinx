# Copyright (C) 2019 Michal Habera
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import numpy
import numpy.random
import pytest

from dolfinx import MPI, MeshFunction, UnitCubeMesh


dtypes = (
    ('int', numpy.intc),
    ('double', numpy.float)
)


@pytest.fixture
def mesh():
    return UnitCubeMesh(MPI.comm_world, 3, 3, 3)


@pytest.mark.parametrize("dtype", dtypes)
def test_data_types(dtype, mesh):
    dtype_str, dtype = dtype
    mf = MeshFunction(dtype_str, mesh, 0, 0)
    assert isinstance(mf.values[0], dtype)


@pytest.mark.parametrize("dtype", dtypes)
def test_numpy_access(dtype, mesh):
    dtype_str, dtype = dtype
    mf = MeshFunction(dtype_str, mesh, 0, 0)
    values = mf.values
    values[:] = numpy.random.rand(len(values))
    assert numpy.all(values == mf.values)
