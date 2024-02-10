"""Tests for DOLFINx integration of various form operations"""

# Copyright (C) 2021 Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from mpi4py import MPI

import pytest

from dolfinx.fem import extract_function_spaces, form, functionspace
from dolfinx.mesh import create_unit_square
from ufl import TestFunction, TrialFunction, dx, inner


def test_extract_forms():
    """Test extraction on unique function spaces for rows and columns of
    a block system"""
    mesh = create_unit_square(MPI.COMM_WORLD, 32, 31)
    V0 = functionspace(mesh, ("Lagrange", 1))
    V1 = functionspace(mesh, ("Lagrange", 2))
    V2 = V0.clone()
    V3 = V1.clone()

    v0, u0 = TestFunction(V0), TrialFunction(V0)
    v1, u1 = TestFunction(V1), TrialFunction(V1)
    v2, u2 = TestFunction(V2), TrialFunction(V2)
    v3, u3 = TestFunction(V3), TrialFunction(V3)

    a = form([[inner(u0, v0) * dx, inner(u1, v1) * dx], [inner(u2, v2) * dx, inner(u3, v3) * dx]])
    with pytest.raises(AssertionError):
        extract_function_spaces(a, 0)
    with pytest.raises(AssertionError):
        extract_function_spaces(a, 1)

    a = form([[inner(u0, v0) * dx, inner(u2, v1) * dx], [inner(u0, v2) * dx, inner(u2, v2) * dx]])
    with pytest.raises(AssertionError):
        extract_function_spaces(a, 0)
    Vc = extract_function_spaces(a, 1)
    assert Vc[0] is V0._cpp_object
    assert Vc[1] is V2._cpp_object

    a = form([[inner(u0, v0) * dx, inner(u1, v0) * dx], [inner(u2, v1) * dx, inner(u3, v1) * dx]])
    Vr = extract_function_spaces(a, 0)
    assert Vr[0] is V0._cpp_object
    assert Vr[1] is V1._cpp_object
    with pytest.raises(AssertionError):
        extract_function_spaces(a, 1)
