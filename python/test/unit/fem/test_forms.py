"""Tests for DOLFINx integration of various form operations"""

# Copyright (C) 2021 Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from mpi4py import MPI

import pytest

import basix
import basix.ufl
import dolfinx
from dolfinx.fem import extract_function_spaces, form, functionspace
from dolfinx.fem.forms import form_cpp_class
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


def test_incorrect_element():
    """Test that an error is raised if an incorrect element is used."""
    if not dolfinx.common.has_debug:
        pytest.skip("Error will only be thrown for incorrect spaecs in debug mode.")

    dtype = dolfinx.default_scalar_type

    mesh = create_unit_square(MPI.COMM_WORLD, 32, 31)
    element = basix.ufl.element(
        "Lagrange",
        "triangle",
        4,
        lagrange_variant=basix.LagrangeVariant.gll_warped,
        dtype=dtype,
    )
    incorrect_element = basix.ufl.element(
        "Lagrange",
        "triangle",
        4,
        lagrange_variant=basix.LagrangeVariant.equispaced,
        dtype=dtype,
    )

    space = functionspace(mesh, element)
    incorrect_space = functionspace(mesh, incorrect_element)

    u = TrialFunction(space)
    v = TestFunction(space)

    a = inner(u, v) * dx

    ftype = form_cpp_class(dtype)

    ufcx_form, module, code = dolfinx.jit.ffcx_jit(
        mesh.comm, a, form_compiler_options={"scalar_type": dtype}
    )

    f = ftype(
        module.ffi.cast("uintptr_t", module.ffi.addressof(ufcx_form)),
        [space._cpp_object, space._cpp_object],
        [],
        [],
        {dolfinx.cpp.fem.IntegralType.cell: []},
        {},
        mesh._cpp_object,
    )
    dolfinx.fem.Form(f, ufcx_form, code)

    with pytest.raises(RuntimeError):
        f = ftype(
            module.ffi.cast("uintptr_t", module.ffi.addressof(ufcx_form)),
            [incorrect_space._cpp_object, space._cpp_object],
            [],
            [],
            {dolfinx.cpp.fem.IntegralType.cell: []},
            {},
            mesh._cpp_object,
        )
        dolfinx.fem.Form(f, ufcx_form, code)

    with pytest.raises(RuntimeError):
        f = ftype(
            module.ffi.cast("uintptr_t", module.ffi.addressof(ufcx_form)),
            [space._cpp_object, incorrect_space._cpp_object],
            [],
            [],
            {dolfinx.cpp.fem.IntegralType.cell: []},
            {},
            mesh._cpp_object,
        )
        dolfinx.fem.Form(f, ufcx_form, code)

    with pytest.raises(RuntimeError):
        f = ftype(
            module.ffi.cast("uintptr_t", module.ffi.addressof(ufcx_form)),
            [incorrect_space._cpp_object, incorrect_space._cpp_object],
            [],
            [],
            {dolfinx.cpp.fem.IntegralType.cell: []},
            {},
            mesh._cpp_object,
        )
        dolfinx.fem.Form(f, ufcx_form, code)
