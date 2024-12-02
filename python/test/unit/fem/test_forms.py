"""Tests for DOLFINx integration of various form operations"""

# Copyright (C) 2021 Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from mpi4py import MPI

import numpy as np
import pytest

import basix
import basix.ufl
import dolfinx
from dolfinx.fem import extract_function_spaces, form, functionspace
from dolfinx.fem.forms import form_cpp_class
from dolfinx.mesh import create_unit_square
from ufl import Measure, SpatialCoordinate, TestFunction, TrialFunction, dx, inner


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
    mesh = create_unit_square(MPI.COMM_WORLD, 32, 31)
    element = basix.ufl.element(
        "Lagrange",
        "triangle",
        4,
        lagrange_variant=basix.LagrangeVariant.gll_warped,
        dtype=dolfinx.default_real_type,
    )
    incorrect_element = basix.ufl.element(
        "Lagrange",
        "triangle",
        4,
        lagrange_variant=basix.LagrangeVariant.equispaced,
        dtype=dolfinx.default_real_type,
    )

    space = functionspace(mesh, element)
    incorrect_space = functionspace(mesh, incorrect_element)

    u = TrialFunction(space)
    v = TestFunction(space)

    a = inner(u, v) * dx

    dtype = dolfinx.default_scalar_type
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
            [incorrect_space._cpp_object, incorrect_space._cpp_object],
            [],
            [],
            {dolfinx.cpp.fem.IntegralType.cell: []},
            {},
            mesh._cpp_object,
        )
        dolfinx.fem.Form(f, ufcx_form, code)


def test_multiple_measures_one_subdomain_data():
    comm = MPI.COMM_WORLD
    mesh = dolfinx.mesh.create_unit_interval(comm, 10)
    x = SpatialCoordinate(mesh)
    num_cells_local = mesh.topology.index_map(mesh.topology.dim).size_local
    ct = dolfinx.mesh.meshtags(
        mesh,
        mesh.topology.dim,
        np.arange(num_cells_local, dtype=np.int32),
        np.arange(num_cells_local, dtype=np.int32),
    )

    dx = Measure("dx", domain=mesh, subdomain_data=ct)
    dx_stand = Measure("dx", domain=mesh)

    J = dolfinx.fem.form(x[0] ** 2 * dx + x[0] * dx_stand)
    J_local = dolfinx.fem.assemble_scalar(J)
    J_global = comm.allreduce(J_local, op=MPI.SUM)
    assert np.isclose(J_global, 1 / 3 + 1 / 2)
