"""Unit tests for custom_data functionality in assembly."""

# Copyright (C) 2025 Susanne Claus
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from mpi4py import MPI

import numpy as np
import pytest

import dolfinx
from dolfinx import la
from dolfinx.fem import Form, IntegralType, form_cpp_class, functionspace
from dolfinx.mesh import create_unit_square
from ffcx.codegeneration.utils import (
    numba_ufcx_kernel_signature,
    voidptr_to_float64_ptr,
)

numba = pytest.importorskip("numba")
ufcx_signature = numba_ufcx_kernel_signature


def tabulate_rank1_with_custom_data(dtype, xdtype):
    """Kernel that reads a scaling factor from custom_data.

    Note: custom_data must be set to a valid pointer before assembly.
    """

    @numba.cfunc(ufcx_signature(dtype, xdtype), nopython=True)
    def tabulate(b_, w_, c_, coords_, local_index, orientation, custom_data):
        b = numba.carray(b_, (3), dtype=dtype)
        coordinate_dofs = numba.carray(coords_, (3, 3), dtype=xdtype)

        # Cast void* to float64* and read the scale value
        typed_ptr = voidptr_to_float64_ptr(custom_data)
        scale = typed_ptr[0]

        x0, y0 = coordinate_dofs[0, :2]
        x1, y1 = coordinate_dofs[1, :2]
        x2, y2 = coordinate_dofs[2, :2]

        # 2x Element area Ae
        Ae = abs((x0 - x1) * (y2 - y1) - (y0 - y1) * (x2 - x1))
        b[:] = scale * Ae / 6.0

    return tabulate


def tabulate_rank2_with_custom_data(dtype, xdtype):
    """Kernel that reads a scaling factor from custom_data for matrix assembly.

    Note: custom_data must be set to a valid pointer before assembly.
    """

    @numba.cfunc(ufcx_signature(dtype, xdtype), nopython=True)
    def tabulate(A_, w_, c_, coords_, entity_local_index, cell_orientation, custom_data):
        A = numba.carray(A_, (3, 3), dtype=dtype)
        coordinate_dofs = numba.carray(coords_, (3, 3), dtype=xdtype)

        # Cast void* to float64* and read the scale value
        typed_ptr = voidptr_to_float64_ptr(custom_data)
        scale = typed_ptr[0]

        x0, y0 = coordinate_dofs[0, :2]
        x1, y1 = coordinate_dofs[1, :2]
        x2, y2 = coordinate_dofs[2, :2]

        # 2x Element area Ae
        Ae = abs((x0 - x1) * (y2 - y1) - (y0 - y1) * (x2 - x1))
        B = np.array([y1 - y2, y2 - y0, y0 - y1, x2 - x1, x0 - x2, x1 - x0], dtype=dtype).reshape(
            2, 3
        )
        A[:, :] = scale * np.dot(B.T, B) / (2 * Ae)

    return tabulate


@pytest.mark.parametrize("dtype", [np.float64])
def test_custom_data_vector_assembly(dtype):
    """Test that custom_data is correctly passed to kernels during vector assembly."""
    xdtype = np.real(dtype(0)).dtype
    k1 = tabulate_rank1_with_custom_data(dtype, xdtype)

    mesh = create_unit_square(MPI.COMM_WORLD, 13, 13, dtype=xdtype)
    V = functionspace(mesh, ("Lagrange", 1))

    tdim = mesh.topology.dim
    num_cells = mesh.topology.index_map(tdim).size_local + mesh.topology.index_map(tdim).num_ghosts
    cells = np.arange(num_cells, dtype=np.int32)
    active_coeffs = np.array([], dtype=np.int8)

    # Create custom_data with scale=1.0
    scale_value = np.array([1.0], dtype=dtype)
    scale_ptr = scale_value.ctypes.data

    # Pass custom_data at form creation time via the 5th element of the integrals tuple
    integrals = {IntegralType.cell: [(0, k1.address, cells, active_coeffs, scale_ptr)]}
    formtype = form_cpp_class(dtype)
    L = Form(formtype([V._cpp_object], integrals, [], [], False, [], mesh=mesh._cpp_object))

    # Assemble with scale=1.0
    b1 = dolfinx.fem.assemble_vector(L)
    b1.scatter_reverse(la.InsertMode.add)
    norm1 = la.norm(b1)

    # Verify we can read back the custom_data pointer
    assert L._cpp_object.custom_data(IntegralType.cell, 0, 0) == scale_ptr

    # Update custom_data to scale=2.0 (by modifying the underlying array)
    scale_value[0] = 2.0
    b2 = dolfinx.fem.assemble_vector(L)
    b2.scatter_reverse(la.InsertMode.add)
    norm2 = la.norm(b2)

    # The norm with scale=2 should be 2x the norm with scale=1
    assert np.isclose(norm2, 2.0 * norm1)

    # Test with scale=3.0
    scale_value[0] = 3.0
    b3 = dolfinx.fem.assemble_vector(L)
    b3.scatter_reverse(la.InsertMode.add)
    norm3 = la.norm(b3)

    assert np.isclose(norm3, 3.0 * norm1)


@pytest.mark.parametrize("dtype", [np.float64])
def test_custom_data_matrix_assembly(dtype):
    """Test that custom_data is correctly passed to kernels during matrix assembly."""
    xdtype = np.real(dtype(0)).dtype
    k2 = tabulate_rank2_with_custom_data(dtype, xdtype)

    mesh = create_unit_square(MPI.COMM_WORLD, 13, 13, dtype=xdtype)
    V = functionspace(mesh, ("Lagrange", 1))

    cells = np.arange(mesh.topology.index_map(mesh.topology.dim).size_local, dtype=np.int32)
    active_coeffs = np.array([], dtype=np.int8)

    # Create custom_data with scale=1.0
    scale_value = np.array([1.0], dtype=dtype)

    # Pass custom_data at form creation time via the 5th element
    integrals = {
        IntegralType.cell: [(0, k2.address, cells, active_coeffs, scale_value.ctypes.data)]
    }
    formtype = form_cpp_class(dtype)
    a = Form(
        formtype(
            [V._cpp_object, V._cpp_object],
            integrals,
            [],
            [],
            False,
            [],
            mesh=mesh._cpp_object,
        )
    )

    # Assemble with scale=1.0
    A1 = dolfinx.fem.assemble_matrix(a)
    A1.scatter_reverse()
    norm1 = np.sqrt(A1.squared_norm())

    # Update custom_data to scale=2.0 (by modifying the underlying array)
    scale_value[0] = 2.0
    A2 = dolfinx.fem.assemble_matrix(a)
    A2.scatter_reverse()
    norm2 = np.sqrt(A2.squared_norm())

    # The norm with scale=2 should be 2x the norm with scale=1
    assert np.isclose(norm2, 2.0 * norm1)


@pytest.mark.parametrize("dtype", [np.float64])
def test_custom_data_default_nullptr(dtype):
    """Test that custom_data defaults to nullptr (None) when not provided."""
    xdtype = np.real(dtype(0)).dtype

    # Define a simple kernel that doesn't use custom_data
    @numba.cfunc(ufcx_signature(dtype, xdtype), nopython=True)
    def tabulate_simple(b_, w_, c_, coords_, local_index, orientation, custom_data):
        b = numba.carray(b_, (3), dtype=dtype)
        coordinate_dofs = numba.carray(coords_, (3, 3), dtype=xdtype)

        x0, y0 = coordinate_dofs[0, :2]
        x1, y1 = coordinate_dofs[1, :2]
        x2, y2 = coordinate_dofs[2, :2]

        Ae = abs((x0 - x1) * (y2 - y1) - (y0 - y1) * (x2 - x1))
        b[:] = Ae / 6.0

    mesh = create_unit_square(MPI.COMM_WORLD, 5, 5, dtype=xdtype)
    V = functionspace(mesh, ("Lagrange", 1))

    tdim = mesh.topology.dim
    num_cells = mesh.topology.index_map(tdim).size_local + mesh.topology.index_map(tdim).num_ghosts
    cells = np.arange(num_cells, dtype=np.int32)
    active_coeffs = np.array([], dtype=np.int8)

    # Pass None as custom_data (5th element) to get std::nullopt
    integrals = {IntegralType.cell: [(0, tabulate_simple.address, cells, active_coeffs, None)]}
    formtype = form_cpp_class(dtype)
    L = Form(formtype([V._cpp_object], integrals, [], [], False, [], mesh=mesh._cpp_object))

    # custom_data should be None (std::nullopt) when passed as None
    assert L._cpp_object.custom_data(IntegralType.cell, 0, 0) is None


@pytest.mark.parametrize("dtype", [np.float64])
def test_custom_data_struct(dtype):
    """Test passing a struct with multiple values through custom_data."""
    xdtype = np.real(dtype(0)).dtype

    # Define a kernel that reads two values from custom_data
    @numba.cfunc(ufcx_signature(dtype, xdtype), nopython=True)
    def tabulate_with_struct(b_, w_, c_, coords_, local_index, orientation, custom_data):
        b = numba.carray(b_, (3), dtype=dtype)
        coordinate_dofs = numba.carray(coords_, (3, 3), dtype=xdtype)

        # Cast void* to float64* and read two values: [scale, offset]
        typed_ptr = voidptr_to_float64_ptr(custom_data)
        scale = typed_ptr[0]
        offset = typed_ptr[1]

        x0, y0 = coordinate_dofs[0, :2]
        x1, y1 = coordinate_dofs[1, :2]
        x2, y2 = coordinate_dofs[2, :2]

        Ae = abs((x0 - x1) * (y2 - y1) - (y0 - y1) * (x2 - x1))
        b[:] = scale * Ae / 6.0 + offset

    mesh = create_unit_square(MPI.COMM_WORLD, 5, 5, dtype=xdtype)
    V = functionspace(mesh, ("Lagrange", 1))

    tdim = mesh.topology.dim
    # Only iterate over local cells (not ghosts) to avoid double-counting
    # contributions after scatter_reverse
    num_local_cells = mesh.topology.index_map(tdim).size_local
    cells = np.arange(num_local_cells, dtype=np.int32)
    active_coeffs = np.array([], dtype=np.int8)

    # Test 1: scale=1.0, offset=0.0 (baseline)
    struct_data = np.array([1.0, 0.0], dtype=dtype)

    # Pass custom_data at form creation time
    integrals = {
        IntegralType.cell: [
            (0, tabulate_with_struct.address, cells, active_coeffs, struct_data.ctypes.data)
        ]
    }
    formtype = form_cpp_class(dtype)
    L = Form(formtype([V._cpp_object], integrals, [], [], False, [], mesh=mesh._cpp_object))

    b_baseline = dolfinx.fem.assemble_vector(L)
    b_baseline.scatter_reverse(la.InsertMode.add)
    norm_baseline = la.norm(b_baseline)

    # Test 2: scale=2.0, offset=0.0 - should double the norm
    struct_data[0] = 2.0
    struct_data[1] = 0.0
    b_scaled = dolfinx.fem.assemble_vector(L)
    b_scaled.scatter_reverse(la.InsertMode.add)
    norm_scaled = la.norm(b_scaled)
    assert np.isclose(norm_scaled, 2.0 * norm_baseline)

    # Test 3: scale=0.0, offset=1.0 - pure offset contribution
    struct_data[0] = 0.0
    struct_data[1] = 1.0
    b_offset = dolfinx.fem.assemble_vector(L)
    b_offset.scatter_reverse(la.InsertMode.add)
    # With offset=1.0, each DOF gets contribution from each cell it touches
    # Interior nodes touch 6 cells, edge nodes touch 3-4, corner nodes touch 1-2
    # The sum of all contributions equals 3 * num_local_cells (3 DOFs per cell, offset=1.0 each)
    # Sum only local DOFs and gather across processes
    local_sum = np.sum(b_offset.array[: V.dofmap.index_map.size_local * V.dofmap.index_map_bs])
    total_contribution = mesh.comm.allreduce(local_sum, op=MPI.SUM)
    total_cells = mesh.comm.allreduce(num_local_cells, op=MPI.SUM)
    assert np.isclose(total_contribution, 3.0 * total_cells)


@pytest.mark.parametrize("dtype", [np.float64])
def test_custom_data_multiple_parameters(dtype):
    """Test custom_data with multiple parameters demonstrating complex data passing.

    This test shows how to pass multiple values through custom_data, simulating
    the use case of passing runtime-computed parameters like material properties
    or integration parameters.
    """
    xdtype = np.real(dtype(0)).dtype

    # Kernel that uses three parameters: coefficient, exponent base, and additive term
    # Computes: coeff * base^area + additive
    @numba.cfunc(ufcx_signature(dtype, xdtype), nopython=True)
    def tabulate_with_params(b_, w_, c_, coords_, local_index, orientation, custom_data):
        b = numba.carray(b_, (3), dtype=dtype)
        coordinate_dofs = numba.carray(coords_, (3, 3), dtype=xdtype)

        # Read three parameters from custom_data
        typed_ptr = voidptr_to_float64_ptr(custom_data)
        coeff = typed_ptr[0]
        power = typed_ptr[1]
        additive = typed_ptr[2]

        x0, y0 = coordinate_dofs[0, :2]
        x1, y1 = coordinate_dofs[1, :2]
        x2, y2 = coordinate_dofs[2, :2]

        Ae = abs((x0 - x1) * (y2 - y1) - (y0 - y1) * (x2 - x1))
        # Use power as a simple multiplier (avoiding actual power function complexity)
        val = coeff * power * Ae / 6.0 + additive
        b[:] = val

    mesh = create_unit_square(MPI.COMM_WORLD, 4, 4, dtype=xdtype)
    V = functionspace(mesh, ("Lagrange", 1))

    tdim = mesh.topology.dim
    num_cells = mesh.topology.index_map(tdim).size_local + mesh.topology.index_map(tdim).num_ghosts
    cells = np.arange(num_cells, dtype=np.int32)
    active_coeffs = np.array([], dtype=np.int8)

    # Test with specific parameters: coeff=2, power=3, additive=0
    params = np.array([2.0, 3.0, 0.0], dtype=dtype)

    # Pass custom_data at form creation time
    integrals = {
        IntegralType.cell: [
            (0, tabulate_with_params.address, cells, active_coeffs, params.ctypes.data)
        ]
    }
    formtype = form_cpp_class(dtype)
    L = Form(formtype([V._cpp_object], integrals, [], [], False, [], mesh=mesh._cpp_object))

    b1 = dolfinx.fem.assemble_vector(L)
    b1.scatter_reverse(la.InsertMode.add)
    norm1 = la.norm(b1)

    # Change parameters: coeff=1, power=6, additive=0 (should give same result: 1*6 = 2*3)
    params[0] = 1.0
    params[1] = 6.0
    b2 = dolfinx.fem.assemble_vector(L)
    b2.scatter_reverse(la.InsertMode.add)
    norm2 = la.norm(b2)

    assert np.isclose(norm1, norm2)


@pytest.mark.parametrize("dtype", [np.float64])
def test_custom_data_global_parameter_update(dtype):
    """Test updating custom_data between assemblies for parameter studies.

    This demonstrates a key use case: running multiple assemblies with
    different parameter values without recreating the form. The custom_data
    points to a parameter that can be modified between assembly calls.
    """
    xdtype = np.real(dtype(0)).dtype

    # Kernel that reads a diffusion coefficient from custom_data
    @numba.cfunc(ufcx_signature(dtype, xdtype), nopython=True)
    def tabulate_diffusion(b_, w_, c_, coords_, local_index, orientation, custom_data):
        b = numba.carray(b_, (3), dtype=dtype)
        coordinate_dofs = numba.carray(coords_, (3, 3), dtype=xdtype)

        # Read diffusion coefficient from custom_data
        typed_ptr = voidptr_to_float64_ptr(custom_data)
        kappa = typed_ptr[0]  # Diffusion coefficient

        x0, y0 = coordinate_dofs[0, :2]
        x1, y1 = coordinate_dofs[1, :2]
        x2, y2 = coordinate_dofs[2, :2]

        # 2x Element area Ae
        Ae = abs((x0 - x1) * (y2 - y1) - (y0 - y1) * (x2 - x1))
        # Simple load vector scaled by diffusion coefficient
        b[:] = kappa * Ae / 6.0

    mesh = create_unit_square(MPI.COMM_WORLD, 8, 8, dtype=xdtype)
    V = functionspace(mesh, ("Lagrange", 1))

    tdim = mesh.topology.dim
    num_cells = mesh.topology.index_map(tdim).size_local + mesh.topology.index_map(tdim).num_ghosts
    cells = np.arange(num_cells, dtype=np.int32)
    active_coeffs = np.array([], dtype=np.int8)

    # Parameter array - this can be updated between assemblies
    kappa = np.array([1.0], dtype=dtype)

    integrals = {
        IntegralType.cell: [
            (0, tabulate_diffusion.address, cells, active_coeffs, kappa.ctypes.data)
        ]
    }
    formtype = form_cpp_class(dtype)
    L = Form(formtype([V._cpp_object], integrals, [], [], False, [], mesh=mesh._cpp_object))

    # Store results for different kappa values
    results = []
    kappa_values = [0.1, 1.0, 10.0, 100.0]

    for k in kappa_values:
        kappa[0] = k
        b = dolfinx.fem.assemble_vector(L)
        b.scatter_reverse(la.InsertMode.add)
        results.append(la.norm(b))

    # Verify linear scaling: norm should scale linearly with kappa
    # norm(kappa=k) / norm(kappa=1) should equal k
    base_norm = results[1]  # kappa=1.0
    for i, k in enumerate(kappa_values):
        expected_ratio = k / 1.0
        actual_ratio = results[i] / base_norm
        assert np.isclose(actual_ratio, expected_ratio, rtol=1e-10)
