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
import ffcx.codegeneration.utils
from dolfinx import la
from dolfinx.fem import Form, IntegralType, form_cpp_class, functionspace
from dolfinx.mesh import create_unit_square

numba = pytest.importorskip("numba")
ufcx_signature = ffcx.codegeneration.utils.numba_ufcx_kernel_signature


# Helper intrinsic to cast void* to a typed pointer for custom_data
@numba.extending.intrinsic
def voidptr_to_float64_ptr(typingctx, src):
    """Cast a void pointer (CPointer(void)) to a float64 pointer.

    This function is used to access custom_data passed through the UFCx
    tabulate_tensor interface. Since custom_data is passed as void*, this
    intrinsic allows casting it to a typed float64 pointer for element access.

    Args:
        typingctx: The typing context.
        src: A void pointer (CPointer(void)) to cast.

    Returns:
        sig: A Numba signature returning CPointer(float64).
        codegen: A code generation function that performs the bitcast.

    Example:
        Inside a Numba cfunc kernel::

            typed_ptr = voidptr_to_float64_ptr(custom_data)
            scale = typed_ptr[0]  # Access first float64 value
    """
    # Accept CPointer(void) which shows as 'none*' in numba type system
    if isinstance(src, numba.types.CPointer) and src.dtype == numba.types.void:
        sig = numba.types.CPointer(numba.types.float64)(src)

        def codegen(context, builder, signature, args):
            [src] = args
            # Cast void* to float64*
            dst_type = context.get_value_type(numba.types.CPointer(numba.types.float64))
            return builder.bitcast(src, dst_type)

        return sig, codegen


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

    integrals = {IntegralType.cell: [(0, k1.address, cells, active_coeffs)]}
    formtype = form_cpp_class(dtype)
    L = Form(formtype([V._cpp_object], integrals, [], [], False, [], mesh=mesh._cpp_object))

    # Create custom_data with scale=1.0 first
    scale_value = np.array([1.0], dtype=dtype)
    scale_ptr = scale_value.ctypes.data
    L._cpp_object.set_custom_data(IntegralType.cell, 0, 0, scale_ptr)

    # Assemble with scale=1.0
    b1 = dolfinx.fem.assemble_vector(L)
    b1.scatter_reverse(la.InsertMode.add)
    norm1 = la.norm(b1)

    # Verify we can read back the custom_data pointer
    assert L._cpp_object.custom_data(IntegralType.cell, 0, 0) == scale_ptr

    # Update custom_data to scale=2.0
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

    integrals = {IntegralType.cell: [(0, k2.address, cells, active_coeffs)]}
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

    # Set custom_data with scale=1.0 first
    scale_value = np.array([1.0], dtype=dtype)
    a._cpp_object.set_custom_data(IntegralType.cell, 0, 0, scale_value.ctypes.data)

    # Assemble with scale=1.0
    A1 = dolfinx.fem.assemble_matrix(a)
    A1.scatter_reverse()
    norm1 = np.sqrt(A1.squared_norm())

    # Update custom_data to scale=2.0
    scale_value[0] = 2.0
    A2 = dolfinx.fem.assemble_matrix(a)
    A2.scatter_reverse()
    norm2 = np.sqrt(A2.squared_norm())

    # The norm with scale=2 should be 2x the norm with scale=1
    assert np.isclose(norm2, 2.0 * norm1)


@pytest.mark.parametrize("dtype", [np.float64])
def test_custom_data_default_nullptr(dtype):
    """Test that custom_data defaults to nullptr (0)."""
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

    integrals = {IntegralType.cell: [(0, tabulate_simple.address, cells, active_coeffs)]}
    formtype = form_cpp_class(dtype)
    L = Form(formtype([V._cpp_object], integrals, [], [], False, [], mesh=mesh._cpp_object))

    # custom_data should be None (std::nullopt) by default
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
    num_cells = mesh.topology.index_map(tdim).size_local + mesh.topology.index_map(tdim).num_ghosts
    cells = np.arange(num_cells, dtype=np.int32)
    active_coeffs = np.array([], dtype=np.int8)

    integrals = {IntegralType.cell: [(0, tabulate_with_struct.address, cells, active_coeffs)]}
    formtype = form_cpp_class(dtype)
    L = Form(formtype([V._cpp_object], integrals, [], [], False, [], mesh=mesh._cpp_object))

    # Create struct data: [scale=2.0, offset=0.5]
    struct_data = np.array([2.0, 0.5], dtype=dtype)
    L._cpp_object.set_custom_data(IntegralType.cell, 0, 0, struct_data.ctypes.data)

    b = dolfinx.fem.assemble_vector(L)
    b.scatter_reverse(la.InsertMode.add)

    # Verify the assembly used our custom values
    # The offset should contribute to each DOF
    assert la.norm(b) > 0
