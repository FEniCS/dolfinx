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
    num_cells = mesh.topology.index_map(tdim).size_local + mesh.topology.index_map(tdim).num_ghosts
    cells = np.arange(num_cells, dtype=np.int32)
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
    # The sum of all contributions equals 3 * num_cells (3 DOFs per cell, offset=1.0 each)
    total_contribution = np.sum(b_offset.array)
    assert np.isclose(total_contribution, 3.0 * num_cells)


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


@pytest.mark.parametrize("dtype", [np.float64])
def test_custom_data_per_cell_material(dtype):
    """Test custom_data with per-cell material properties using cell index.

    This test demonstrates the use case where a kernel needs to access
    cell-specific data. The cell index is now passed through entity_local_index
    for cell integrals, allowing the kernel to look up per-cell values.

    The custom_data pointer points to an array of values indexed by cell number.
    The kernel uses entity_local_index[0] (which contains the cell index for
    cell integrals) to look up the appropriate value.
    """
    xdtype = np.real(dtype(0)).dtype

    # Kernel that reads per-cell material property from custom_data
    # custom_data points to array: material_values[cell_index]
    # entity_local_index[0] contains the cell index for cell integrals
    @numba.cfunc(ufcx_signature(dtype, xdtype), nopython=True)
    def tabulate_per_cell_material(
        b_, w_, c_, coords_, entity_local_index, cell_orientation, custom_data
    ):
        b = numba.carray(b_, (3), dtype=dtype)
        coordinate_dofs = numba.carray(coords_, (3, 3), dtype=xdtype)

        # Cast void* to float64* - this points to per-cell material array
        material_array = voidptr_to_float64_ptr(custom_data)

        # entity_local_index[0] contains the cell index for cell integrals
        cell_idx_ptr = numba.carray(entity_local_index, (1,), dtype=np.int32)
        cell_idx = cell_idx_ptr[0]

        # Look up the material property for this specific cell
        material_value = material_array[cell_idx]

        x0, y0 = coordinate_dofs[0, :2]
        x1, y1 = coordinate_dofs[1, :2]
        x2, y2 = coordinate_dofs[2, :2]

        # 2x Element area Ae
        Ae = abs((x0 - x1) * (y2 - y1) - (y0 - y1) * (x2 - x1))
        b[:] = material_value * Ae / 6.0

    mesh = create_unit_square(MPI.COMM_WORLD, 4, 4, dtype=xdtype)
    V = functionspace(mesh, ("Lagrange", 1))

    tdim = mesh.topology.dim
    num_cells = mesh.topology.index_map(tdim).size_local + mesh.topology.index_map(tdim).num_ghosts
    cells = np.arange(num_cells, dtype=np.int32)
    active_coeffs = np.array([], dtype=np.int8)

    # Create per-cell material array - all cells have material=1.0
    material_values = np.ones(num_cells, dtype=dtype)

    # Pass pointer to material array as custom_data
    integrals = {
        IntegralType.cell: [
            (
                0,
                tabulate_per_cell_material.address,
                cells,
                active_coeffs,
                material_values.ctypes.data,
            )
        ]
    }
    formtype = form_cpp_class(dtype)
    L = Form(formtype([V._cpp_object], integrals, [], [], False, [], mesh=mesh._cpp_object))

    # Assemble with uniform material=1.0
    b_uniform = dolfinx.fem.assemble_vector(L)
    b_uniform.scatter_reverse(la.InsertMode.add)
    norm_uniform = la.norm(b_uniform)

    # Now set material=2.0 for all cells - should double the result
    material_values[:] = 2.0
    b_doubled = dolfinx.fem.assemble_vector(L)
    b_doubled.scatter_reverse(la.InsertMode.add)
    norm_doubled = la.norm(b_doubled)

    assert np.isclose(norm_doubled, 2.0 * norm_uniform)

    # Test heterogeneous material: first half of cells have material=1.0,
    # second half have material=3.0
    material_values[: num_cells // 2] = 1.0
    material_values[num_cells // 2 :] = 3.0
    b_hetero = dolfinx.fem.assemble_vector(L)
    b_hetero.scatter_reverse(la.InsertMode.add)

    # The result should be between uniform material=1.0 and material=3.0
    norm_hetero = la.norm(b_hetero)
    assert norm_uniform < norm_hetero < 3.0 * norm_uniform

    # Verify the total contribution matches expected:
    # The kernel computes Ae = 2*area (determinant formula), so:
    # Each cell contributes material[i] * Ae / 6 = material[i] * area / 3 to each of 3 DOFs
    # Total per cell = 3 * material[i] * area / 3 = material[i] * area
    # Total = sum_i (material[i] * area_i)
    # For uniform mesh on unit square with 4x4 grid: 32 triangles, each area 1/32
    total_contribution = np.sum(b_hetero.array)
    expected_sum = np.sum(material_values) * (1.0 / num_cells)
    assert np.isclose(total_contribution, expected_sum)


@pytest.mark.parametrize("dtype", [np.float64])
def test_custom_data_runtime_quadrature(dtype):
    """Test custom_data with per-cell runtime quadrature rules.

    This test demonstrates the key use case of passing runtime-computed
    quadrature points and weights to a kernel via custom_data. This enables:
    - Adaptive quadrature based on solution features
    - Different quadrature rules per cell (hp-adaptivity)
    - Quadrature rules computed from external sources

    The custom_data points to an array of quadrature rule data for each cell.
    The kernel uses entity_local_index[0] (cell index) to look up the
    quadrature rule for that specific cell.

    Layout per cell: [num_points, xi_0, eta_0, w_0, xi_1, eta_1, w_1, ...]
    We use fixed-size slots (max 3 points = 10 doubles per cell) for simplicity.
    """
    xdtype = np.real(dtype(0)).dtype

    # Fixed slot size: 1 (num_points) + 3*3 (max 3 points with xi, eta, weight) = 10
    SLOT_SIZE = 10

    # Kernel that integrates using per-cell quadrature from custom_data
    @numba.cfunc(ufcx_signature(dtype, xdtype), nopython=True)
    def tabulate_with_per_cell_quadrature(
        b_, w_, c_, coords_, entity_local_index, orientation, custom_data
    ):
        b = numba.carray(b_, (3), dtype=dtype)
        coordinate_dofs = numba.carray(coords_, (3, 3), dtype=xdtype)

        # Get cell index from entity_local_index
        cell_idx_ptr = numba.carray(entity_local_index, (1,), dtype=np.int32)
        cell_idx = cell_idx_ptr[0]

        # Read quadrature data for this cell from custom_data
        # Layout: quad_data[cell_idx * SLOT_SIZE : (cell_idx + 1) * SLOT_SIZE]
        quad_data = voidptr_to_float64_ptr(custom_data)
        slot_start = cell_idx * 10  # SLOT_SIZE = 10
        num_points = int(quad_data[slot_start])

        # Get physical coordinates of triangle vertices
        x0, y0 = coordinate_dofs[0, :2]
        x1, y1 = coordinate_dofs[1, :2]
        x2, y2 = coordinate_dofs[2, :2]

        # Jacobian determinant (2 * area)
        detJ = abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))

        # Integrate: sum over quadrature points
        b0 = dtype(0.0)
        b1 = dtype(0.0)
        b2 = dtype(0.0)

        for q in range(num_points):
            # Each quad point has: xi, eta, weight
            xi = quad_data[slot_start + 1 + q * 3]
            eta = quad_data[slot_start + 1 + q * 3 + 1]
            w = quad_data[slot_start + 1 + q * 3 + 2]

            # Basis function values at (xi, eta)
            # N0 = 1 - xi - eta, N1 = xi, N2 = eta
            N0 = 1.0 - xi - eta
            N1 = xi
            N2 = eta

            # Accumulate: integral of N_i over element
            b0 += N0 * w * detJ
            b1 += N1 * w * detJ
            b2 += N2 * w * detJ

        b[0] = b0
        b[1] = b1
        b[2] = b2

    mesh = create_unit_square(MPI.COMM_WORLD, 4, 4, dtype=xdtype)
    V = functionspace(mesh, ("Lagrange", 1))

    tdim = mesh.topology.dim
    num_cells = mesh.topology.index_map(tdim).size_local + mesh.topology.index_map(tdim).num_ghosts
    cells = np.arange(num_cells, dtype=np.int32)
    active_coeffs = np.array([], dtype=np.int8)
    formtype = form_cpp_class(dtype)

    # Define quadrature rules
    # 1-point centroid rule (exact for degree 1)
    quad_1pt = [1.0, 1.0 / 3.0, 1.0 / 3.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # 3-point Gauss rule (exact for degree 2)
    quad_3pt = [
        3.0,
        1.0 / 6.0,
        1.0 / 6.0,
        1.0 / 6.0,
        2.0 / 3.0,
        1.0 / 6.0,
        1.0 / 6.0,
        1.0 / 6.0,
        2.0 / 3.0,
        1.0 / 6.0,
    ]

    # Test 1: All cells use 1-point rule
    quad_data_uniform_1pt = np.zeros((num_cells, SLOT_SIZE), dtype=dtype)
    for c in range(num_cells):
        quad_data_uniform_1pt[c, :] = quad_1pt

    integrals_1pt = {
        IntegralType.cell: [
            (
                0,
                tabulate_with_per_cell_quadrature.address,
                cells,
                active_coeffs,
                quad_data_uniform_1pt.ctypes.data,
            )
        ]
    }
    L_1pt = Form(formtype([V._cpp_object], integrals_1pt, [], [], False, [], mesh=mesh._cpp_object))

    b_1pt = dolfinx.fem.assemble_vector(L_1pt)
    b_1pt.scatter_reverse(la.InsertMode.add)
    total_1pt = np.sum(b_1pt.array)

    # Test 2: All cells use 3-point rule
    quad_data_uniform_3pt = np.zeros((num_cells, SLOT_SIZE), dtype=dtype)
    for c in range(num_cells):
        quad_data_uniform_3pt[c, :] = quad_3pt

    integrals_3pt = {
        IntegralType.cell: [
            (
                0,
                tabulate_with_per_cell_quadrature.address,
                cells,
                active_coeffs,
                quad_data_uniform_3pt.ctypes.data,
            )
        ]
    }
    L_3pt = Form(formtype([V._cpp_object], integrals_3pt, [], [], False, [], mesh=mesh._cpp_object))

    b_3pt = dolfinx.fem.assemble_vector(L_3pt)
    b_3pt.scatter_reverse(la.InsertMode.add)
    total_3pt = np.sum(b_3pt.array)

    # Both should give exact integral = 1.0 for linear basis functions
    assert np.isclose(total_1pt, 1.0, rtol=1e-10)
    assert np.isclose(total_3pt, 1.0, rtol=1e-10)

    # Test 3: Mixed quadrature - first half uses 1-point, second half uses 3-point
    # This simulates adaptive quadrature where some cells need higher accuracy
    quad_data_mixed = np.zeros((num_cells, SLOT_SIZE), dtype=dtype)
    for c in range(num_cells):
        if c < num_cells // 2:
            quad_data_mixed[c, :] = quad_1pt  # Low-order cells
        else:
            quad_data_mixed[c, :] = quad_3pt  # High-order cells

    integrals_mixed = {
        IntegralType.cell: [
            (
                0,
                tabulate_with_per_cell_quadrature.address,
                cells,
                active_coeffs,
                quad_data_mixed.ctypes.data,
            )
        ]
    }
    L_mixed = Form(
        formtype([V._cpp_object], integrals_mixed, [], [], False, [], mesh=mesh._cpp_object)
    )

    b_mixed = dolfinx.fem.assemble_vector(L_mixed)
    b_mixed.scatter_reverse(la.InsertMode.add)
    total_mixed = np.sum(b_mixed.array)

    # Mixed quadrature should also give exact result for linear basis functions
    assert np.isclose(total_mixed, 1.0, rtol=1e-10)

    # Test 4: Verify per-cell access works by using scaled weights
    # Cells in first half: weight scaled by 2.0, second half: normal weight
    # This should NOT give exact integral but demonstrates per-cell differentiation
    quad_data_scaled = np.zeros((num_cells, SLOT_SIZE), dtype=dtype)
    for c in range(num_cells):
        if c < num_cells // 2:
            # Scaled 1-point rule (weight = 1.0 instead of 0.5)
            quad_data_scaled[c, :] = [1.0, 1.0 / 3.0, 1.0 / 3.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        else:
            # Normal 1-point rule (weight = 0.5)
            quad_data_scaled[c, :] = quad_1pt

    integrals_scaled = {
        IntegralType.cell: [
            (
                0,
                tabulate_with_per_cell_quadrature.address,
                cells,
                active_coeffs,
                quad_data_scaled.ctypes.data,
            )
        ]
    }
    L_scaled = Form(
        formtype([V._cpp_object], integrals_scaled, [], [], False, [], mesh=mesh._cpp_object)
    )

    b_scaled = dolfinx.fem.assemble_vector(L_scaled)
    b_scaled.scatter_reverse(la.InsertMode.add)
    total_scaled = np.sum(b_scaled.array)

    # First half contributes 2x, second half contributes 1x
    # Total = 0.5 * 2 + 0.5 * 1 = 1.5
    assert np.isclose(total_scaled, 1.5, rtol=1e-10)
