# Copyright (C) 2018-2025 Garth N. Wells, Jørgen S. Dokken and Paul T. Kühner
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for assembly."""

import os

from mpi4py import MPI

import numpy as np
import pytest
import scipy.sparse

import basix
import dolfinx.cpp
import ufl
from dolfinx import cpp as _cpp
from dolfinx import fem, la, mesh
from dolfinx.fem import (
    Constant,
    Function,
    assemble_matrix,
    assemble_scalar,
    assemble_vector,
    form,
    functionspace,
    pack_coefficients,
    pack_constants,
)
from dolfinx.mesh import (
    GhostMode,
    create_unit_square,
    locate_entities,
    meshtags,
)
from ufl import derivative, ds, dx, inner

dtype_parametrize = pytest.mark.parametrize(
    "dtype",
    [
        np.float32,
        np.float64,
        pytest.param(np.complex64, marks=pytest.mark.xfail_win32_complex),
        pytest.param(np.complex128, marks=pytest.mark.xfail_win32_complex),
    ],
)


@pytest.mark.parametrize("mode", [GhostMode.none, GhostMode.shared_facet])
@dtype_parametrize
def test_assemble_functional_dx(mode, dtype):
    xtype = dtype(0).real.dtype
    mesh = create_unit_square(MPI.COMM_WORLD, 12, 12, ghost_mode=mode, dtype=xtype)
    M = form(1.0 * dx(domain=mesh), dtype=dtype)
    value = assemble_scalar(M)
    value = mesh.comm.allreduce(value, op=MPI.SUM)
    assert value == pytest.approx(1.0, 1e-5)
    x = ufl.SpatialCoordinate(mesh)
    M = form(x[0] * dx(domain=mesh), dtype=dtype)
    value = assemble_scalar(M)
    value = mesh.comm.allreduce(value, op=MPI.SUM)
    assert value == pytest.approx(0.5, 1e-6)


@pytest.mark.parametrize("mode", [GhostMode.none, GhostMode.shared_facet])
@dtype_parametrize
def test_assemble_functional_ds(mode, dtype):
    xtype = dtype(0).real.dtype
    mesh = create_unit_square(MPI.COMM_WORLD, 12, 12, ghost_mode=mode, dtype=xtype)
    M = form(1.0 * ds(domain=mesh), dtype=dtype)
    value = assemble_scalar(M)
    value = mesh.comm.allreduce(value, op=MPI.SUM)
    assert value == pytest.approx(4.0, 1e-6)


@dtype_parametrize
def test_assemble_derivatives(dtype):
    """Test the original_coefficient_positions.

    Positions  may change under differentiation (some coefficients and
    constants are eliminated.
    """
    mesh = create_unit_square(MPI.COMM_WORLD, 12, 12, dtype=dtype(0).real.dtype)
    Q = functionspace(mesh, ("Lagrange", 1))
    u = Function(Q, dtype=dtype)
    v = ufl.TestFunction(Q)
    du = ufl.TrialFunction(Q)
    b = Function(Q, dtype=dtype)
    c1 = Constant(mesh, np.array([[1.0, 0.0], [3.0, 4.0]], dtype=dtype))
    c2 = Constant(mesh, dtype(2.0))

    b.x.array[:] = 2.0

    # derivative eliminates 'u' and 'c1'
    L = ufl.inner(c1, c1) * v * dx + c2 * b * inner(u, v) * dx
    a = form(derivative(L, u, du), dtype=dtype)

    A1 = assemble_matrix(a)
    A1.scatter_reverse()
    a = form(c2 * b * inner(du, v) * dx, dtype=dtype)
    A2 = assemble_matrix(a)
    A2.scatter_reverse()
    assert np.allclose(A1.data, A2.data)


@pytest.mark.parametrize("mode", [GhostMode.none, GhostMode.shared_facet])
@dtype_parametrize
def test_basic_assembly(mode, dtype):
    mesh = create_unit_square(MPI.COMM_WORLD, 12, 12, ghost_mode=mode, dtype=dtype(0).real.dtype)
    V = functionspace(mesh, ("Lagrange", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

    f = Function(V, dtype=dtype)
    f.x.array[:] = 10.0
    a = inner(f * u, v) * dx + inner(u, v) * ds
    L = inner(f, v) * dx + inner(2.0, v) * ds
    a, L = form(a, dtype=dtype), form(L, dtype=dtype)

    # Initial assembly
    A = assemble_matrix(a)
    A.scatter_reverse()
    assert isinstance(A, la.MatrixCSR)
    b = assemble_vector(L)
    b.scatter_reverse(la.InsertMode.add)
    assert isinstance(b, la.Vector)

    # Second assembly
    normA = A.squared_norm()
    A.set_value(0)
    A = assemble_matrix(A, a)
    A.scatter_reverse()
    assert isinstance(A, la.MatrixCSR)
    assert normA == pytest.approx(A.squared_norm())
    normb = la.norm(b)
    b.array[:] = 0
    assemble_vector(b.array, L)
    b.scatter_reverse(la.InsertMode.add)
    assert normb == pytest.approx(la.norm(b))

    # Vector re-assembly - no zeroing (but need to zero ghost entries)
    b.array[b.index_map.size_local * b.block_size :] = 0
    assemble_vector(b.array, L)
    b.scatter_reverse(la.InsertMode.add)
    assert 2 * normb == pytest.approx(la.norm(b))

    # Matrix re-assembly (no zeroing)
    assemble_matrix(A, a)
    A.scatter_reverse()
    assert 4 * normA == pytest.approx(A.squared_norm())


@pytest.mark.parametrize("mode", [GhostMode.none, GhostMode.shared_facet])
@dtype_parametrize
def test_basic_assembly_constant(mode, dtype):
    """Tests assembly with Constant.

    The following test should be sensitive to order of flattening the
    matrix-valued constant.
    """
    xtype = dtype(0).real.dtype
    mesh = create_unit_square(MPI.COMM_WORLD, 5, 5, ghost_mode=mode, dtype=xtype)
    V = functionspace(mesh, ("Lagrange", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

    c = Constant(mesh, np.array([[1.0, 2.0], [5.0, 3.0]], dtype=dtype))

    a = inner(c[1, 0] * u, v) * dx + inner(c[1, 0] * u, v) * ds
    L = inner(c[1, 0], v) * dx + inner(c[1, 0], v) * ds
    a, L = form(a, dtype=dtype), form(L, dtype=dtype)

    # Initial assembly
    A1 = assemble_matrix(a)
    A1.scatter_reverse()

    b1 = assemble_vector(L)
    b1.scatter_reverse(la.InsertMode.add)

    c.value = [[1.0, 2.0], [3.0, 4.0]]

    A2 = assemble_matrix(a)
    A2.scatter_reverse()
    assert np.linalg.norm(A1.data * 3.0 - A2.data * 5.0) == pytest.approx(0.0, abs=1.0e-5)

    b2 = assemble_vector(L)
    b2.scatter_reverse(la.InsertMode.add)
    assert np.linalg.norm(b1.array * 3.0 - b2.array * 5.0) == pytest.approx(0.0, abs=1.0e-5)


def test_lambda_assembler():
    """Tests assembly with a lambda function."""
    mesh = create_unit_square(MPI.COMM_WORLD, 5, 5)
    V = functionspace(mesh, ("Lagrange", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

    a = inner(u, v) * dx

    # Initial assembly
    a_form = form(a)

    rdata = []
    cdata = []
    vdata = []

    def mat_insert(rows, cols, vals):
        vdata.append(list(vals))
        rdata.append(list(np.repeat(rows, len(cols))))
        cdata.append(list(np.tile(cols, len(rows))))
        return 0

    _cpp.fem.assemble_matrix(mat_insert, a_form._cpp_object, [])
    vdata = np.array(vdata).flatten()
    cdata = np.array(cdata).flatten()
    rdata = np.array(rdata).flatten()
    mat = scipy.sparse.coo_matrix((vdata, (rdata, cdata)))
    v = np.ones(mat.shape[1])
    s = MPI.COMM_WORLD.allreduce(mat.dot(v).sum(), MPI.SUM)
    assert np.isclose(s, 1.0)


@pytest.mark.xfail_win32_complex
def test_vector_types():
    """Assemble form using different types."""
    mesh0 = create_unit_square(MPI.COMM_WORLD, 3, 5, dtype=np.float32)
    mesh1 = create_unit_square(MPI.COMM_WORLD, 3, 5, dtype=np.float64)
    V0, V1 = functionspace(mesh0, ("Lagrange", 3)), functionspace(mesh1, ("Lagrange", 3))
    v0, v1 = ufl.TestFunction(V0), ufl.TestFunction(V1)

    c = Constant(mesh1, np.float64(1))
    L = inner(c, v1) * ufl.dx
    x0 = la.vector(V1.dofmap.index_map, V1.dofmap.index_map_bs, dtype=np.float64)
    L = form(L, dtype=x0.array.dtype)
    c0 = pack_constants(L)
    c1 = pack_coefficients(L)
    assemble_vector(x0.array, L, c0, c1)
    x0.scatter_reverse(la.InsertMode.add)

    c = Constant(mesh1, np.complex128(1))
    L = inner(c, v1) * ufl.dx
    x1 = la.vector(V1.dofmap.index_map, V1.dofmap.index_map_bs, dtype=np.complex128)
    L = form(L, dtype=x1.array.dtype)
    c0 = pack_constants(L)
    c1 = pack_coefficients(L)
    assemble_vector(x1.array, L, c0, c1)
    x1.scatter_reverse(la.InsertMode.add)

    c = Constant(mesh0, np.float32(1))
    L = inner(c, v0) * ufl.dx
    x2 = la.vector(V0.dofmap.index_map, V0.dofmap.index_map_bs, dtype=np.float32)
    L = form(L, dtype=x2.array.dtype)
    c0 = pack_constants(L)
    c1 = pack_coefficients(L)
    assemble_vector(x2.array, L, c0, c1)
    x2.scatter_reverse(la.InsertMode.add)

    assert np.linalg.norm(x0.array - x1.array) == pytest.approx(0.0)
    assert np.linalg.norm(x0.array - x2.array) == pytest.approx(0.0, abs=1e-7)


@dtype_parametrize
@pytest.mark.parametrize("method", ["degree", "metadata"])
def test_mixed_quadrature(dtype, method):
    xtype = dtype(0).real.dtype
    mesh = create_unit_square(MPI.COMM_WORLD, 12, 12, dtype=xtype)

    V = functionspace(mesh, ("Lagrange", 1))
    u = Function(V, dtype=dtype)
    u.interpolate(lambda x: x[0])

    tol = 500 * np.finfo(dtype).eps
    num_cells_local = (
        mesh.topology.index_map(mesh.topology.dim).size_local
        + mesh.topology.index_map(mesh.topology.dim).num_ghosts
    )
    values = np.full(num_cells_local, 1, dtype=np.int32)
    left_cells = locate_entities(mesh, mesh.topology.dim, lambda x: x[0] <= 0.5 + tol)
    values[left_cells] = 2
    top_cells = locate_entities(mesh, mesh.topology.dim, lambda x: x[1] >= 0.5 - tol)
    values[top_cells] = 3
    ct = meshtags(mesh, mesh.topology.dim, np.arange(num_cells_local, dtype=np.int32), values)

    dx = ufl.Measure("dx", domain=mesh, subdomain_data=ct)

    if method == "degree":
        dx_1 = dx(subdomain_id=(1,), degree=1)
        dx_2 = dx(subdomain_id=(1, 2), degree=2)
        dx_3 = dx(subdomain_id=(2, 3), degree=3)
    elif method == "metadata":
        dx_1 = dx(subdomain_id=(1,), metadata={"quadrature_degree": 1})
        dx_2 = dx(subdomain_id=(1, 2), metadata={"quadrature_degree": 2})
        dx_3 = dx(subdomain_id=(2, 3), metadata={"quadrature_degree": 3})
    else:
        raise ValueError(f"Invalid method {method}")
    form_1 = u * dx_1
    form_2 = u * dx_2
    form_3 = u * dx_3
    summed_form = form_1 + form_2 + form_3

    compiled_forms = form([form_1, form_2, form_3], dtype=dtype)
    local_contributions = 0
    for compiled_form in compiled_forms:
        local_contributions += assemble_scalar(compiled_form)
    global_contribution = mesh.comm.allreduce(local_contributions, op=MPI.SUM)

    compiled_form = form(summed_form, dtype=dtype)
    local_sum = assemble_scalar(compiled_form)
    global_sum = mesh.comm.allreduce(local_sum, op=MPI.SUM)
    assert np.isclose(global_contribution, global_sum, rtol=tol, atol=tol)


def vertex_to_dof_map(V):
    """Create a map from the vertices of the mesh to the corresponding degree of freedom."""
    mesh = V.mesh
    num_vertices_per_cell = dolfinx.cpp.mesh.cell_num_entities(mesh.topology.cell_type, 0)

    dof_layout2 = np.empty((num_vertices_per_cell,), dtype=np.int32)
    for i in range(num_vertices_per_cell):
        var = V.dofmap.dof_layout.entity_dofs(0, i)
        assert len(var) == 1
        dof_layout2[i] = var[0]

    num_vertices = mesh.topology.index_map(0).size_local + mesh.topology.index_map(0).num_ghosts

    c_to_v = mesh.topology.connectivity(mesh.topology.dim, 0)
    assert (
        c_to_v.num_nodes == 0
        or (c_to_v.offsets[1:] - c_to_v.offsets[:-1] == c_to_v.offsets[1]).all()
    ), "Single cell type supported"

    vertex_to_dof_map = np.empty(num_vertices, dtype=np.int32)
    vertex_to_dof_map[c_to_v.array] = V.dofmap.list[:, dof_layout2].reshape(-1)
    return vertex_to_dof_map


@pytest.mark.parametrize(
    "cell_type",
    [
        mesh.CellType.interval,
        mesh.CellType.triangle,
        mesh.CellType.quadrilateral,
        mesh.CellType.tetrahedron,
        # mesh.CellType.pyramid,
        mesh.CellType.prism,
        mesh.CellType.hexahedron,
    ],
)
@pytest.mark.parametrize("ghost_mode", [mesh.GhostMode.none, mesh.GhostMode.shared_facet])
@pytest.mark.parametrize(
    "dtype",
    [
        np.float32,
        np.float64,
        pytest.param(
            np.complex64,
            marks=pytest.mark.skipif(
                os.name == "nt", reason="win32 platform does not support C99 _Complex numbers"
            ),
        ),
        pytest.param(
            np.complex128,
            marks=pytest.mark.skipif(
                os.name == "nt", reason="win32 platform does not support C99 _Complex numbers"
            ),
        ),
    ],
)
def test_vertex_integral_rank_0(cell_type, ghost_mode, dtype):
    comm = MPI.COMM_WORLD
    rdtype = np.real(dtype(0)).dtype

    msh = None
    cell_dim = mesh.cell_dim(cell_type)
    if cell_dim == 1:
        msh = mesh.create_unit_interval(comm, 4, dtype=rdtype, ghost_mode=ghost_mode)
    elif cell_dim == 2:
        msh = mesh.create_unit_square(
            comm, 4, 4, cell_type=cell_type, dtype=rdtype, ghost_mode=ghost_mode
        )
    elif cell_dim == 3:
        msh = mesh.create_unit_cube(
            comm, 4, 4, 4, cell_type=cell_type, dtype=rdtype, ghost_mode=ghost_mode
        )
    else:
        raise RuntimeError("Bad dimension")

    vertex_map = msh.topology.index_map(0)

    def check_vertex_integral_against_sum(form, vertices, weighted=False):
        """Weighting assumes the vertex integral to be weighted by a P1 function, each vertex value
        corresponding to its global index.
        """
        weights = vertex_map.local_to_global(vertices) if weighted else np.ones_like(vertices)
        expected_value_l = np.sum(msh.geometry.x[vertices, 0] * weights)
        value_l = fem.assemble_scalar(fem.form(form, dtype=dtype))
        assert expected_value_l == pytest.approx(value_l, abs=5e4 * np.finfo(rdtype).eps)

        expected_value = comm.allreduce(expected_value_l)
        value = comm.allreduce(value_l)
        assert expected_value == pytest.approx(value, abs=5e4 * np.finfo(rdtype).eps)

    num_vertices = vertex_map.size_local
    x = ufl.SpatialCoordinate(msh)

    # Full domain
    check_vertex_integral_against_sum(x[0] * ufl.dP, np.arange(num_vertices))

    # Split domain into left half of vertices (1) and right half of vertices (2)
    vertices_1 = mesh.locate_entities(msh, 0, lambda x: x[0] <= 0.5)
    vertices_1 = vertices_1[vertices_1 < num_vertices]
    vertices_2 = mesh.locate_entities(msh, 0, lambda x: x[0] > 0.5)
    vertices_2 = vertices_2[vertices_2 < num_vertices]

    tags = np.full(num_vertices, 1)
    tags[vertices_2] = 2
    vertices = np.arange(0, num_vertices, dtype=np.int32)
    meshtags = mesh.meshtags(msh, 0, vertices, tags)

    dP = ufl.Measure("dP", domain=msh, subdomain_data=meshtags)

    # Combinations of sub domains
    check_vertex_integral_against_sum(x[0] * dP(1), vertices_1)
    check_vertex_integral_against_sum(x[0] * dP(2), vertices_2)
    check_vertex_integral_against_sum(x[0] * (dP(1) + dP(2)), np.arange(num_vertices))

    V = fem.functionspace(msh, ("P", 1))
    u = fem.Function(V, dtype=dtype)
    vertex_to_dof = vertex_to_dof_map(V)
    vertices = np.arange(num_vertices + vertex_map.num_ghosts)
    u.x.array[vertex_to_dof[vertices]] = vertex_map.local_to_global(vertices)

    check_vertex_integral_against_sum(u * x[0] * ufl.dP, np.arange(num_vertices), True)
    check_vertex_integral_against_sum(u * x[0] * dP(1), vertices_1, True)
    check_vertex_integral_against_sum(u * x[0] * dP(2), vertices_2, True)
    check_vertex_integral_against_sum(u * x[0] * (dP(1) + dP(2)), np.arange(num_vertices), True)

    # Check custom packing
    if cell_type is mesh.CellType.prism:
        return

    msh.topology.create_entities(1)
    msh.topology.create_connectivity(cell_dim - 1, cell_dim)

    v_to_c = msh.topology.connectivity(0, cell_dim)
    c_to_v = msh.topology.connectivity(cell_dim, 0)

    cell_vertex_pairs = np.array([], dtype=np.int32)
    for v in range(num_vertices):
        c = v_to_c.links(v)[0]
        v_l = np.where(c_to_v.links(c) == v)[0]
        cell_vertex_pairs = np.append(cell_vertex_pairs, [c, *v_l])

    # a) With subdomain_data
    check_vertex_integral_against_sum(
        x[0] * ufl.dP(domain=msh, subdomain_data=[(1, cell_vertex_pairs)], subdomain_id=1),
        np.arange(num_vertices),
    )

    # b) With create_form
    vertices = np.arange(num_vertices)
    fem.compute_integration_domains(fem.IntegralType.exterior_facet, msh.topology, vertices)
    subdomains = {fem.IntegralType.exterior_facet: [(0, cell_vertex_pairs)]}

    compiled_form = fem.compile_form(
        comm, x[0] * ufl.dP, form_compiler_options={"scalar_type": dtype}
    )
    form = fem.create_form(compiled_form, [], msh, subdomains, {}, {}, [])
    expected_value_l = np.sum(msh.geometry.x[vertices, 0])
    value_l = fem.assemble_scalar(form)
    assert expected_value_l == pytest.approx(value_l, abs=5e4 * np.finfo(rdtype).eps)


@pytest.mark.parametrize(
    "cell_type",
    [
        mesh.CellType.interval,
        mesh.CellType.triangle,
        mesh.CellType.quadrilateral,
        mesh.CellType.tetrahedron,
        # mesh.CellType.pyramid,
        mesh.CellType.prism,
        mesh.CellType.hexahedron,
    ],
)
@pytest.mark.parametrize("ghost_mode", [mesh.GhostMode.none, mesh.GhostMode.shared_facet])
@pytest.mark.parametrize(
    "dtype",
    [
        np.float32,
        np.float64,
        pytest.param(
            np.complex64,
            marks=pytest.mark.skipif(
                os.name == "nt", reason="win32 platform does not support C99 _Complex numbers"
            ),
        ),
        pytest.param(
            np.complex128,
            marks=pytest.mark.skipif(
                os.name == "nt", reason="win32 platform does not support C99 _Complex numbers"
            ),
        ),
    ],
)
def test_vertex_integral_rank_1(cell_type, ghost_mode, dtype):
    comm = MPI.COMM_WORLD
    rdtype = np.real(dtype(0)).dtype

    msh = None
    cell_dim = mesh.cell_dim(cell_type)
    if cell_dim == 1:
        msh = mesh.create_unit_interval(comm, 4, ghost_mode=ghost_mode, dtype=rdtype)
    elif cell_dim == 2:
        msh = mesh.create_unit_square(
            comm, 4, 4, cell_type=cell_type, ghost_mode=ghost_mode, dtype=rdtype
        )
    elif cell_dim == 3:
        msh = mesh.create_unit_cube(
            comm, 4, 4, 4, cell_type=cell_type, ghost_mode=ghost_mode, dtype=rdtype
        )
    else:
        raise RuntimeError("Bad dimension")

    vertex_map = msh.topology.index_map(0)
    num_vertices = vertex_map.size_local

    def check_vertex_integral_against_sum(form, vertices, weighted=False):
        """Weighting assumes the vertex integral to be weighted by a P1 function, each vertex value
        corresponding to its global index.
        """
        weights = vertex_map.local_to_global(vertices) if weighted else np.ones_like(vertices)
        expected_value_l = np.zeros(num_vertices, dtype=rdtype)
        expected_value_l[vertices] = msh.geometry.x[vertices, 0] * weights
        value_l = fem.assemble_vector(fem.form(form, dtype=dtype))
        equal_l = np.allclose(
            expected_value_l, np.real(value_l.array[:num_vertices]), atol=1e3 * np.finfo(rdtype).eps
        )
        assert equal_l
        assert comm.allreduce(equal_l, MPI.BAND)

    x = ufl.SpatialCoordinate(msh)
    V = fem.functionspace(msh, ("P", 1))
    v = ufl.conj(ufl.TestFunction(V))

    # Full domain
    check_vertex_integral_against_sum(x[0] * v * ufl.dP, np.arange(num_vertices))

    # Split domain into left half of vertices (1) and right half of vertices (2)
    vertices_1 = mesh.locate_entities(msh, 0, lambda x: x[0] <= 0.5)
    vertices_1 = vertices_1[vertices_1 < num_vertices]
    vertices_2 = mesh.locate_entities(msh, 0, lambda x: x[0] > 0.5)
    vertices_2 = vertices_2[vertices_2 < num_vertices]

    tags = np.full(num_vertices, 1)
    tags[vertices_2] = 2
    vertices = np.arange(0, num_vertices, dtype=np.int32)
    meshtags = mesh.meshtags(msh, 0, vertices, tags)

    dP = ufl.Measure("dP", domain=msh, subdomain_data=meshtags)

    check_vertex_integral_against_sum(x[0] * v * dP(1), vertices_1)
    check_vertex_integral_against_sum(x[0] * v * dP(2), vertices_2)
    check_vertex_integral_against_sum(x[0] * v * (dP(1) + dP(2)), np.arange(num_vertices))

    V = fem.functionspace(msh, ("P", 1))
    u = fem.Function(V, dtype=dtype)
    u.x.array[:] = vertex_map.local_to_global(np.arange(num_vertices + vertex_map.num_ghosts))
    vertex_to_dof = vertex_to_dof_map(V)
    vertices = np.arange(num_vertices + vertex_map.num_ghosts)
    u.x.array[vertex_to_dof[vertices]] = vertex_map.local_to_global(vertices)

    check_vertex_integral_against_sum(u * x[0] * v * ufl.dP, np.arange(num_vertices), True)
    check_vertex_integral_against_sum(u * x[0] * v * dP(1), vertices_1, True)
    check_vertex_integral_against_sum(u * x[0] * v * dP(2), vertices_2, True)
    check_vertex_integral_against_sum(u * x[0] * v * (dP(1) + dP(2)), np.arange(num_vertices), True)

    # Check custom packing
    if cell_type is mesh.CellType.prism:
        return

    msh.topology.create_entities(1)
    msh.topology.create_connectivity(cell_dim - 1, cell_dim)

    v_to_c = msh.topology.connectivity(0, cell_dim)
    c_to_v = msh.topology.connectivity(cell_dim, 0)

    cell_vertex_pairs = np.array([], dtype=np.int32)
    for v in range(num_vertices):
        c = v_to_c.links(v)[0]
        v_l = np.where(c_to_v.links(c) == v)[0]
        cell_vertex_pairs = np.append(cell_vertex_pairs, [c, *v_l])

    # a) With subdomain_data
    v = ufl.conj(ufl.TestFunction(V))
    check_vertex_integral_against_sum(
        x[0] * v * ufl.dP(domain=msh, subdomain_data=[(1, cell_vertex_pairs)], subdomain_id=1),
        np.arange(num_vertices),
    )

    # b) With create_form
    vertices = np.arange(num_vertices)
    fem.compute_integration_domains(fem.IntegralType.exterior_facet, msh.topology, vertices)
    subdomains = {fem.IntegralType.exterior_facet: [(0, cell_vertex_pairs)]}

    compiled_form = fem.compile_form(
        comm, x[0] * v * ufl.dP, form_compiler_options={"scalar_type": dtype}
    )
    form = fem.create_form(compiled_form, [V], msh, subdomains, {}, {}, [])
    expected_value_l = np.sum(msh.geometry.x[vertices, 0])
    expected_value_l = np.zeros(num_vertices, dtype=rdtype)
    expected_value_l[vertices] = msh.geometry.x[vertices, 0]
    value_l = fem.assemble_vector(form)
    assert expected_value_l == pytest.approx(
        value_l.array[: expected_value_l.size], abs=5e4 * np.finfo(rdtype).eps
    )


@pytest.mark.parametrize(
    "cell_type",
    [
        mesh.CellType.triangle,
        mesh.CellType.quadrilateral,
    ],
)
@pytest.mark.parametrize("ghost_mode", [mesh.GhostMode.none, mesh.GhostMode.shared_facet])
@pytest.mark.parametrize(
    "dtype",
    [
        np.float32,
        np.float64,
        pytest.param(
            np.complex64,
            marks=pytest.mark.skipif(
                os.name == "nt", reason="win32 platform does not support C99 _Complex numbers"
            ),
        ),
        pytest.param(
            np.complex128,
            marks=pytest.mark.skipif(
                os.name == "nt", reason="win32 platform does not support C99 _Complex numbers"
            ),
        ),
    ],
)
def test_ridge_integrals_rank1_2D(cell_type, ghost_mode, dtype):
    comm = MPI.COMM_WORLD
    rdtype = np.real(dtype(0)).dtype

    msh = None
    msh = mesh.create_unit_square(
        comm, 4, 4, cell_type=cell_type, ghost_mode=ghost_mode, dtype=rdtype
    )
    gdim = msh.geometry.dim
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 3))

    x = ufl.SpatialCoordinate(msh)
    u = ufl.TestFunction(V)

    integrand = ufl.conj(u) * x[gdim - 1]
    dr = ufl.Measure("dr", domain=msh)
    F = dolfinx.fem.form(integrand * dr, dtype=dtype)
    b = dolfinx.fem.assemble_vector(F)

    dP = ufl.Measure("dP", domain=msh)
    Fp = dolfinx.fem.form(integrand * dP, dtype=dtype)
    b_p = dolfinx.fem.assemble_vector(Fp)
    tol = np.finfo(rdtype).eps
    np.testing.assert_allclose(b.array, b_p.array, atol=tol, rtol=tol)


@pytest.mark.parametrize(
    "cell_type",
    [
        mesh.CellType.triangle,
        mesh.CellType.quadrilateral,
        mesh.CellType.tetrahedron,
        # mesh.CellType.pyramid,
        # mesh.CellType.prism,
        mesh.CellType.hexahedron,
    ],
)
@pytest.mark.parametrize("ghost_mode", [mesh.GhostMode.none, mesh.GhostMode.shared_facet])
@pytest.mark.parametrize(
    "dtype",
    [
        np.float32,
        np.float64,
        pytest.param(
            np.complex64,
            marks=pytest.mark.skipif(
                os.name == "nt", reason="win32 platform does not support C99 _Complex numbers"
            ),
        ),
        pytest.param(
            np.complex128,
            marks=pytest.mark.skipif(
                os.name == "nt", reason="win32 platform does not support C99 _Complex numbers"
            ),
        ),
    ],
)
def test_ridge_integrals_rank0(cell_type, ghost_mode, dtype):
    comm = MPI.COMM_WORLD
    rdtype = np.real(dtype(0)).dtype

    msh = None
    cell_dim = mesh.cell_dim(cell_type)
    if cell_dim == 2:
        msh = mesh.create_unit_square(
            comm, 4, 4, cell_type=cell_type, ghost_mode=ghost_mode, dtype=rdtype
        )
    elif cell_dim == 3:
        msh = mesh.create_unit_cube(
            comm, 4, 4, 4, cell_type=cell_type, ghost_mode=ghost_mode, dtype=rdtype
        )
    else:
        raise RuntimeError("Bad dimension")
    gdim = msh.geometry.dim

    x = ufl.SpatialCoordinate(msh)

    def marked_ridges(x):
        return np.isclose(x[0], 1) & np.isclose(x[1], 0.5)

    exterior_ridges = dolfinx.mesh.locate_entities_boundary(
        msh, msh.topology.dim - 2, marked_ridges
    )
    et = dolfinx.mesh.meshtags(
        msh, msh.topology.dim - 2, exterior_ridges, np.full_like(exterior_ridges, 33)
    )
    dr = ufl.Measure("dr", domain=msh, subdomain_data=et, subdomain_id=33)
    integrand = x[0] + x[1] + x[gdim - 1] ** 2
    J_compiled = dolfinx.fem.form(integrand * dr, dtype=dtype)
    J = dolfinx.fem.assemble_scalar(J_compiled)
    J_sum = msh.comm.allreduce(J, op=MPI.SUM)
    if cell_dim == 2:
        ref_sol = 1 + 1 / 2 + (1 / 2) ** 2
    elif cell_dim == 3:
        ref_sol = 1 + 1 / 2 + 1 / 3
    else:
        raise RuntimeError("Bad dimension")
    tol = 50 * np.finfo(rdtype).eps
    assert np.isclose(J_sum, ref_sol, atol=tol, rtol=tol)


@pytest.mark.parametrize("coefficient", [True, False])
@pytest.mark.parametrize(
    "cell_type",
    [
        mesh.CellType.tetrahedron,
        # mesh.CellType.pyramid,
        # mesh.CellType.prism,
        mesh.CellType.hexahedron,
    ],
)
@pytest.mark.parametrize("ghost_mode", [mesh.GhostMode.none, mesh.GhostMode.shared_facet])
@pytest.mark.parametrize(
    "dtype",
    [
        np.float32,
        np.float64,
        pytest.param(
            np.complex64,
            marks=pytest.mark.skipif(
                os.name == "nt", reason="win32 platform does not support C99 _Complex numbers"
            ),
        ),
        pytest.param(
            np.complex128,
            marks=pytest.mark.skipif(
                os.name == "nt", reason="win32 platform does not support C99 _Complex numbers"
            ),
        ),
    ],
)
def test_ridge_integrals_rank1_3D(cell_type, ghost_mode, dtype, coefficient):
    comm = MPI.COMM_WORLD
    rdtype = np.real(dtype(0)).dtype

    N = 4
    msh = mesh.create_unit_cube(
        comm, N, N, N, cell_type=cell_type, ghost_mode=ghost_mode, dtype=rdtype
    )
    gdim = msh.geometry.dim

    el = ("Lagrange", 3, (gdim,))

    volume_element = basix.ufl.element(el[0], msh.basix_cell(), el[1], shape=el[2], dtype=rdtype)
    V = dolfinx.fem.functionspace(msh, volume_element)

    x = ufl.SpatialCoordinate(msh)
    v = ufl.TestFunction(V)

    def marked_ridges(x):
        return np.isclose(x[0], 1) & np.isclose(x[1], 0.5)

    exterior_ridges = dolfinx.mesh.locate_entities_boundary(
        msh, msh.topology.dim - 2, marked_ridges
    )
    et = dolfinx.mesh.meshtags(
        msh, msh.topology.dim - 2, exterior_ridges, np.full_like(exterior_ridges, 33)
    )

    def f(mod, x):
        return x[0] + mod.sin(x[1]), x[2] + x[1], x[0] ** 2 - 3 * x[1]

    def integrand(x, v):
        if coefficient:
            Z = dolfinx.fem.functionspace(v.ufl_function_space().mesh, ("Lagrange", 1, (gdim,)))
            z = dolfinx.fem.Function(Z, dtype=dtype)
            z.interpolate(lambda x: f(np, x))
        else:
            z = ufl.as_vector(f(ufl, x))
        return ufl.inner(z, v)

    metadata = {"quadrature_degree": 10}
    dr = ufl.Measure("dr", domain=msh, subdomain_data=et, subdomain_id=33, metadata=metadata)
    F = dolfinx.fem.form(integrand(x, v) * dr, dtype=dtype)
    b = dolfinx.fem.assemble_vector(F)
    b.scatter_reverse(la.InsertMode.add)
    b.scatter_forward()

    # Create reference solution on unit interval
    assert gdim == 3
    if comm.rank == 0:
        nodes = np.zeros((N + 1, gdim), dtype=rdtype)
        nodes[:, 0] = 1
        nodes[:, 1] = 0.5
        nodes[:, 2] = np.linspace(0, 1, N + 1)
        connectivity = (
            np.repeat(np.arange(nodes.shape[0]), 2)[1:-1]
            .reshape(nodes.shape[0] - 1, 2)
            .astype(np.int64)
        )
    else:
        nodes = np.zeros((0, gdim), dtype=rdtype)
        connectivity = np.zeros((0, 2), dtype=np.int64)

    c_el = ufl.Mesh(
        basix.ufl.element(
            "Lagrange", basix.CellType.interval, 1, shape=(nodes.shape[1],), dtype=rdtype
        ),
    )
    line_mesh = dolfinx.mesh.create_mesh(
        MPI.COMM_WORLD,
        x=nodes,
        cells=connectivity,
        e=c_el,
        partitioner=dolfinx.mesh.create_cell_partitioner(ghost_mode, 2),
    )

    line_element = basix.ufl.element(
        el[0], line_mesh.basix_cell(), el[1], shape=el[2], dtype=rdtype
    )
    Q = dolfinx.fem.functionspace(line_mesh, line_element)
    q = ufl.TestFunction(Q)

    x_e = ufl.SpatialCoordinate(line_mesh)
    F_ref = dolfinx.fem.form(
        integrand(x_e, q) * ufl.dx(domain=line_mesh, metadata=metadata), dtype=dtype
    )
    b_ref = dolfinx.fem.assemble_vector(F_ref)
    b_ref.scatter_reverse(la.InsertMode.add)
    b_ref.scatter_forward()
    tol = 10 * np.finfo(rdtype).eps
    assert np.isclose(la.norm(b), la.norm(b_ref), rtol=tol, atol=tol)
