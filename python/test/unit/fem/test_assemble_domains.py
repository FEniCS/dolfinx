# Copyright (C) 2019 Chris Richardson
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for assembly over domains"""

from mpi4py import MPI

import numpy as np
import pytest

import ufl
from dolfinx import default_scalar_type, fem, la
from dolfinx.fem import Constant, Function, assemble_scalar, dirichletbc, form, functionspace
from dolfinx.graph import adjacencylist
from dolfinx.mesh import (
    GhostMode,
    Mesh,
    create_unit_square,
    locate_entities,
    locate_entities_boundary,
    meshtags,
    meshtags_from_entities,
)


@pytest.fixture
def mesh():
    return create_unit_square(MPI.COMM_WORLD, 10, 10)


def create_cell_meshtags_from_entities(mesh: Mesh, dim: int, cells: np.ndarray, values: np.ndarray):
    mesh.topology.create_connectivity(mesh.topology.dim, 0)
    cell_to_vertices = mesh.topology.connectivity(mesh.topology.dim, 0)
    entities = adjacencylist(np.array([cell_to_vertices.links(cell) for cell in cells]))
    return meshtags_from_entities(mesh, dim, entities, values)


parametrize_ghost_mode = pytest.mark.parametrize(
    "mode",
    [
        pytest.param(
            GhostMode.none,
            marks=pytest.mark.skipif(
                condition=MPI.COMM_WORLD.size > 1,
                reason="Unghosted interior facets fail in parallel",
            ),
        ),
        GhostMode.shared_facet,
    ],
)


@pytest.mark.parametrize("mode", [GhostMode.none, GhostMode.shared_facet])
@pytest.mark.parametrize("meshtags_factory", [meshtags, create_cell_meshtags_from_entities])
def test_assembly_dx_domains(mode, meshtags_factory):
    mesh = create_unit_square(MPI.COMM_WORLD, 10, 10, ghost_mode=mode)
    V = functionspace(mesh, ("Lagrange", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

    # Prepare a marking structures
    # indices cover all cells
    # values are [1, 2, 3, 3, ...]
    cell_map = mesh.topology.index_map(mesh.topology.dim)
    num_cells = cell_map.size_local + cell_map.num_ghosts
    indices = np.arange(0, num_cells)
    values = np.full(indices.shape, 3, dtype=np.intc)
    values[0] = 1
    values[1] = 2
    marker = meshtags_factory(mesh, mesh.topology.dim, indices, values)
    dx = ufl.Measure("dx", subdomain_data=marker, domain=mesh)
    w = Function(V)
    w.x.array[:] = 0.5

    # Assemble matrix
    a = form(w * ufl.inner(u, v) * (dx(1) + dx(2) + dx(3)))
    A = fem.assemble_matrix(a)
    A.scatter_reverse()
    a2 = form(w * ufl.inner(u, v) * dx)
    A2 = fem.assemble_matrix(a2)
    A2.scatter_reverse()
    assert np.allclose(A.data, A2.data)

    bc = dirichletbc(Function(V), np.arange(V.dofmap.index_map.size_local // 2, dtype=np.int32))

    # Assemble vector
    L = form(ufl.inner(w, v) * (dx(1) + dx(2) + dx(3)))
    b = fem.assemble_vector(L)

    fem.apply_lifting(b.array, [a], [[bc]])
    b.scatter_reverse(la.InsertMode.add)
    bc.set(b.array)

    L2 = form(ufl.inner(w, v) * dx)
    b2 = fem.assemble_vector(L2)
    fem.apply_lifting(b2.array, [a], [[bc]])
    b2.scatter_reverse(la.InsertMode.add)
    bc.set(b2.array)
    assert np.allclose(b.array, b2.array)

    # Assemble scalar
    L = form(w * (dx(1) + dx(2) + dx(3)))
    s = assemble_scalar(L)
    s = mesh.comm.allreduce(s, op=MPI.SUM)
    assert s == pytest.approx(0.5, rel=1.0e-6)
    L2 = form(w * dx)
    s2 = assemble_scalar(L2)
    s2 = mesh.comm.allreduce(s2, op=MPI.SUM)
    assert s == pytest.approx(s2, rel=1.0e-6)

    # Assemble scalar, using both dx("everywhere") and dx(i), i = 1, 2, 3
    L = form(w * (dx(1) + dx(2) + dx(3) + dx))
    s_sum = assemble_scalar(L)
    s_sum = mesh.comm.allreduce(s_sum, op=MPI.SUM)
    assert s_sum == pytest.approx(s + s2, rel=1.0e-6)

    L2 = form(2 * w * dx)
    s2 = assemble_scalar(L2)
    s2 = mesh.comm.allreduce(s2, op=MPI.SUM)
    assert s_sum == pytest.approx(s2, rel=1.0e-6)


@pytest.mark.parametrize("mode", [GhostMode.none, GhostMode.shared_facet])
def test_assembly_ds_domains(mode):
    mesh = create_unit_square(MPI.COMM_WORLD, 10, 10, ghost_mode=mode)
    V = functionspace(mesh, ("Lagrange", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

    def bottom(x):
        return np.isclose(x[1], 0.0)

    def top(x):
        return np.isclose(x[1], 1.0)

    def left(x):
        return np.isclose(x[0], 0.0)

    def right(x):
        return np.isclose(x[0], 1.0)

    bottom_facets = locate_entities_boundary(mesh, mesh.topology.dim - 1, bottom)
    bottom_vals = np.full(bottom_facets.shape, 1, np.intc)

    top_facets = locate_entities_boundary(mesh, mesh.topology.dim - 1, top)
    top_vals = np.full(top_facets.shape, 2, np.intc)

    left_facets = locate_entities_boundary(mesh, mesh.topology.dim - 1, left)
    left_vals = np.full(left_facets.shape, 3, np.intc)

    right_facets = locate_entities_boundary(mesh, mesh.topology.dim - 1, right)
    right_vals = np.full(right_facets.shape, 6, np.intc)

    indices = np.hstack((bottom_facets, top_facets, left_facets, right_facets))
    values = np.hstack((bottom_vals, top_vals, left_vals, right_vals))

    indices, pos = np.unique(indices, return_index=True)
    marker = meshtags(mesh, mesh.topology.dim - 1, indices, values[pos])

    ds = ufl.Measure("ds", subdomain_data=marker, domain=mesh)

    w = Function(V)
    w.x.array[:] = 0.5

    bc = dirichletbc(Function(V), np.arange(V.dofmap.index_map.size_local // 2, dtype=np.int32))

    # Assemble matrix
    a = form(w * ufl.inner(u, v) * (ds(1) + ds(2) + ds(3) + ds(6)))
    A = fem.assemble_matrix(a)
    A.scatter_reverse()
    a2 = form(w * ufl.inner(u, v) * ds)
    A2 = fem.assemble_matrix(a2)
    A2.scatter_reverse()
    assert np.allclose(A.data, A2.data)

    # Assemble vector
    L = form(ufl.inner(w, v) * (ds(1) + ds(2) + ds(3) + ds(6)))
    b = fem.assemble_vector(L)

    fem.apply_lifting(b.array, [a], [[bc]])
    b.scatter_reverse(la.InsertMode.add)
    bc.set(b.array)

    L2 = form(ufl.inner(w, v) * ds)
    b2 = fem.assemble_vector(L2)
    fem.apply_lifting(b2.array, [a2], [[bc]])
    b2.scatter_reverse(la.InsertMode.add)
    bc.set(b2.array)
    assert np.allclose(b.array, b2.array)

    # Assemble scalar
    L = form(w * (ds(1) + ds(2) + ds(3) + ds(6)))
    s = assemble_scalar(L)
    s = mesh.comm.allreduce(s, op=MPI.SUM)
    L2 = form(w * ds)
    s2 = assemble_scalar(L2)
    s2 = mesh.comm.allreduce(s2, op=MPI.SUM)
    assert s == pytest.approx(s2, 1.0e-6)
    assert 2.0 == pytest.approx(s, 1.0e-6)  # /NOSONAR


@parametrize_ghost_mode
def test_assembly_dS_domains(mode):
    N = 10
    mesh = create_unit_square(MPI.COMM_WORLD, N, N, ghost_mode=mode)
    one = Constant(mesh, default_scalar_type(1))
    val = assemble_scalar(form(one * ufl.dS))
    val = mesh.comm.allreduce(val, op=MPI.SUM)
    assert val == pytest.approx(2 * (N - 1) + N * np.sqrt(2), 1.0e-5)


@parametrize_ghost_mode
def test_additivity(mode):
    mesh = create_unit_square(MPI.COMM_WORLD, 12, 12, ghost_mode=mode)
    V = functionspace(mesh, ("Lagrange", 1))

    f1 = Function(V)
    f2 = Function(V)
    f3 = Function(V)
    f1.x.array[:] = 1.0
    f2.x.array[:] = 2.0
    f3.x.array[:] = 3.0
    j1 = ufl.inner(f1, f1) * ufl.dx(mesh)
    j2 = ufl.inner(f2, f2) * ufl.ds(mesh)
    j3 = ufl.inner(ufl.avg(f3), ufl.avg(f3)) * ufl.dS(mesh)

    # Assemble each scalar form separately
    J1 = mesh.comm.allreduce(assemble_scalar(form(j1)), op=MPI.SUM)
    J2 = mesh.comm.allreduce(assemble_scalar(form(j2)), op=MPI.SUM)
    J3 = mesh.comm.allreduce(assemble_scalar(form(j3)), op=MPI.SUM)

    # Sum forms and assemble the result
    J12 = mesh.comm.allreduce(assemble_scalar(form(j1 + j2)), op=MPI.SUM)
    J13 = mesh.comm.allreduce(assemble_scalar(form(j1 + j3)), op=MPI.SUM)
    J23 = mesh.comm.allreduce(assemble_scalar(form(j2 + j3)), op=MPI.SUM)
    J123 = mesh.comm.allreduce(assemble_scalar(form(j1 + j2 + j3)), op=MPI.SUM)

    # Compare assembled values
    assert (J1 + J2) == pytest.approx(J12)
    assert (J1 + J3) == pytest.approx(J13)
    assert (J2 + J3) == pytest.approx(J23)
    assert (J1 + J2 + J3) == pytest.approx(J123)


def test_manual_integration_domains():
    """Test that specifying integration domains manually i.e.
    by passing a list of cell indices or (cell, local facet) pairs to
    form gives the same result as the usual approach of tagging"""
    n = 4
    msh = create_unit_square(MPI.COMM_WORLD, n, n)

    V = functionspace(msh, ("Lagrange", 1))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Create meshtags to mark some cells
    tdim = msh.topology.dim
    cell_map = msh.topology.index_map(tdim)
    num_cells = cell_map.size_local + cell_map.num_ghosts
    cell_indices = np.arange(0, num_cells)
    cell_values = np.zeros_like(cell_indices, dtype=np.intc)
    marked_cells = locate_entities(msh, tdim, lambda x: x[0] < 0.75)
    cell_values[marked_cells] = 7
    mt_cells = meshtags(msh, tdim, cell_indices, cell_values)

    # Create meshtags to mark some exterior facets
    msh.topology.create_entities(tdim - 1)
    facet_map = msh.topology.index_map(tdim - 1)
    num_facets = facet_map.size_local + facet_map.num_ghosts
    facet_indices = np.arange(0, num_facets)
    facet_values = np.zeros_like(facet_indices, dtype=np.intc)
    marked_ext_facets = locate_entities_boundary(msh, tdim - 1, lambda x: np.isclose(x[0], 0.0))
    marked_int_facets = locate_entities(msh, tdim - 1, lambda x: x[0] < 0.75)
    # marked_int_facets will also contain facets on the boundary, so set
    # these values first, followed by the values for marked_ext_facets
    facet_values[marked_int_facets] = 3
    facet_values[marked_ext_facets] = 6
    mt_facets = meshtags(msh, tdim - 1, facet_indices, facet_values)

    # Create measures
    dx_mt = ufl.Measure("dx", subdomain_data=mt_cells, domain=msh)
    ds_mt = ufl.Measure("ds", subdomain_data=mt_facets, domain=msh)
    dS_mt = ufl.Measure("dS", subdomain_data=mt_facets, domain=msh)

    g = Function(V)
    g.interpolate(lambda x: x[1] ** 2)

    def create_forms(dx, ds, dS):
        a = form(
            ufl.inner(g * u, v) * (dx(0) + dx(7) + ds(6))
            + ufl.inner(g * u("+"), v("+") + v("-")) * dS(3)
        )
        L = form(ufl.inner(g, v) * (dx(0) + dx(7) + ds(6)) + ufl.inner(g, v("+") + v("-")) * dS(3))
        return (a, L)

    # Create forms and assemble
    a, L = create_forms(dx_mt, ds_mt, dS_mt)
    A_mt = fem.assemble_matrix(a)
    A_mt.scatter_reverse()
    b_mt = fem.assemble_vector(L)

    # Manually specify cells to integrate over (removing ghosts
    # to give same result as above)
    cell_domains = [
        (domain_id, cell_indices[(cell_values == domain_id) & (cell_indices < cell_map.size_local)])
        for domain_id in [7, 0]
    ]

    # Manually specify exterior facets to integrate over as
    # (cell, local facet) pairs
    ext_facet_domain = []
    msh.topology.create_connectivity(tdim, tdim - 1)
    msh.topology.create_connectivity(tdim - 1, tdim)
    c_to_f = msh.topology.connectivity(tdim, tdim - 1)
    f_to_c = msh.topology.connectivity(tdim - 1, tdim)
    for f in marked_ext_facets:
        if f < facet_map.size_local:
            c = f_to_c.links(f)[0]
            local_f = np.where(c_to_f.links(c) == f)[0][0]
            ext_facet_domain.append(c)
            ext_facet_domain.append(local_f)
    ext_facet_domains = [(6, ext_facet_domain)]

    # Manually specify interior facets to integrate over
    int_facet_domain = []
    for f in marked_int_facets:
        if f >= facet_map.size_local or len(f_to_c.links(f)) != 2:
            continue
        c_0, c_1 = f_to_c.links(f)[0], f_to_c.links(f)[1]
        local_f_0 = np.where(c_to_f.links(c_0) == f)[0][0]
        local_f_1 = np.where(c_to_f.links(c_1) == f)[0][0]
        int_facet_domain.append(c_0)
        int_facet_domain.append(local_f_0)
        int_facet_domain.append(c_1)
        int_facet_domain.append(local_f_1)
    int_facet_domains = [(3, int_facet_domain)]

    # Create measures
    dx_manual = ufl.Measure("dx", subdomain_data=cell_domains, domain=msh)
    ds_manual = ufl.Measure("ds", subdomain_data=ext_facet_domains, domain=msh)
    dS_manual = ufl.Measure("dS", subdomain_data=int_facet_domains, domain=msh)

    # Assemble forms and check
    a, L = create_forms(dx_manual, ds_manual, dS_manual)
    A = fem.assemble_matrix(a)
    A.scatter_reverse()
    b = fem.assemble_vector(L)

    assert np.allclose(A.data, A_mt.data)
    assert np.allclose(b.array, b_mt.array)
