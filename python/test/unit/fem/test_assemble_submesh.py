# Copyright (C) 2022-2024 Joseph P. Dean, Jørgen S. Dokken
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

# TODO Test replacing mesh with submesh for existing assembler tests

from mpi4py import MPI

import numpy as np
import pytest

import ufl
from dolfinx import default_scalar_type, fem, la
from dolfinx.fem import compute_integration_domains
from dolfinx.mesh import (
    CellType,
    GhostMode,
    compute_incident_entities,
    create_box,
    create_rectangle,
    create_submesh,
    create_unit_cube,
    create_unit_interval,
    create_unit_square,
    entities_to_geometry,
    exterior_facet_indices,
    locate_entities,
    locate_entities_boundary,
    meshtags,
)


def assemble(mesh, space, k):
    V = fem.functionspace(mesh, (space, k))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    dx = ufl.Measure("dx", domain=mesh)
    ds = ufl.Measure("ds", domain=mesh)

    c = fem.Constant(mesh, default_scalar_type(0.75))
    a = fem.form(ufl.inner(c * u, v) * (dx + ds))

    facet_dim = mesh.topology.dim - 1
    facets = locate_entities_boundary(mesh, facet_dim, lambda x: np.isclose(x[0], 1))
    dofs = fem.locate_dofs_topological(V, facet_dim, facets)

    bc_func = fem.Function(V)
    bc_func.interpolate(lambda x: np.sin(x[0]))
    bc = fem.dirichletbc(bc_func, dofs)

    A = fem.assemble_matrix(a, bcs=[bc])
    A.scatter_reverse()

    # TODO Test assembly with fem.Function
    x = ufl.SpatialCoordinate(mesh)
    f = 1.5 + x[0]
    L = fem.form(ufl.inner(c * f, v) * (dx + ds))
    b = fem.assemble_vector(L)
    fem.apply_lifting(b.array, [a], bcs=[[bc]])
    b.scatter_reverse(la.InsertMode.add)
    bc.set(b.array)
    s = mesh.comm.allreduce(
        fem.assemble_scalar(fem.form(ufl.inner(c * f, f) * (dx + ds))), op=MPI.SUM
    )
    return A, b, s


@pytest.mark.parametrize("d", [2, 3])
@pytest.mark.parametrize("n", [2, 6])
@pytest.mark.parametrize("k", [1, 4])
@pytest.mark.parametrize("space", ["Lagrange", "Discontinuous Lagrange"])
@pytest.mark.parametrize("ghost_mode", [GhostMode.none, GhostMode.shared_facet])
def test_submesh_cell_assembly(d, n, k, space, ghost_mode):
    """Check that assembling a form over a unit square gives the same
    result as assembling over half of a 2x1 rectangle with the same
    triangulation."""
    if d == 2:
        mesh_0 = create_unit_square(MPI.COMM_WORLD, n, n, ghost_mode=ghost_mode)
        mesh_1 = create_rectangle(
            MPI.COMM_WORLD, ((0.0, 0.0), (2.0, 1.0)), (2 * n, n), ghost_mode=ghost_mode
        )
    else:
        mesh_0 = create_unit_cube(MPI.COMM_WORLD, n, n, n, ghost_mode=ghost_mode)
        mesh_1 = create_box(
            MPI.COMM_WORLD, ((0.0, 0.0, 0.0), (2.0, 1.0, 1.0)), (2 * n, n, n), ghost_mode=ghost_mode
        )

    A_mesh_0, b_mesh_0, s_mesh_0 = assemble(mesh_0, space, k)

    edim = mesh_1.topology.dim
    entities = locate_entities(mesh_1, edim, lambda x: x[0] <= 1.0)
    submesh = create_submesh(mesh_1, edim, entities)[0]
    A_submesh, b_submesh, s_submesh = assemble(submesh, space, k)

    assert A_mesh_0.squared_norm() == pytest.approx(
        A_submesh.squared_norm(), rel=1.0e-4, abs=1.0e-4
    )
    assert la.norm(b_mesh_0) == pytest.approx(la.norm(b_submesh), rel=1.0e-4)
    assert np.isclose(s_mesh_0, s_submesh)


@pytest.mark.parametrize("n", [2, 6])
@pytest.mark.parametrize("k", [1, 4])
@pytest.mark.parametrize("space", ["Lagrange", "Discontinuous Lagrange"])
@pytest.mark.parametrize("ghost_mode", [GhostMode.none, GhostMode.shared_facet])
def test_submesh_facet_assembly(n, k, space, ghost_mode):
    """Test that assembling a form over the face of a unit cube gives
    the same result as assembling it over a unit square."""
    cube_mesh = create_unit_cube(MPI.COMM_WORLD, n, n, n, ghost_mode=ghost_mode)
    edim = cube_mesh.topology.dim - 1
    entities = locate_entities_boundary(cube_mesh, edim, lambda x: np.isclose(x[2], 0.0))
    submesh = create_submesh(cube_mesh, edim, entities)[0]

    A_submesh, b_submesh, s_submesh = assemble(submesh, space, k)

    square_mesh = create_unit_square(MPI.COMM_WORLD, n, n, ghost_mode=ghost_mode)
    A_square_mesh, b_square_mesh, s_square_mesh = assemble(square_mesh, space, k)

    assert A_submesh.squared_norm() == pytest.approx(
        A_square_mesh.squared_norm(), rel=1.0e-5, abs=1.0e-5
    )
    assert la.norm(b_submesh) == pytest.approx(la.norm(b_square_mesh))
    assert np.isclose(s_submesh, s_square_mesh)


def create_measure(msh, integral_type):
    """Helper function to create an integration measure of type `integral_type`
    over domain `msh`"""

    def create_meshtags(msh, dim, entities):
        values = np.full_like(entities, 1, dtype=np.intc)
        perm = np.argsort(entities)
        return meshtags(msh, dim, entities[perm], values[perm])

    tdim = msh.topology.dim
    fdim = tdim - 1
    if integral_type == "dx":
        cells = locate_entities(msh, tdim, lambda x: x[0] <= 0.5)
        mt = create_meshtags(msh, tdim, cells)
    elif integral_type == "ds":
        facets = locate_entities_boundary(
            msh, tdim - 1, lambda x: np.isclose(x[1], 0.0) & (x[0] <= 0.5)
        )
        mt = create_meshtags(msh, fdim, facets)
    else:
        assert integral_type == "dS"

        def interior_marker(x):
            dist = 1 / 12
            return (x[0] > dist) & (x[0] < 1 - dist) & (x[1] > dist) & (x[1] < 1 - dist)

        facets = locate_entities(msh, fdim, interior_marker)
        mt = create_meshtags(msh, fdim, facets)

    return ufl.Measure(integral_type, domain=msh, subdomain_data=mt)(1)


def a_ufl(u, v, f, g, measure):
    "Helper function to create a UFL bilinear form. The form depends on the integral type"
    if measure.integral_type() == "cell" or measure.integral_type() == "exterior_facet":
        return ufl.inner(f * g * u, v) * measure
    else:
        assert measure.integral_type() == "interior_facet"
        return ufl.inner(f("-") * g("-") * (u("+") + u("-")), v("+") + v("-")) * measure


def L_ufl(v, f, g, measure):
    "Helper function to create a UFL linear form. The form depends on the integral type"
    if measure.integral_type() == "cell" or measure.integral_type() == "exterior_facet":
        return ufl.inner(f * g, v) * measure
    else:
        assert measure.integral_type() == "interior_facet"
        return ufl.inner(f("+") * g("+"), v("+") + v("-")) * measure


def M_ufl(f, g, measure):
    if measure.integral_type() == "cell" or measure.integral_type() == "exterior_facet":
        return f * g * measure
    else:
        assert measure.integral_type() == "interior_facet"
        return (f("+") + f("-")) * (g("+") + g("-")) * measure


@pytest.mark.parametrize("n", [4, 6])
@pytest.mark.parametrize("k", [1, 3])
@pytest.mark.parametrize("space", ["Lagrange", "Discontinuous Lagrange"])
@pytest.mark.parametrize("integral_type", ["dx", "ds", "dS"])
def test_mixed_dom_codim_0(n, k, space, integral_type):
    """Test assembling forms where the trial and test functions
    are defined over different meshes"""

    # Create a mesh
    msh = create_rectangle(
        MPI.COMM_WORLD, ((0.0, 0.0), (2.0, 1.0)), (2 * n, n), ghost_mode=GhostMode.shared_facet
    )

    # Create a submesh of the left half of the mesh
    tdim = msh.topology.dim
    cells = locate_entities(msh, tdim, lambda x: x[0] <= 1.0)
    smsh, entity_map = create_submesh(msh, tdim, cells)[:2]

    # Define function spaces over the mesh and submesh
    V = fem.functionspace(msh, (space, k))
    W = fem.functionspace(msh, (space, k))
    Q = fem.functionspace(smsh, (space, k))

    # Trial and test functions on the mesh
    u = ufl.TrialFunction(V)
    w = ufl.TestFunction(W)

    # Test function on the submesh
    q = ufl.TestFunction(Q)

    # Coefficients
    def coeff_expr(x):
        return np.sin(np.pi * x[0])

    # Coefficient defined over the mesh
    f = fem.Function(V)
    f.interpolate(coeff_expr)

    # Coefficient defined over the submesh
    g = fem.Function(Q)
    g.interpolate(coeff_expr)

    # Create an integration measure defined over msh
    measure_msh = create_measure(msh, integral_type)

    # Create a Dirichlet boundary condition
    u_bc = fem.Function(V)
    u_bc.interpolate(lambda x: np.sin(np.pi * x[1]))
    dirichlet_facets = locate_entities_boundary(
        msh, msh.topology.dim - 1, lambda x: np.isclose(x[0], 0.0)
    )
    dirichlet_dofs = fem.locate_dofs_topological(V, msh.topology.dim - 1, dirichlet_facets)
    bc = fem.dirichletbc(u_bc, dirichlet_dofs)

    # Single-domain assembly over msh as a reference to check against
    a = fem.form(a_ufl(u, w, f, f, measure_msh))
    A = fem.assemble_matrix(a, bcs=[bc])
    A.scatter_reverse()

    L = fem.form(L_ufl(w, f, f, measure_msh))
    b = fem.assemble_vector(L)
    fem.apply_lifting(b.array, [a], bcs=[[bc]])
    b.scatter_reverse(la.InsertMode.add)

    M = fem.form(M_ufl(f, f, measure_msh))
    c = msh.comm.allreduce(fem.assemble_scalar(M), op=MPI.SUM)

    # Assemble a mixed-domain form using msh as integration domain.
    # We must pass the entity map relating `smsh` cells to `msh` cells
    entity_maps = [entity_map]
    a0 = fem.form(a_ufl(u, q, f, g, measure_msh), entity_maps=entity_maps)
    A0 = fem.assemble_matrix(a0, bcs=[bc])
    A0.scatter_reverse()
    assert np.isclose(A0.squared_norm(), A.squared_norm())

    L0 = fem.form(L_ufl(q, f, g, measure_msh), entity_maps=entity_maps)
    b0 = fem.assemble_vector(L0)
    fem.apply_lifting(b0.array, [a0], bcs=[[bc]])
    b0.scatter_reverse(la.InsertMode.add)
    assert np.isclose(la.norm(b0), la.norm(b))
    M0 = fem.form(M_ufl(f, g, measure_msh), entity_maps=entity_maps)
    c0 = msh.comm.allreduce(fem.assemble_scalar(M0), op=MPI.SUM)
    assert np.isclose(c0, c)

    # Now assemble a mixed-domain form taking smsh to be the integration
    # domain.

    # Create the measure (this time defined over the submesh)
    measure_smsh = create_measure(smsh, integral_type)

    a1 = fem.form(a_ufl(u, q, f, g, measure_smsh), entity_maps=entity_maps)
    A1 = fem.assemble_matrix(a1, bcs=[bc])
    A1.scatter_reverse()
    assert np.isclose(A1.squared_norm(), A.squared_norm())

    L1 = fem.form(L_ufl(q, f, g, measure_smsh), entity_maps=entity_maps)
    b1 = fem.assemble_vector(L1)
    fem.apply_lifting(b1.array, [a1], bcs=[[bc]])
    b1.scatter_reverse(la.InsertMode.add)
    assert np.isclose(la.norm(b1), la.norm(b))

    M1 = fem.form(M_ufl(f, g, measure_smsh), entity_maps=entity_maps)
    c1 = msh.comm.allreduce(fem.assemble_scalar(M1), op=MPI.SUM)
    assert np.isclose(c1, c)


@pytest.mark.parametrize("n", [4, 6])
@pytest.mark.parametrize("k", [1, 3])
def test_mixed_dom_codim_1(n, k):
    """Test assembling forms where the trial functions, test functions
    and coefficients are defined over different meshes of different topological
    dimension."""
    msh = create_unit_square(MPI.COMM_WORLD, n, n)

    # Create a submesh of the boundary
    tdim = msh.topology.dim
    msh.topology.create_connectivity(tdim - 1, tdim)
    boundary_facets = exterior_facet_indices(msh.topology)

    smsh, entity_map = create_submesh(msh, tdim - 1, boundary_facets)[:2]

    # Define function spaces over the mesh and submesh
    V = fem.functionspace(msh, ("Lagrange", k))
    W = fem.functionspace(msh, ("Lagrange", k))
    Vbar = fem.functionspace(smsh, ("Lagrange", k))

    # Create a Dirichlet boundary condition
    u_bc = fem.Function(V)
    u_bc.interpolate(lambda x: np.sin(np.pi * x[1]))
    dirichlet_facets = locate_entities_boundary(
        msh, msh.topology.dim - 1, lambda x: np.isclose(x[0], 0.0)
    )
    dirichlet_dofs = fem.locate_dofs_topological(V, msh.topology.dim - 1, dirichlet_facets)
    bc = fem.dirichletbc(u_bc, dirichlet_dofs)

    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(W)
    vbar = ufl.TestFunction(Vbar)

    # Coefficients
    def coeff_expr(x):
        return np.sin(np.pi * x[0])

    # Coefficient defined over the mesh
    f = fem.Function(V)
    f.interpolate(coeff_expr)

    # Coefficient defined over the submesh
    g = fem.Function(Vbar)
    g.interpolate(coeff_expr)

    # Create the integration measure. Mixed-dimensional forms use the
    # higher-dimensional domain as the integration domain
    ds = ufl.Measure("ds", domain=msh)

    # Create reference forms to compare to
    a = fem.form(a_ufl(u, v, f, f, ds))
    A = fem.assemble_matrix(a, bcs=[bc])
    A.scatter_reverse()

    L = fem.form(L_ufl(v, f, f, ds))
    b = fem.assemble_vector(L)
    fem.apply_lifting(b.array, [a], bcs=[[bc]])
    b.scatter_reverse(la.InsertMode.add)

    M = fem.form(M_ufl(f, f, ds))
    c = msh.comm.allreduce(fem.assemble_scalar(M), op=MPI.SUM)

    entity_maps = [entity_map]

    # Create forms and compare
    a1 = fem.form(a_ufl(u, vbar, f, g, ds), entity_maps=entity_maps)
    A1 = fem.assemble_matrix(a1, bcs=[bc])
    A1.scatter_reverse()

    assert np.isclose(A.squared_norm(), A1.squared_norm())

    L1 = fem.form(L_ufl(vbar, f, g, ds), entity_maps=entity_maps)
    b1 = fem.assemble_vector(L1)
    fem.apply_lifting(b1.array, [a1], bcs=[[bc]])
    b1.scatter_reverse(la.InsertMode.add)

    assert np.isclose(la.norm(b), la.norm(b1))

    M1 = fem.form(M_ufl(f, g, ds), entity_maps=entity_maps)
    c1 = msh.comm.allreduce(fem.assemble_scalar(M1), op=MPI.SUM)

    assert np.isclose(c, c1)


def test_disjoint_submeshes():
    # FIXME Simplify this test
    """Test assembly with multiple disjoint submeshes in same variational form"""
    N = 10
    tol = 1e-14
    mesh = create_unit_interval(MPI.COMM_WORLD, N, ghost_mode=GhostMode.shared_facet)
    tdim = mesh.topology.dim
    dx = 1.0 / N
    center_tag = 1
    left_tag = 2
    right_tag = 3
    left_interface_tag = 4
    right_interface_tag = 5

    def left(x):
        return x[0] < N // 3 * dx + tol

    def right(x):
        return x[0] > 2 * N // 3 * dx - tol

    cell_map = mesh.topology.index_map(tdim)
    num_cells_local = cell_map.size_local + cell_map.num_ghosts
    values = np.full(num_cells_local, center_tag, dtype=np.int32)
    values[locate_entities(mesh, tdim, left)] = left_tag
    values[locate_entities(mesh, tdim, right)] = right_tag

    cell_tag = meshtags(mesh, tdim, np.arange(num_cells_local, dtype=np.int32), values)
    left_facets = compute_incident_entities(mesh.topology, cell_tag.find(left_tag), tdim, tdim - 1)
    center_facets = compute_incident_entities(
        mesh.topology, cell_tag.find(center_tag), tdim, tdim - 1
    )
    right_facets = compute_incident_entities(
        mesh.topology, cell_tag.find(right_tag), tdim, tdim - 1
    )

    # Create parent facet tag where left interface is tagged with 4,
    # right with 5
    left_interface = np.intersect1d(left_facets, center_facets)
    right_interface = np.intersect1d(right_facets, center_facets)
    facet_map = mesh.topology.index_map(tdim)
    num_facet_local = facet_map.size_local + cell_map.num_ghosts
    facet_values = np.full(num_facet_local, 1, dtype=np.int32)
    facet_values[left_interface] = left_interface_tag
    facet_values[right_interface] = right_interface_tag
    facet_tag = meshtags(mesh, tdim - 1, np.arange(num_facet_local, dtype=np.int32), facet_values)

    # Create facet integrals on each interface
    left_mesh, left_emap, _, _ = create_submesh(mesh, tdim, cell_tag.find(left_tag))
    right_mesh, right_emap, _, _ = create_submesh(mesh, tdim, cell_tag.find(right_tag))

    # One sided interface integral uses only "+" restriction. Sort
    # integration entities such that this is always satisfied
    def compute_mapped_interior_facet_data(mesh, facet_tag, value, parent_to_sub_map):
        """Compute integration data for interior facet integrals, where
        the positive restriction is always taken on the side that has a
        cell in the sub mesh.

        Args:
            mesh: Parent mesh
            facet_tag: Meshtags object for facets
            value: Value of the facets to extract
            parent_to_sub_map: Mapping from parent mesh to sub mesh

        Returns:
            Integration data for interior facets
        """
        mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
        assert facet_tag.dim == mesh.topology.dim - 1
        integration_data = compute_integration_domains(
            fem.IntegralType.interior_facet, mesh.topology, facet_tag.find(value)
        )
        mapped_cell_0 = parent_to_sub_map[integration_data[0::4]]
        mapped_cell_1 = parent_to_sub_map[integration_data[2::4]]
        switch = mapped_cell_1 > mapped_cell_0
        # Order restriction on one side
        ordered_integration_data = integration_data.reshape(-1, 4).copy()
        if True in switch:
            ordered_integration_data[switch, [0, 1, 2, 3]] = ordered_integration_data[
                switch, [2, 3, 0, 1]
            ]
        return (value, ordered_integration_data.reshape(-1))

    cells = np.arange(num_cells_local, dtype=np.int32)
    parent_to_left = left_emap.sub_topology_to_topology(cells, inverse=True)
    parent_to_right = right_emap.sub_topology_to_topology(cells, inverse=True)
    integral_data = [
        compute_mapped_interior_facet_data(mesh, facet_tag, left_interface_tag, parent_to_left),
        compute_mapped_interior_facet_data(mesh, facet_tag, right_interface_tag, parent_to_right),
    ]

    dS = ufl.Measure("dS", domain=mesh, subdomain_data=integral_data)

    def f_left(x):
        return np.sin(x[0])

    def f_right(x):
        return x[0]

    V_left = fem.functionspace(left_mesh, ("Lagrange", 1))
    u_left = fem.Function(V_left)
    u_left.interpolate(f_left)

    V_right = fem.functionspace(right_mesh, ("Lagrange", 1))
    u_right = fem.Function(V_right)
    u_right.interpolate(f_right)

    # Create single integral with different submeshes restrictions
    x = ufl.SpatialCoordinate(mesh)
    res = "+"
    J = x[0] * u_left(res) * dS(left_interface_tag) + ufl.cos(x[0]) * u_right(res) * dS(
        right_interface_tag
    )

    # Create entity maps for each submesh
    entity_maps = [left_emap, right_emap]

    J_compiled = fem.form(J, entity_maps=entity_maps)
    J_local = fem.assemble_scalar(J_compiled)
    J_sum = mesh.comm.allreduce(J_local, op=MPI.SUM)

    vertex_map = mesh.topology.index_map(mesh.topology.dim - 1)
    num_vertices_local = vertex_map.size_local

    # Compute value of expression at left interface
    if len(facets := facet_tag.find(left_interface_tag)) > 0:
        assert len(facets) == 1
        left_vertex = entities_to_geometry(mesh, mesh.topology.dim - 1, facets)
        if left_vertex[0, 0] < num_vertices_local:
            left_coord = mesh.geometry.x[left_vertex].reshape(3, -1)
            left_val = left_coord[0, 0] * f_left(left_coord)[0]
        else:
            left_val = 0.0
    else:
        left_val = 0.0

    # Compute value of expression at right interface
    if len(facets := facet_tag.find(right_interface_tag)) > 0:
        assert len(facets) == 1
        right_vertex = entities_to_geometry(mesh, mesh.topology.dim - 1, facets)
        if right_vertex[0, 0] < num_vertices_local:
            right_coord = mesh.geometry.x[right_vertex].reshape(3, -1)
            right_val = np.cos(right_coord[0, 0]) * f_right(right_coord)[0]
        else:
            right_val = 0.0
    else:
        right_val = 0.0

    glob_left_val = mesh.comm.allreduce(left_val, op=MPI.SUM)
    glob_right_val = mesh.comm.allreduce(right_val, op=MPI.SUM)
    assert np.isclose(J_sum, glob_left_val + glob_right_val)


@pytest.mark.petsc4py
def test_mixed_measures():
    """Test block assembly of forms where the integration measure in each
    block may be different"""
    from petsc4py import PETSc

    from dolfinx.fem.petsc import assemble_vector

    comm = MPI.COMM_WORLD
    msh = create_unit_square(comm, 16, 21, ghost_mode=GhostMode.none)

    # Create a submesh of some cells
    tdim = msh.topology.dim
    smsh_cells = locate_entities(msh, tdim, lambda x: x[0] <= 0.5)
    smsh, entity_map = create_submesh(msh, tdim, smsh_cells)[:2]

    # Create function spaces over each mesh
    V = fem.functionspace(msh, ("Lagrange", 1))
    Q = fem.functionspace(smsh, ("Lagrange", 1))

    # Define two integration measures, one over the mesh, the other over the submesh
    dx_msh = ufl.Measure("dx", msh, subdomain_data=[(1, smsh_cells)])
    dx_smsh = ufl.Measure("dx", smsh)

    # Trial and test functions
    v = ufl.TestFunction(V)
    q = ufl.TestFunction(Q)

    entity_maps = [entity_map]
    # First, assemble a block vector using both dx_msh and dx_smsh
    L = [fem.form(ufl.inner(2.3, v) * dx_msh), fem.form(ufl.inner(1.3, q) * dx_smsh)]
    b0 = assemble_vector(L, kind=PETSc.Vec.Type.MPI)
    b0.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    # Now, assemble the same vector using only dx_msh
    L = [
        fem.form(ufl.inner(2.3, v) * dx_msh),
        fem.form(ufl.inner(1.3, q) * dx_msh(1), entity_maps=entity_maps),
    ]
    b1 = assemble_vector(L, kind=PETSc.Vec.Type.MPI)
    b1.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    # Check the results are the same
    assert np.allclose(b0.norm(), b1.norm())


@pytest.mark.parametrize(
    "msh",
    [
        pytest.param(
            create_unit_interval(MPI.COMM_WORLD, 10),
            marks=pytest.mark.xfail(
                reason="Interior facet submesh of dimension 0 not supported in submesh creation",
                strict=True,
            ),
        ),
        create_unit_square(
            MPI.COMM_WORLD, 10, 10, cell_type=CellType.triangle, ghost_mode=GhostMode.shared_facet
        ),
        create_unit_cube(
            MPI.COMM_WORLD,
            3,
            3,
            3,
            cell_type=CellType.tetrahedron,
            ghost_mode=GhostMode.shared_facet,
        ),
    ],
)
def test_interior_facet_codim_1(msh):
    """
    Check that assembly on an interior facet with coefficients defined on a co-dim 1
    mesh gives the correct result.
    """
    # Collect mesh properties
    tdim = msh.topology.dim
    fdim = tdim - 1
    msh.topology.create_connectivity(fdim, tdim)
    facet_imap = msh.topology.index_map(fdim)

    # Mark all local and owned interior facets and "unmark" exterior facets
    facet_vector = la.vector(facet_imap, 1, dtype=np.int32)
    facet_vector.array[: facet_imap.size_local] = 1
    facet_vector.array[facet_imap.size_local :] = 0
    facet_vector.array[exterior_facet_indices(msh.topology)] = 0
    facet_vector.scatter_forward()
    interior_facets = np.flatnonzero(facet_vector.array)

    # Create submesh with all owned and ghosted interior facets
    submesh, entity_map, _, _ = create_submesh(msh, fdim, interior_facets)

    # Create inverse map
    entity_maps = [entity_map]

    def assemble_interior_facet_formulation(formulation, entity_maps):
        F = fem.form(formulation, entity_maps=entity_maps)
        if F.rank == 0:
            return msh.comm.allreduce(fem.assemble_scalar(F), op=MPI.SUM)
        elif F.rank == 1:
            b = fem.assemble_vector(F)
            b.scatter_reverse(la.InsertMode.add)
            b.scatter_forward()
            return b
        raise NotImplementedError(f"Unexpected formulation of rank {F.rank}")

    def f(x):
        return 2 + x[0] + 3 * x[1]

    # Compare evaluation of finite element formulations on the submesh
    # and the parent mesh
    metadata = {"quadrature_degree": 4}
    v = ufl.TestFunction(fem.functionspace(msh, ("DG", 2)))

    # Assemble forms using function interpolated on the submesh
    dS_submesh = ufl.Measure("dS", domain=msh, metadata=metadata)
    j = fem.Function(fem.functionspace(submesh, ("Lagrange", 1)))
    j.interpolate(f)
    j.x.scatter_forward()
    J_submesh = assemble_interior_facet_formulation(ufl.avg(j) * dS_submesh, entity_maps)
    b_submesh = assemble_interior_facet_formulation(
        ufl.inner(j, ufl.jump(v)) * dS_submesh, entity_maps
    )

    # Assemble reference value forms on the parent mesh using function
    # defined with UFL
    x = ufl.SpatialCoordinate(msh)
    J_ref = assemble_interior_facet_formulation(ufl.avg(f(x)) * ufl.dS(metadata=metadata), None)
    b_ref = assemble_interior_facet_formulation(
        ufl.inner(f(x), ufl.jump(v)) * ufl.dS(metadata=metadata), None
    )

    # Ensure both are equivalent
    tol = 100 * np.finfo(default_scalar_type()).eps
    assert np.isclose(J_submesh, J_ref, atol=tol)
    np.testing.assert_allclose(b_submesh.array, b_ref.array, atol=tol)


def test_interior_interface():
    """This is a test for assembling a form over an interface between
    two domains that don't overlap. The test function is defined on one
    domain, and the trial function is defined on the other.
    """

    def interface_int_entities(msh, interface_facets, marker):
        """This helper function computes the integration entities for
        interior facet integrals (i.e. a list of (cell_0, local_facet_0,
        cell_1, local_facet_1)) over an interface. The integration
        entities are ordered consistently such that cells for which
        `marker[cell] != 0` correspond to the "+" restriction, and cells
        for which `marker[cell] == 0` correspond to the "-" restriction.

        Parameters:
            msh: the mesh
            interface_facets: Facet indices of interior facets on an
                interface
            marker: If `marker[cell] != 0`, then that cell corresponds
                to a "+" restriction. Otherwise, it corresponds to a
                negative restriction.
        """
        tdim = msh.topology.dim
        fdim = tdim - 1
        msh.topology.create_connectivity(tdim, fdim)
        msh.topology.create_connectivity(fdim, tdim)
        facet_imap = msh.topology.index_map(fdim)
        c_to_f = msh.topology.connectivity(tdim, fdim)
        f_to_c = msh.topology.connectivity(fdim, tdim)

        interface_entities = []
        for facet in interface_facets:
            # Check if this facet is owned
            if facet < facet_imap.size_local:
                cells = f_to_c.links(facet)
                assert len(cells) == 2
                if marker[cells[0]] == 0:
                    cell_plus, cell_minus = cells[1], cells[0]
                else:
                    cell_plus, cell_minus = cells[0], cells[1]

                local_facet_plus = np.where(c_to_f.links(cell_plus) == facet)[0][0]
                local_facet_minus = np.where(c_to_f.links(cell_minus) == facet)[0][0]

                interface_entities.extend(
                    [cell_plus, local_facet_plus, cell_minus, local_facet_minus]
                )

        return interface_entities

    n = 10  # NOTE: Test assumes that n is even
    comm = MPI.COMM_WORLD
    msh = create_unit_square(comm, n, n)

    # Locate cells in left and right half of domain and create sub-meshes
    tdim = msh.topology.dim
    left_cells = locate_entities(msh, tdim, lambda x: x[0] <= 0.5)
    right_cells = locate_entities(msh, tdim, lambda x: x[0] >= 0.5)

    smsh_0, sm_0_emap = create_submesh(msh, tdim, left_cells)[0:2]
    smsh_1, sm_1_emap = create_submesh(msh, tdim, right_cells)[0:2]

    # Define trial function on one region and the test function on the other
    V_0 = fem.functionspace(smsh_0, ("Lagrange", 2))
    V_1 = fem.functionspace(smsh_1, ("Lagrange", 1))

    u_0 = ufl.TrialFunction(V_0)
    v_1 = ufl.TestFunction(V_1)

    # Find the facet on the interface
    fdim = tdim - 1
    interface_facets = locate_entities(msh, fdim, lambda x: np.isclose(x[0], 0.5))

    # Create a marker to identify cells on the "+" side of the interface
    cell_imap = msh.topology.index_map(tdim)
    num_cells = cell_imap.size_local + cell_imap.num_ghosts
    marker = np.zeros(num_cells)
    marker[left_cells] = 1

    # Create entity maps for each domain
    entity_maps = [sm_0_emap, sm_1_emap]

    # Create a list of integration entities
    interface_ents = interface_int_entities(msh, interface_facets, marker)

    # Assemble the form
    dS = ufl.Measure("dS", domain=msh, subdomain_data=[(1, interface_ents)])

    def f_expr(x):
        return x[0]

    f_0 = fem.Function(V_0)
    f_0.interpolate(f_expr)

    def bc_marker(x):
        return np.isclose(x[1], 1.0) & (x[0] <= 0.5)

    # Create a Dirichlet boundary condition
    c_bc = 1.0
    bc_facets = locate_entities_boundary(smsh_0, fdim, bc_marker)
    bc_dofs = fem.locate_dofs_topological(V_0, fdim, bc_facets)
    bc = fem.dirichletbc(c_bc, bc_dofs, V_0)

    a = fem.form(ufl.inner(f_0("+") * u_0("+"), v_1("-")) * dS(1), entity_maps=entity_maps)

    A = fem.assemble_matrix(a, bcs=[bc])
    A.scatter_reverse()

    A_sqnorm = A.squared_norm()

    # Now assemble using a single domain to compare to a reference
    V = fem.functionspace(msh, ("Lagrange", 2))
    W = fem.functionspace(msh, ("Lagrange", 1))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(W)
    f = fem.Function(V)
    f.interpolate(f_expr)

    a_ref = fem.form(ufl.inner(f("+") * u("+"), v("-")) * dS(1))

    bc_facets_ref = locate_entities_boundary(msh, fdim, bc_marker)
    bc_dofs_ref = fem.locate_dofs_topological(V, fdim, bc_facets_ref)
    bc_ref = fem.dirichletbc(c_bc, bc_dofs_ref, V)

    A_ref = fem.assemble_matrix(a_ref, bcs=[bc_ref])
    A_ref.scatter_reverse()

    A_ref_sqrnorm = A.squared_norm()

    assert np.isclose(A_sqnorm, A_ref_sqrnorm)

    # Same for a linear form
    L = fem.form(ufl.inner(f_0("+"), v_1("-")) * dS(1), entity_maps=entity_maps)
    b = fem.assemble_vector(L)
    fem.apply_lifting(b.array, [a], bcs=[[bc]])
    b.scatter_reverse(la.InsertMode.add)

    # Create a reference linear form
    L_ref = fem.form(ufl.inner(f("+"), v("-")) * dS(1))
    b_ref = fem.assemble_vector(L_ref)
    fem.apply_lifting(b_ref.array, [a_ref], bcs=[[bc_ref]])
    b_ref.scatter_reverse(la.InsertMode.add)

    assert np.isclose(la.norm(b), la.norm(b_ref))
