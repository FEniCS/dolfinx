# Copyright (C) 2022 Joseph P. Dean
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

# TODO Test replacing mesh with submesh for existing assembler tests
# TODO Use pygmsh

import numpy as np
import pytest

import ufl
from dolfinx import fem
from dolfinx.mesh import (GhostMode, create_box, create_rectangle,
                          create_submesh, create_unit_cube, create_unit_square,
                          locate_entities, locate_entities_boundary,
                          meshtags_from_entities, create_mesh,
                          create_cell_partitioner)
from dolfinx.cpp.mesh import cell_num_entities

from mpi4py import MPI
from petsc4py import PETSc
from dolfinx.graph import create_adjacencylist
import random
import basix.ufl_wrapper


def create_random_mesh(corners, n, ghost_mode):
    """Create a rectangular mesh made of randomly ordered simplices"""
    if MPI.COMM_WORLD.rank == 0:
        h_x = (corners[1][0] - corners[0][0]) / n[0]
        h_y = (corners[1][1] - corners[0][1]) / n[1]

        points = [(i * h_x, j * h_y)
                  for i in range(n[0] + 1) for j in range(n[1] + 1)]

        random.seed(6)

        cells = []
        for i in range(n[0]):
            for j in range(n[1]):
                v = (n[1] + 1) * i + j
                cell_0 = [v, v + 1, v + n[1] + 2]
                random.shuffle(cell_0)
                cells.append(cell_0)

                cell_1 = [v, v + n[1] + 1, v + n[1] + 2]
                random.shuffle(cell_1)
                cells.append(cell_1)
        cells = np.array(cells)
        points = np.array(points)
    else:
        cells, points = np.empty([0, 3]), np.empty([0, 2])

    domain = ufl.Mesh(basix.ufl_wrapper.create_vector_element(
        "Lagrange", "triangle", 1))
    partitioner = create_cell_partitioner(ghost_mode)
    return create_mesh(MPI.COMM_WORLD, cells, points, domain,
                       partitioner=partitioner)


def assemble_forms_0(mesh, space, k):
    """Helper function to assemble some forms for testing"""
    V = fem.FunctionSpace(mesh, (space, k))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    dx = ufl.Measure("dx", domain=mesh)
    ds = ufl.Measure("ds", domain=mesh)

    c = fem.Constant(mesh, PETSc.ScalarType(0.75))
    x = ufl.SpatialCoordinate(mesh)
    f = 1.5 + x[0]
    g = fem.Function(V)
    g.interpolate(lambda x: x[0]**2 + x[1])
    a = fem.form(ufl.inner(c * f * g * u, v) * (dx + ds))

    facet_dim = mesh.topology.dim - 1
    facets = locate_entities_boundary(
        mesh, facet_dim, lambda x: np.isclose(x[0], 1))
    dofs = fem.locate_dofs_topological(V, facet_dim, facets)

    bc_func = fem.Function(V)
    bc_func.interpolate(lambda x: x[0]**2)

    bc = fem.dirichletbc(bc_func, dofs)

    A = fem.petsc.assemble_matrix(a, bcs=[bc])
    A.assemble()

    L = fem.form(ufl.inner(c * f * g, v) * (dx + ds))
    b = fem.petsc.assemble_vector(L)
    fem.petsc.apply_lifting(b, [a], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD,
                  mode=PETSc.ScatterMode.REVERSE)
    fem.petsc.set_bc(b, [bc])

    s = mesh.comm.allreduce(fem.assemble_scalar(
        fem.form(ufl.inner(c * f * g, f) * (dx + ds))), op=MPI.SUM)

    return A, b, s


@pytest.mark.parametrize("d", [2, 3])
@pytest.mark.parametrize("n", [2, 6])
@pytest.mark.parametrize("k", [1, 4])
@pytest.mark.parametrize("space", ["Lagrange", "Discontinuous Lagrange"])
@pytest.mark.parametrize("ghost_mode", [GhostMode.none,
                                        GhostMode.shared_facet])
@pytest.mark.parametrize("random_ordering", [False, True])
def test_submesh_cell_assembly(d, n, k, space, ghost_mode, random_ordering):
    """Check that assembling a form over a unit square gives the same
    result as assembling over half of a 2x1 rectangle with the same
    triangulation."""
    if d == 2:
        mesh_0 = create_unit_square(
            MPI.COMM_WORLD, n, n, ghost_mode=ghost_mode)
        if random_ordering:
            mesh_1 = create_random_mesh(((0.0, 0.0), (2.0, 1.0)), (2 * n, n),
                                        ghost_mode=ghost_mode)
        else:
            mesh_1 = create_rectangle(
                MPI.COMM_WORLD, ((0.0, 0.0), (2.0, 1.0)), (2 * n, n),
                ghost_mode=ghost_mode)
    else:
        mesh_0 = create_unit_cube(
            MPI.COMM_WORLD, n, n, n, ghost_mode=ghost_mode)
        mesh_1 = create_box(
            MPI.COMM_WORLD, ((0.0, 0.0, 0.0), (2.0, 1.0, 1.0)),
            (2 * n, n, n), ghost_mode=ghost_mode)

    A_mesh_0, b_mesh_0, s_mesh_0 = assemble_forms_0(mesh_0, space, k)

    edim = mesh_1.topology.dim
    entities = locate_entities(mesh_1, edim, lambda x: x[0] <= 1.0)
    submesh = create_submesh(mesh_1, edim, entities)[0]
    A_submesh, b_submesh, s_submesh = assemble_forms_0(submesh, space, k)

    assert np.isclose(A_mesh_0.norm(), A_submesh.norm())
    assert np.isclose(b_mesh_0.norm(), b_submesh.norm())
    assert np.isclose(s_mesh_0, s_submesh)


@pytest.mark.parametrize("n", [2, 6])
@pytest.mark.parametrize("k", [1, 4])
@pytest.mark.parametrize("space", ["Lagrange", "Discontinuous Lagrange"])
@pytest.mark.parametrize("ghost_mode", [GhostMode.none,
                                        GhostMode.shared_facet])
def test_submesh_facet_assembly(n, k, space, ghost_mode):
    """Test that assembling a form over the face of a unit cube gives
    the same result as assembling it over a unit square."""
    cube_mesh = create_unit_cube(
        MPI.COMM_WORLD, n, n, n, ghost_mode=ghost_mode)
    edim = cube_mesh.topology.dim - 1
    entities = locate_entities_boundary(
        cube_mesh, edim, lambda x: np.isclose(x[2], 0.0))
    submesh = create_submesh(cube_mesh, edim, entities)[0]

    A_submesh, b_submesh, s_submesh = assemble_forms_0(submesh, space, k)

    square_mesh = create_unit_square(
        MPI.COMM_WORLD, n, n, ghost_mode=ghost_mode)
    A_square_mesh, b_square_mesh, s_square_mesh = assemble_forms_0(
        square_mesh, space, k)

    assert np.isclose(A_submesh.norm(), A_square_mesh.norm())
    assert np.isclose(b_submesh.norm(), b_square_mesh.norm())
    assert np.isclose(s_submesh, s_square_mesh)


def assemble_forms_1(comm, f, g, h, u, v, dx, ds, bc, entity_maps={}):
    """Helper function to assemble some forms for testing"""
    a = fem.form(ufl.inner(f[0] * f[1] * g * h * u, v) * (dx + ds),
                 entity_maps=entity_maps)
    A = fem.petsc.assemble_matrix(a, bcs=[bc])
    A.assemble()

    L = fem.form(ufl.inner(f[0] * f[1] * g * h, v) * (dx + ds),
                 entity_maps=entity_maps)
    b = fem.petsc.assemble_vector(L)
    fem.petsc.apply_lifting(b, [a], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD,
                  mode=PETSc.ScatterMode.REVERSE)
    fem.petsc.set_bc(b, [bc])

    M = fem.form(f[0] * f[1] * g * h * (dx + ds), entity_maps=entity_maps)
    s = comm.allreduce(fem.assemble_scalar(M), op=MPI.SUM)

    return A, b, s


@pytest.mark.parametrize("d", [2, 3])
@pytest.mark.parametrize("n", [2, 6])
@pytest.mark.parametrize("k", [1, 4])
@pytest.mark.parametrize("space", ["Lagrange", "Discontinuous Lagrange"])
@pytest.mark.parametrize("ghost_mode", [GhostMode.none,
                                        GhostMode.shared_facet])
@pytest.mark.parametrize("random_ordering", [False, True])
def test_mixed_codim_0_assembly_coeffs(d, n, k, space, ghost_mode,
                                       random_ordering):
    # TODO see if this test can be combined with the below
    """Test that assembling a form where the coefficients are defined on
    different meshes gives the expected result"""

    # Create two meshes. mesh_0 is used to check the result. mesh_1 is
    # used to create submeshes
    if d == 2:
        mesh_0 = create_unit_square(
            MPI.COMM_WORLD, n, n, ghost_mode=ghost_mode)
        if random_ordering:
            mesh_1 = create_random_mesh(((0.0, 0.0), (2.0, 1.0)), (2 * n, n),
                                        ghost_mode=ghost_mode)
        else:
            mesh_1 = create_rectangle(
                MPI.COMM_WORLD, ((0.0, 0.0), (2.0, 1.0)), (2 * n, n),
                ghost_mode=ghost_mode)
    else:
        mesh_0 = create_unit_cube(
            MPI.COMM_WORLD, n, n, n, ghost_mode=ghost_mode)
        mesh_1 = create_box(
            MPI.COMM_WORLD, ((0.0, 0.0, 0.0), (2.0, 1.0, 1.0)),
            (2 * n, n, n), ghost_mode=ghost_mode)

    # Create two different submeshes of mesh_1
    edim = mesh_1.topology.dim
    entities_0 = locate_entities(mesh_1, edim, lambda x: x[0] <= 1.0)
    submesh_0, entity_map_0, vertex_map_0, geom_map_0 = create_submesh(
        mesh_1, edim, entities_0)
    entities_1 = locate_entities(mesh_1, edim, lambda x: x[0] <= 1.5)
    submesh_1, entity_map_1, vertex_map_1, geom_map_1 = create_submesh(
        mesh_1, edim, entities_1)

    # Create function spaces on mesh_1, submesh_0, and submesh_1. We use
    # a Raviart-Thomas space on mesh_1 to test dof transformations
    V_m_1 = fem.FunctionSpace(mesh_1, ("Raviart-Thomas", k))
    V_sm_0 = fem.FunctionSpace(submesh_0, (space, k))
    V_sm_1 = fem.FunctionSpace(submesh_1, (space, k))

    # Create trial and test functions, as well as integration measures
    # on submesh_0
    u_sm = ufl.TrialFunction(V_sm_0)
    v_sm = ufl.TestFunction(V_sm_0)
    dx_sm = ufl.Measure("dx", domain=submesh_0)
    ds_sm = ufl.Measure("ds", domain=submesh_0)

    # Create functions defined over mesh_1, submesh_0, and submesh_1
    # Since all functions are well defined on the integration domain,
    # forms involving them make sense
    f = fem.Function(V_m_1)
    f.interpolate(lambda x: np.vstack([x[i] for i in range(d)]))
    g = fem.Function(V_sm_0)
    g.interpolate(lambda x: x[0]**2)
    h = fem.Function(V_sm_1)
    h.interpolate(lambda x: x[0])

    facet_dim = edim - 1
    facets = locate_entities_boundary(
        submesh_0, facet_dim, lambda x: np.isclose(x[0], 0))
    dofs = fem.locate_dofs_topological(V_sm_0, facet_dim, facets)
    bc_func = fem.Function(V_sm_0)
    bc_func.interpolate(lambda x: x[0]**2)
    bc = fem.dirichletbc(bc_func, dofs)

    # Since the coefficients are defined over meshes that differ from
    # the integration domain mesh (submesh_0), entity maps must be
    # provided. In the case of mesh_1, we must relate the cells in
    # submesh_0 to the cells in mesh_1, which is just the entity map
    # returned from create_submesh. In the case of submesh_1, however,
    # we must relate cells in submesh_0 to cells in submesh_1. This is
    # done by using entity_map_0 to get the cell index in mesh_1,
    # followed by the inverse of entity_map_1 to get the cell index in
    # submesh_1 NOTE: This is done using list comprehension for
    # simplicity, but there are far superior ways of doing this from the
    # perspective of performance
    entity_maps = {mesh_1: entity_map_0,
                   submesh_1: [entity_map_1.index(entity)
                               for entity in entity_map_0]}
    A_sm, b_sm, s_sm = assemble_forms_1(
        submesh_0.comm, f, g, h, u_sm, v_sm, dx_sm, ds_sm, bc, entity_maps)

    # Assemble the same form on a unit square and compare results
    V_m_RT = fem.FunctionSpace(mesh_0, ("Raviart-Thomas", k))
    V_m = fem.FunctionSpace(mesh_0, (space, k))
    f_m = fem.Function(V_m_RT)
    f_m.interpolate(lambda x: np.vstack([x[i] for i in range(d)]))

    g_m = fem.Function(V_m)
    g_m.interpolate(lambda x: x[0]**2)

    h_m = fem.Function(V_m)
    h_m.interpolate(lambda x: x[0])

    u_m = ufl.TrialFunction(V_m)
    v_m = ufl.TestFunction(V_m)

    facets = locate_entities_boundary(
        mesh_0, facet_dim, lambda x: np.isclose(x[0], 0))
    dofs = fem.locate_dofs_topological(V_m, facet_dim, facets)
    bc_func = fem.Function(V_m)
    bc_func.interpolate(lambda x: x[0]**2)
    bc = fem.dirichletbc(bc_func, dofs)

    A_m, b_m, s_m = assemble_forms_1(
        mesh_0.comm, f_m, g_m, h_m, u_m, v_m, ufl.dx, ufl.ds, bc)

    assert np.isclose(A_sm.norm(), A_m.norm())
    assert np.isclose(b_sm.norm(), b_m.norm())
    assert np.isclose(s_sm, s_m)


def compute_expected_norms(d, n, space, k, ghost_mode, f_expr, g_expr):
    """A helper function to assemble some forms on the unit square for
    testing."""
    if d == 2:
        mesh = create_unit_square(MPI.COMM_WORLD, n, n, ghost_mode=ghost_mode)
    else:
        mesh = create_unit_cube(MPI.COMM_WORLD, n, n, n, ghost_mode=ghost_mode)
    V_0 = fem.FunctionSpace(mesh, (space, k))
    V_1 = fem.FunctionSpace(mesh, (space, k))
    u = ufl.TrialFunction(V_0)
    v = ufl.TestFunction(V_1)

    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim - 1, 0)
    f_to_v = mesh.topology.connectivity(tdim - 1, 0)
    num_facets = f_to_v.num_nodes
    facets = create_adjacencylist([f_to_v.links(f)
                                   for f in range(num_facets)])
    facet_values = np.zeros((num_facets), dtype=np.int32)
    left_facets = locate_entities_boundary(
        mesh, tdim - 1, lambda x: np.isclose(x[0], 0.0))
    facet_values[left_facets] = 1
    facet_mt = meshtags_from_entities(mesh, tdim - 1, facets, facet_values)

    ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_mt)

    boundary_facets = locate_entities_boundary(
        mesh, tdim - 1, lambda x: np.isclose(x[1], 1.0))
    dofs = fem.locate_dofs_topological(V_0, tdim - 1, boundary_facets)
    bc_func = fem.Function(V_0)
    bc_func.interpolate(lambda x: x[0])
    bc = fem.dirichletbc(bc_func, dofs)

    f = fem.Function(V_0)
    f.interpolate(f_expr)

    g = fem.Function(V_0)
    g.interpolate(g_expr)

    a = fem.form(ufl.inner(f * g * u, v) * (ufl.dx + ds(1)))
    A = fem.petsc.assemble_matrix(a, bcs=[bc])
    A.assemble()

    L = fem.form(ufl.inner(f * g, v) * (ufl.dx + ds(1)))
    b = fem.petsc.assemble_vector(L)
    fem.petsc.apply_lifting(b, [a], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD,
                  mode=PETSc.ScatterMode.REVERSE)

    return A.norm(), b.norm()


@pytest.mark.parametrize("d", [2, 3])
@pytest.mark.parametrize("n", [2, 6])
@pytest.mark.parametrize("k", [1, 4])
@pytest.mark.parametrize("space", ["Lagrange", "Discontinuous Lagrange"])
@pytest.mark.parametrize("ghost_mode", [GhostMode.none,
                                        GhostMode.shared_facet])
@pytest.mark.parametrize("random_ordering", [False, True])
def test_mixed_codim_0_assembly_0(d, n, k, space, ghost_mode,
                                  random_ordering):
    """Test that assembling a form where the trial and test functions
    are defined on different meshes gives the correct result"""
    # Create a rectangle mesh, and create a submesh of half of it
    if d == 2:
        if random_ordering:
            mesh = create_random_mesh(((0.0, 0.0), (2.0, 1.0)), (2 * n, n),
                                      ghost_mode=ghost_mode)
        else:
            mesh = create_rectangle(
                MPI.COMM_WORLD, ((0.0, 0.0), (2.0, 1.0)), (2 * n, n),
                ghost_mode=ghost_mode)
    else:
        mesh = create_box(
            MPI.COMM_WORLD, ((0.0, 0.0, 0.0), (2.0, 1.0, 1.0)),
            (2 * n, n, n), ghost_mode=ghost_mode)
    tdim = mesh.topology.dim
    edim = tdim
    entities = locate_entities(mesh, edim, lambda x: x[0] <= 1.0)
    submesh, entity_map, vertex_map, geom_map = create_submesh(
        mesh, edim, entities)

    # Create cell meshtags to mark half of the rectangle mesh
    c_to_v = mesh.topology.connectivity(tdim, 0)
    num_cells = c_to_v.num_nodes
    cells = create_adjacencylist([c_to_v.links(c)
                                  for c in range(num_cells)])
    cell_values = np.zeros((num_cells), dtype=np.int32)
    cell_values[entities] = 1
    cell_mt = meshtags_from_entities(mesh, tdim, cells, cell_values)

    # Create facet meshtags to mark the left side of the rectangle
    mesh.topology.create_connectivity(tdim - 1, 0)
    f_to_v = mesh.topology.connectivity(tdim - 1, 0)
    num_facets = f_to_v.num_nodes
    facets = create_adjacencylist([f_to_v.links(f)
                                   for f in range(num_facets)])
    facet_values = np.zeros((num_facets), dtype=np.int32)
    left_facets = locate_entities_boundary(
        mesh, tdim - 1, lambda x: np.isclose(x[0], 0.0))
    facet_values[left_facets] = 1
    facet_mt = meshtags_from_entities(mesh, tdim - 1, facets, facet_values)

    # Create trial and test functions defined on the mesh and submesh
    # respectively
    V_m = fem.FunctionSpace(mesh, (space, k))
    V_sm = fem.FunctionSpace(submesh, (space, k))
    u = ufl.TrialFunction(V_m)
    v = ufl.TestFunction(V_sm)

    # Apply boundary condition to the top boundary
    boundary_facets = locate_entities_boundary(
        mesh, edim - 1, lambda x: np.isclose(x[1], 1.0))
    dofs = fem.locate_dofs_topological(V_m, edim - 1, boundary_facets)
    bc_func = fem.Function(V_m)
    bc_func.interpolate(lambda x: x[0])
    bc = fem.dirichletbc(bc_func, dofs)

    # Create integration measure on the mesh
    dx = ufl.Measure("dx", domain=mesh, subdomain_data=cell_mt)
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_mt)

    def f_expr(x):
        return 1 + x[0]

    def g_expr(x):
        return 1 + x[1]**2

    f = fem.Function(V_m)
    f.interpolate(f_expr)

    g = fem.Function(V_sm)
    g.interpolate(g_expr)

    # Create entity map from mesh cells to submesh cells (this is the inverse
    # of the entity map provided by create_submesh)
    entity_maps = {submesh: [entity_map.index(entity)
                             if entity in entity_map else -1
                             for entity in range(num_cells)]}
    # Create and assemble some forms
    a = fem.form(ufl.inner(f * g * u, v) * (dx(1) + ds(1)),
                 entity_maps=entity_maps)
    A = fem.petsc.assemble_matrix(a, bcs=[bc])
    A.assemble()

    L = fem.form(ufl.inner(f * g, v) * (dx(1) + ds(1)),
                 entity_maps=entity_maps)
    b = fem.petsc.assemble_vector(L)
    fem.petsc.apply_lifting(b, [a], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD,
                  mode=PETSc.ScatterMode.REVERSE)

    # Compute expected norms and compare
    A_expected_norm, b_expected_norm = compute_expected_norms(
        d, n, space, k, ghost_mode, f_expr, g_expr)
    assert np.isclose(A.norm(), A_expected_norm)
    assert np.isclose(b.norm(), b_expected_norm)


@pytest.mark.parametrize("d", [2, 3])
@pytest.mark.parametrize("n", [2, 6])
@pytest.mark.parametrize("k", [1, 4])
@pytest.mark.parametrize("space", ["Lagrange", "Discontinuous Lagrange"])
@pytest.mark.parametrize("ghost_mode", [GhostMode.none,
                                        GhostMode.shared_facet])
@pytest.mark.parametrize("random_ordering", [False, True])
def test_mixed_codim_0_assembly_1(d, n, k, space, ghost_mode, random_ordering):
    """Same test as test_mixed_codim_0_assembly_0, but this time assembling
    with respect to the submesh rather than the mesh."""
    if d == 2:
        if random_ordering:
            mesh = create_random_mesh(((0.0, 0.0), (2.0, 1.0)), (2 * n, n),
                                      ghost_mode=ghost_mode)
        else:
            mesh = create_rectangle(
                MPI.COMM_WORLD, ((0.0, 0.0), (2.0, 1.0)), (2 * n, n),
                ghost_mode=ghost_mode)
    else:
        mesh = create_box(
            MPI.COMM_WORLD, ((0.0, 0.0, 0.0), (2.0, 1.0, 1.0)),
            (2 * n, n, n), ghost_mode=ghost_mode)
    tdim = mesh.topology.dim
    edim = tdim
    entities = locate_entities(mesh, edim, lambda x: x[0] <= 1.0)
    submesh, entity_map, vertex_map, geom_map = create_submesh(
        mesh, edim, entities)

    submesh.topology.create_connectivity(tdim - 1, 0)
    sm_f_to_v = submesh.topology.connectivity(tdim - 1, 0)
    sm_num_facets = sm_f_to_v.num_nodes
    sm_facets = create_adjacencylist([sm_f_to_v.links(f)
                                      for f in range(sm_num_facets)])
    sm_facet_values = np.zeros((sm_num_facets), dtype=np.int32)
    left_facets = locate_entities_boundary(
        submesh, tdim - 1, lambda x: np.isclose(x[0], 0.0))
    sm_facet_values[left_facets] = 1
    sm_facet_mt = meshtags_from_entities(
        submesh, tdim - 1, sm_facets, sm_facet_values)

    V_m = fem.FunctionSpace(mesh, (space, k))
    V_sm = fem.FunctionSpace(submesh, (space, k))
    u = ufl.TrialFunction(V_m)
    v = ufl.TestFunction(V_sm)

    boundary_facets = locate_entities_boundary(
        mesh, edim - 1, lambda x: np.isclose(x[1], 1.0))
    dofs = fem.locate_dofs_topological(V_m, edim - 1, boundary_facets)
    bc_func = fem.Function(V_m)
    bc_func.interpolate(lambda x: x[0])
    bc = fem.dirichletbc(bc_func, dofs)

    dx = ufl.Measure("dx", domain=submesh)
    ds = ufl.Measure("ds", domain=submesh, subdomain_data=sm_facet_mt)

    def f_expr(x):
        return 1 + x[0]

    def g_expr(x):
        return 1 + x[1]**2

    f = fem.Function(V_m)
    f.interpolate(f_expr)

    g = fem.Function(V_sm)
    g.interpolate(g_expr)

    entity_maps = {mesh: entity_map}
    a = fem.form(ufl.inner(f * g * u, v) * (dx + ds(1)),
                 entity_maps=entity_maps)
    A = fem.petsc.assemble_matrix(a, bcs=[bc])
    A.assemble()

    L = fem.form(ufl.inner(f * g, v) * (dx + ds(1)),
                 entity_maps=entity_maps)
    b = fem.petsc.assemble_vector(L)
    fem.petsc.apply_lifting(b, [a], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD,
                  mode=PETSc.ScatterMode.REVERSE)

    A_expected_norm, b_expected_norm = compute_expected_norms(
        d, n, space, k, ghost_mode, f_expr, g_expr)
    assert np.isclose(A.norm(), A_expected_norm)
    assert np.isclose(b.norm(), b_expected_norm)


@pytest.mark.parametrize("d", [2, 3])
@pytest.mark.parametrize("n", [2, 6])
@pytest.mark.parametrize("k", [1, 4])
@pytest.mark.parametrize("space", ["Lagrange", "Discontinuous Lagrange"])
@pytest.mark.parametrize("ghost_mode", [GhostMode.none,
                                        GhostMode.shared_facet])
@pytest.mark.parametrize("random_ordering", [False, True])
def test_codim_1_coeffs(d, n, k, space, ghost_mode, random_ordering):
    """Test that assembling forms with coefficients defined only on the
    boundary of the mesh gives the expected result."""
    if d == 2:
        if random_ordering:
            mesh = create_random_mesh(((0.0, 0.0), (2.0, 1.0)), (2 * n, n),
                                      ghost_mode=ghost_mode)
        else:
            mesh = create_rectangle(
                MPI.COMM_WORLD, ((0.0, 0.0), (2.0, 1.0)), (2 * n, n),
                ghost_mode=ghost_mode)

        def boundary_marker(x):
            return np.logical_or(np.logical_or(np.isclose(x[0], 2.0),
                                               np.isclose(x[0], 0.0)),
                                 np.logical_or(np.isclose(x[1], 1.0),
                                               np.isclose(x[1], 0.0)))
    else:
        mesh = create_box(
            MPI.COMM_WORLD, ((0.0, 0.0, 0.0), (2.0, 1.0, 1.0)),
            (2 * n, n, n), ghost_mode=ghost_mode)

        def boundary_marker(x):
            b_0 = np.logical_or(np.isclose(x[0], 2.0), np.isclose(x[0], 0.0))
            b_1 = np.logical_or(np.isclose(x[1], 1.0), np.isclose(x[1], 0.0))
            b_2 = np.logical_or(np.isclose(x[2], 1.0), np.isclose(x[2], 0.0))
            return np.logical_or(np.logical_or(b_0, b_1), b_2)

    # Create a submesh of the boundary of the mesh
    edim = mesh.topology.dim - 1
    num_facets = mesh.topology.create_entities(edim)
    entities = locate_entities_boundary(mesh, edim, boundary_marker)
    submesh, entity_map, vertex_map, geom_map = create_submesh(
        mesh, edim, entities)

    element = (space, k)
    V_m = fem.FunctionSpace(mesh, element)
    V_sm = fem.FunctionSpace(submesh, element)

    # Create a coefficient on submesh and the mesh (for comparison)
    f = fem.Function(V_sm)
    f.interpolate(lambda x: x[0])
    f_m = fem.Function(V_m)
    f_m.interpolate(lambda x: x[0])

    u = ufl.TrialFunction(V_m)
    v = ufl.TestFunction(V_m)

    # Create a map from facets in the mesh to facets in the submesh
    # (the inverse of the entity map retuned by create_submesh)
    entity_maps = {submesh: [entity_map.index(entity)
                             if entity in entity_map else -1
                             for entity in range(num_facets)]}

    # Assemble a matrix and compare with coefficient defined over whole mesh
    ds = ufl.Measure("ds", domain=mesh)
    a = fem.form(ufl.inner(f * u, v) * ds,
                 entity_maps=entity_maps)
    A = fem.petsc.assemble_matrix(a)
    A.assemble()
    A_norm = A.norm()

    a = fem.form(ufl.inner(f_m * u, v) * ds)
    A = fem.petsc.assemble_matrix(a)
    A.assemble()
    A_expected_norm = A.norm()
    assert np.isclose(A_norm, A_expected_norm)

    # Assemble a vector and compare
    L = fem.form(ufl.inner(f, v) * ds,
                 entity_maps=entity_maps)
    b = fem.petsc.assemble_vector(L)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD,
                  mode=PETSc.ScatterMode.REVERSE)
    b_norm = b.norm()

    L = fem.form(ufl.inner(f_m, v) * ds)
    b = fem.petsc.assemble_vector(L)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD,
                  mode=PETSc.ScatterMode.REVERSE)
    b_expected_norm = b.norm()
    assert np.isclose(b_norm, b_expected_norm)

    # Assemble a scalar and compare
    M = fem.form(f * ds,
                 entity_maps=entity_maps)
    s = mesh.comm.allreduce(fem.assemble_scalar(M), op=MPI.SUM)

    M = fem.form(f_m * ds)
    s_expected = mesh.comm.allreduce(fem.assemble_scalar(M), op=MPI.SUM)
    assert np.isclose(s, s_expected)


@pytest.mark.parametrize("d", [2, 3])
@pytest.mark.parametrize("n", [2, 6])
@pytest.mark.parametrize("k", [1, 4])
@pytest.mark.parametrize("space", ["Lagrange"])
@pytest.mark.parametrize("ghost_mode", [GhostMode.none,
                                        GhostMode.shared_facet])
@pytest.mark.parametrize("random_ordering", [False, True])
def test_codim_1_assembly(d, n, k, space, ghost_mode, random_ordering):
    """Test that assembling a form with a trial function defined over
    the mesh and a test function defined only over the mesh boundary
    gives the expected result"""
    # TODO Test discontinuous Lagrange spaces. Can't just compare to
    # the same spaces on the mesh in that case because a discontinuous
    # Lagrange space on the boundary facets will be discontinuous at
    # corners, which is not the case for a discontinuous Lagrange
    # space on the cells of the mesh.
    if d == 2:
        if random_ordering:
            mesh = create_random_mesh(((0.0, 0.0), (2.0, 1.0)), (2 * n, n),
                                      ghost_mode=ghost_mode)
        else:
            mesh = create_rectangle(
                MPI.COMM_WORLD, ((0.0, 0.0), (2.0, 1.0)), (2 * n, n),
                ghost_mode=ghost_mode)

        def boundary_marker(x):
            return np.logical_or(np.logical_or(np.isclose(x[0], 2.0),
                                               np.isclose(x[0], 0.0)),
                                 np.logical_or(np.isclose(x[1], 1.0),
                                               np.isclose(x[1], 0.0)))
    else:
        mesh = create_box(
            MPI.COMM_WORLD, ((0.0, 0.0, 0.0), (2.0, 1.0, 1.0)),
            (2 * n, n, n), ghost_mode=ghost_mode)

        def boundary_marker(x):
            b_0 = np.logical_or(np.isclose(x[0], 2.0), np.isclose(x[0], 0.0))
            b_1 = np.logical_or(np.isclose(x[1], 1.0), np.isclose(x[1], 0.0))
            b_2 = np.logical_or(np.isclose(x[2], 1.0), np.isclose(x[2], 0.0))
            return np.logical_or(np.logical_or(b_0, b_1), b_2)

    edim = mesh.topology.dim - 1
    num_facets = mesh.topology.create_entities(edim)
    entities = locate_entities_boundary(
        mesh, edim, boundary_marker)
    submesh, entity_map, vertex_map, geom_map = create_submesh(
        mesh, edim, entities)

    element = (space, k)
    V_m = fem.FunctionSpace(mesh, element)
    V_sm = fem.FunctionSpace(submesh, element)

    u_m = ufl.TrialFunction(V_m)
    v_m = ufl.TestFunction(V_m)
    v_sm = ufl.TestFunction(V_sm)

    f = fem.Function(V_m)
    f.interpolate(lambda x: np.cos(np.pi * x[0]))

    def g_expr(x):
        return 1.0 + x[0]**2

    g_m = fem.Function(V_m)
    g_m.interpolate(g_expr)

    g_sm = fem.Function(V_sm)
    g_sm.interpolate(g_expr)

    ds = ufl.Measure("ds", domain=mesh)
    mp = [entity_map.index(entity) if entity in entity_map else -1
          for entity in range(num_facets)]
    entity_maps = {submesh: mp}
    a = fem.form(ufl.inner(f * g_sm * u_m, v_sm) * ds,
                 entity_maps=entity_maps)
    A = fem.petsc.assemble_matrix(a)
    A.assemble()

    a_2 = fem.form(ufl.inner(f * g_m * u_m, v_m) * ds)
    A_2 = fem.petsc.assemble_matrix(a_2)
    A_2.assemble()

    assert np.isclose(A.norm(), A_2.norm())

    L = fem.form(ufl.inner(f * g_sm, v_sm) * ds,
                 entity_maps=entity_maps)
    b = fem.petsc.assemble_vector(L)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD,
                  mode=PETSc.ScatterMode.REVERSE)

    L_2 = fem.form(ufl.inner(f * g_m, v_m) * ds)
    b_2 = fem.petsc.assemble_vector(L_2)
    b_2.ghostUpdate(addv=PETSc.InsertMode.ADD,
                    mode=PETSc.ScatterMode.REVERSE)

    assert np.isclose(b.norm(), b_2.norm())


@pytest.mark.parametrize("random_ordering", [False, True])
def test_assemble_block(random_ordering):
    n = 2
    if random_ordering:
        msh = create_random_mesh(((0.0, 0.0), (1.0, 1.0)), (n, n),
                                 ghost_mode=GhostMode.shared_facet)
    else:
        msh = create_unit_square(MPI.COMM_WORLD, n, n)
    edim = msh.topology.dim - 1
    num_facets = msh.topology.create_entities(edim)
    entities = locate_entities_boundary(
        msh, edim,
        lambda x: np.logical_or(np.logical_or(np.isclose(x[0], 1.0),
                                              np.isclose(x[0], 0.0)),
                                np.logical_or(np.isclose(x[1], 1.0),
                                              np.isclose(x[1], 0.0))))
    submesh, entity_map, vertex_map, geom_map = create_submesh(
        msh, edim, entities)

    V = fem.FunctionSpace(msh, ("Lagrange", 1))
    W = fem.FunctionSpace(submesh, ("Lagrange", 1))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    lmbda = ufl.TrialFunction(W)
    mu = ufl.TestFunction(W)

    dx_m = ufl.Measure("dx", domain=msh)
    dx_sm = ufl.Measure("dx", domain=submesh)
    ds = ufl.Measure("ds", domain=msh)

    mp = [entity_map.index(entity) if entity in entity_map else -1
          for entity in range(num_facets)]
    entity_maps = {submesh: mp}

    a_00 = fem.form(ufl.inner(u, v) * dx_m)
    a_01 = fem.form(ufl.inner(lmbda, v) * ds, entity_maps=entity_maps)
    a_10 = fem.form(ufl.inner(u, mu) * ds, entity_maps=entity_maps)
    a_11 = fem.form(ufl.inner(lmbda, mu) * dx_sm)
    L_0 = fem.form(ufl.inner(1.0, v) * dx_m)
    L_1 = fem.form(ufl.inner(1.0, mu) * dx_sm)

    dofs = fem.locate_dofs_topological(V, edim, entities)
    bc = fem.dirichletbc(PETSc.ScalarType(0.0), dofs, V)

    a = [[a_00, a_01],
         [a_10, a_11]]
    L = [L_0, L_1]

    A = fem.petsc.assemble_matrix_block(a, bcs=[bc])
    A.assemble()
    b = fem.petsc.assemble_vector_block(L, a, bcs=[bc])

    # TODO Check value
    assert np.isclose(A.norm(), 3.0026030373660784)
    assert np.isclose(b.norm(), 1.4361406616345072)


def test_mixed_coeff_form():
    """Test that a form with coefficients involving dx and ds integrals assembles as expected"""
    n = 4
    msh = create_unit_square(MPI.COMM_WORLD, n, n)
    fdim = msh.topology.dim - 1
    num_facets = msh.topology.create_entities(fdim)
    boundary_facets = locate_entities_boundary(
        msh, fdim, lambda x: np.logical_or(np.logical_or(np.isclose(x[0], 0.0),
                                                         np.isclose(x[0], 1.0)),
                                           np.logical_or(np.isclose(x[1], 0.0),
                                                         np.isclose(x[1], 1.0))))
    submesh, entity_map = create_submesh(msh, fdim, boundary_facets)[0:2]

    # Create function spaces
    V = fem.FunctionSpace(msh, ("Lagrange", 1))
    W = fem.FunctionSpace(submesh, ("Lagrange", 1))
    v = ufl.TestFunction(V)

    # Create integration measure and entity maps
    ds = ufl.Measure("ds", domain=msh)
    entity_maps = {submesh: [entity_map.index(entity)
                             if entity in entity_map else -1
                             for entity in range(num_facets)]}

    # Define forms
    f = fem.Function(V)
    f.interpolate(lambda x: x[0])
    g = fem.Function(W)
    g.interpolate(lambda x: x[1])
    L = fem.form(ufl.inner(f, v) * ufl.dx + ufl.inner(g, v) * ds, entity_maps=entity_maps)

    b = fem.petsc.assemble_vector(L)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD,
                  mode=PETSc.ScatterMode.REVERSE)

    # TODO Check value
    assert np.isclose(b.norm(), 0.6937782992069734)


def reorder_mesh(msh):
    # FIXME Check this is correct
    # FIXME Needs generalising for high-order mesh
    # FIXME What about quads / hexes?
    tdim = msh.topology.dim
    num_cell_vertices = cell_num_entities(msh.topology.cell_type, 0)
    c_to_v = msh.topology.connectivity(tdim, 0)
    geom_dofmap = msh.geometry.dofmap
    vertex_imap = msh.topology.index_map(0)
    geom_imap = msh.geometry.index_map()
    for i in range(0, len(c_to_v.array), num_cell_vertices):
        topo_perm = np.argsort(vertex_imap.local_to_global(
            c_to_v.array[i:i + num_cell_vertices]))
        geom_perm = np.argsort(geom_imap.local_to_global(
            geom_dofmap.array[i:i + num_cell_vertices]))

        c_to_v.array[i:i + num_cell_vertices] = \
            c_to_v.array[i:i + num_cell_vertices][topo_perm]
        geom_dofmap.array[i:i + num_cell_vertices] = \
            geom_dofmap.array[i:i + num_cell_vertices][geom_perm]


@pytest.mark.parametrize("n", [2, 4, 8])
@pytest.mark.parametrize("d", [2, 3])
@pytest.mark.parametrize("random_ordering", [False, True])
def test_int_facet(n, d, random_ordering):
    if d == 2:
        if random_ordering:
            msh_0 = create_random_mesh(((0.0, 0.0), (2.0, 1.0)), (2 * n, n),
                                      ghost_mode=GhostMode.shared_facet)
        else:
            msh_0 = create_rectangle(
                MPI.COMM_WORLD, ((0.0, 0.0), (2.0, 1.0)), (2 * n, n),
                ghost_mode=GhostMode.shared_facet)
        msh_1 = create_unit_square(MPI.COMM_WORLD, n, n,
                                   ghost_mode=GhostMode.shared_facet)
    else:
        msh_0 = create_box(
            MPI.COMM_WORLD, ((0.0, 0.0, 0.0), (2.0, 1.0, 1.0)),
            (2 * n, n, n), ghost_mode=GhostMode.shared_facet)
        msh_1 = create_unit_cube(MPI.COMM_WORLD, n, n, n,
                                 ghost_mode=GhostMode.shared_facet)

    tdim = msh_0.topology.dim
    entities = locate_entities(msh_0, tdim, lambda x: x[0] <= 1.0)
    submesh, entity_map = create_submesh(
        msh_0, tdim, entities)[:2]

    V_msh = fem.FunctionSpace(msh_0, ("Lagrange", 1))
    V_submesh = fem.FunctionSpace(submesh, ("Lagrange", 1))

    u, v = ufl.TrialFunction(V_msh), ufl.TestFunction(V_submesh)

    boundary_facets = locate_entities_boundary(
        msh_0, tdim - 1, lambda x: np.isclose(x[1], 1.0))
    dofs = fem.locate_dofs_topological(V_msh, tdim - 1, boundary_facets)
    bc_func = fem.Function(V_msh)
    bc_func.interpolate(lambda x: x[0])
    bc = fem.dirichletbc(bc_func, dofs)

    # TODO Remove duplicate code
    dS = ufl.Measure("dS", domain=submesh)
    x = ufl.SpatialCoordinate(submesh)
    c = fem.Function(V_msh)
    c.interpolate(lambda x: 1 + x[0]**2)
    a = fem.form(ufl.inner((1 + x[0]) * c * u("+"), v("-")) * dS, entity_maps={msh_0: entity_map})
    A = fem.petsc.assemble_matrix(a, bcs=[bc])
    A.assemble()
    A_norm = A.norm()

    L = fem.form(ufl.inner((1 + x[0]) * c, v("-")) * dS, entity_maps={msh_0: entity_map})
    b = fem.petsc.assemble_vector(L)
    fem.petsc.apply_lifting(b, [a], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD,
                  mode=PETSc.ScatterMode.REVERSE)
    b_norm = b.norm()

    V_0 = fem.FunctionSpace(msh_1, ("Lagrange", 1))
    V_1 = fem.FunctionSpace(msh_1, ("Lagrange", 1))
    u, v = ufl.TrialFunction(V_0), ufl.TestFunction(V_1)

    boundary_facets = locate_entities_boundary(
        msh_1, tdim - 1, lambda x: np.isclose(x[1], 1.0))
    dofs = fem.locate_dofs_topological(V_0, tdim - 1, boundary_facets)
    bc_func = fem.Function(V_0)
    bc_func.interpolate(lambda x: x[0])
    bc = fem.dirichletbc(bc_func, dofs)

    x = ufl.SpatialCoordinate(msh_1)
    c = fem.Function(V_0)
    c.interpolate(lambda x: 1 + x[0]**2)
    a = fem.form(ufl.inner((1 + x[0]) * c * u("+"), v("-")) * ufl.dS)
    A = fem.petsc.assemble_matrix(a, bcs=[bc])
    A.assemble()
    assert np.isclose(A_norm, A.norm())

    L = fem.form(ufl.inner((1 + x[0]) * c, v("-")) * ufl.dS)
    b = fem.petsc.assemble_vector(L)
    fem.petsc.apply_lifting(b, [a], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD,
                  mode=PETSc.ScatterMode.REVERSE)
    assert np.isclose(b_norm, b.norm())
