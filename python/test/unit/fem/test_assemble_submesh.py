# Copyright (C) 2022 Joseph P. Dean
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

# TODO Test replacing mesh with submesh for existing assembler tests

import numpy as np
import pytest

import ufl
from dolfinx import fem
from dolfinx.mesh import (GhostMode, create_box, create_rectangle,
                          create_submesh, create_unit_cube, create_unit_square,
                          locate_entities, locate_entities_boundary,
                          meshtags_from_entities)

from mpi4py import MPI
from petsc4py import PETSc
from dolfinx.graph import create_adjacencylist


def assemble_forms_0(mesh, space, k):
    V = fem.FunctionSpace(mesh, (space, k))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    dx = ufl.Measure("dx", domain=mesh)
    ds = ufl.Measure("ds", domain=mesh)

    c = fem.Constant(mesh, PETSc.ScalarType(0.75))
    a = fem.form(ufl.inner(c * u, v) * (dx + ds))

    facet_dim = mesh.topology.dim - 1
    facets = locate_entities_boundary(
        mesh, facet_dim, lambda x: np.isclose(x[0], 1))
    dofs = fem.locate_dofs_topological(V, facet_dim, facets)

    bc_func = fem.Function(V)
    # TODO Interpolate when issue #2126 has been resolved
    bc_func.x.array[:] = 1.0

    bc = fem.dirichletbc(bc_func, dofs)

    A = fem.petsc.assemble_matrix(a, bcs=[bc])
    A.assemble()

    # TODO Test assembly with fem.Function
    x = ufl.SpatialCoordinate(mesh)
    f = 1.5 + x[0]
    L = fem.form(ufl.inner(c * f, v) * (dx + ds))
    b = fem.petsc.assemble_vector(L)
    fem.petsc.apply_lifting(b, [a], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD,
                  mode=PETSc.ScatterMode.REVERSE)
    fem.petsc.set_bc(b, [bc])

    s = mesh.comm.allreduce(fem.assemble_scalar(
        fem.form(ufl.inner(c * f, f) * (dx + ds))), op=MPI.SUM)

    return A, b, s


@pytest.mark.parametrize("d", [2, 3])
@pytest.mark.parametrize("n", [2, 6])
@pytest.mark.parametrize("k", [1, 4])
@pytest.mark.parametrize("space", ["Lagrange", "Discontinuous Lagrange"])
@pytest.mark.parametrize("ghost_mode", [GhostMode.none,
                                        GhostMode.shared_facet])
def test_submesh_cell_assembly(d, n, k, space, ghost_mode):
    """Check that assembling a form over a unit square gives the same
    result as assembling over half of a 2x1 rectangle with the same
    triangulation."""
    if d == 2:
        mesh_0 = create_unit_square(
            MPI.COMM_WORLD, n, n, ghost_mode=ghost_mode)
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

    assert(np.isclose(A_mesh_0.norm(), A_submesh.norm()))
    assert(np.isclose(b_mesh_0.norm(), b_submesh.norm()))
    assert(np.isclose(s_mesh_0, s_submesh))


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

    assert(np.isclose(A_submesh.norm(), A_square_mesh.norm()))
    assert(np.isclose(b_submesh.norm(), b_square_mesh.norm()))
    assert(np.isclose(s_submesh, s_square_mesh))


def assemble_forms_1(comm, f, g, h, u, v, dx, ds, entity_maps={}):
    # TODO Add ds
    a = fem.form(ufl.inner(f[0] * f[1] * g * h * u, v) * (dx + ds),
                 entity_maps=entity_maps)
    A = fem.petsc.assemble_matrix(a)
    A.assemble()

    L = fem.form(ufl.inner(f[0] * f[1] * g * h, v) * (dx + ds),
                 entity_maps=entity_maps)
    b = fem.petsc.assemble_vector(L)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD,
                  mode=PETSc.ScatterMode.REVERSE)

    M = fem.form(f[0] * f[1] * g * h * (dx + ds), entity_maps=entity_maps)
    s = comm.allreduce(fem.assemble_scalar(M), op=MPI.SUM)

    return A, b, s


@pytest.mark.parametrize("d", [2, 3])
@pytest.mark.parametrize("n", [2, 6])
@pytest.mark.parametrize("k", [1, 4])
@pytest.mark.parametrize("space", ["Lagrange", "Discontinuous Lagrange"])
@pytest.mark.parametrize("ghost_mode", [GhostMode.none,
                                        GhostMode.shared_facet])
def test_mixed_codim_0_assembly(d, n, k, space, ghost_mode):
    """Test that assembling a form where the coefficients are defined on
    different meshes gives the expected result"""

    # Create two meshes. mesh_0 is used to check the result. mesh_1 is
    # used to create submeshes
    if d == 2:
        mesh_0 = create_unit_square(
            MPI.COMM_WORLD, n, n, ghost_mode=ghost_mode)
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
    # TODO Interpolate g and h when issue #2126 has been resolved
    g.x.array[:] = 1.0
    h = fem.Function(V_sm_1)
    h.x.array[:] = 2.0

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
        submesh_0.comm, f, g, h, u_sm, v_sm, dx_sm, ds_sm, entity_maps)

    # Assemble the same form on a unit square and compare results
    V_m_RT = fem.FunctionSpace(mesh_0, ("Raviart-Thomas", k))
    V_m = fem.FunctionSpace(mesh_0, (space, k))
    f_m = fem.Function(V_m_RT)
    f_m.interpolate(lambda x: np.vstack([x[i] for i in range(d)]))

    g_m = fem.Function(V_m)
    g_m.x.array[:] = 1.0

    h_m = fem.Function(V_m)
    h_m.x.array[:] = 2.0

    u_m = ufl.TrialFunction(V_m)
    v_m = ufl.TestFunction(V_m)

    A_m, b_m, s_m = assemble_forms_1(
        mesh_0.comm, f_m, g_m, h_m, u_m, v_m, ufl.dx, ufl.ds)

    assert(np.isclose(A_sm.norm(), A_m.norm()))
    assert(np.isclose(b_sm.norm(), b_m.norm()))
    assert(np.isclose(s_sm, s_m))


@pytest.mark.parametrize("n", [2, 6])
@pytest.mark.parametrize("k", [1, 4])
@pytest.mark.parametrize("space", ["Lagrange", "Discontinuous Lagrange"])
@pytest.mark.parametrize("ghost_mode", [GhostMode.none,
                                        GhostMode.shared_facet])
def test_mixed_codim_0_test_func_assembly(n, k, space, ghost_mode):
    # TODO Test vector is correct
    mesh = create_rectangle(
        MPI.COMM_WORLD, ((0.0, 0.0), (2.0, 1.0)), (2 * n, n),
        ghost_mode=ghost_mode)
    tdim = mesh.topology.dim
    edim = tdim
    entities = locate_entities(mesh, edim, lambda x: x[0] <= 1.0)
    submesh, entity_map, vertex_map, geom_map = create_submesh(
        mesh, edim, entities)

    c_to_v = mesh.topology.connectivity(tdim, 0)
    num_cells = c_to_v.num_nodes
    entities_mt = create_adjacencylist([c_to_v.links(c)
                                        for c in range(num_cells)])
    values = np.zeros((num_cells), dtype=np.int32)
    values[entities] = 1
    mt = meshtags_from_entities(mesh, tdim, entities_mt, values)

    V_sm = fem.FunctionSpace(submesh, (space, k))
    v = ufl.TestFunction(V_sm)
    dx = ufl.Measure("dx", domain=mesh, subdomain_data=mt)

    mp = [entity_map.index(entity) if entity in entity_map else -1
          for entity in range(num_cells)]

    entity_maps = {submesh: mp}
    L = fem.form(v * dx(1),
                 entity_maps=entity_maps)
    b = fem.petsc.assemble_vector(L)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD,
                  mode=PETSc.ScatterMode.REVERSE)
    print(b[:])


# def test_mixed_mesh_codim_0_assembly(n, k, space, ghost_mode):
    # TODO Add test to check vector is actually correct
    # mesh = create_rectangle(
    #     MPI.COMM_WORLD, ((0.0, 0.0), (2.0, 1.0)), (2 * n, n),
    #     ghost_mode=ghost_mode)
    # edim = mesh.topology.dim
    # entities = locate_entities(mesh, edim, lambda x: x[0] <= 1.0)
    # submesh, entity_map, vertex_map, geom_map = create_submesh(
    #     mesh, edim, entities)

    # element = (space, k)
    # V_m = fem.FunctionSpace(mesh, element)

    # v = ufl.TestFunction(V_m)
    # dx_sm = ufl.Measure("dx", domain=submesh)
    # ds_sm = ufl.Measure("ds", domain=submesh)

    # entity_maps = {mesh: entity_map}
    # L = fem.form(v * (dx_sm + ds_sm),
    #              entity_maps=entity_maps)
    # b = fem.petsc.assemble_vector(L)
    # b.ghostUpdate(addv=PETSc.InsertMode.ADD,
    #               mode=PETSc.ScatterMode.REVERSE)

    # norm_0 = b.norm()
    # print(norm_0)


# np.set_printoptions(linewidth=200, suppress=True)
# n = 2
# ghost_mode = GhostMode.none
# space = "Lagrange"
# k = 1

# mesh = create_rectangle(
#     MPI.COMM_WORLD, ((0.0, 0.0), (2.0, 1.0)), (2 * n, n),
#     ghost_mode=ghost_mode)
# edim = mesh.topology.dim - 1
# num_facets = mesh.topology.create_entities(edim)
# entities = locate_entities_boundary(mesh, edim, lambda x: np.logical_or(np.logical_or(np.isclose(x[0], 2.0),
#                                                                                       np.isclose(x[0], 0.0)),
#                                                                         np.logical_or(np.isclose(x[1], 1.0),
#                                                                                       np.isclose(x[1], 0.0))))
# submesh, entity_map, vertex_map, geom_map = create_submesh(
#     mesh, edim, entities)

# element = (space, k)
# V_m = fem.FunctionSpace(mesh, element)
# V_sm = fem.FunctionSpace(submesh, element)

# f = fem.Function(V_sm)
# f.interpolate(lambda x: x[0])

# from dolfinx import io
# with io.XDMFFile(submesh.comm, "submesh.xdmf", "w") as file:
#     file.write_mesh(submesh)
#     file.write_function(f)

# mp = [entity_map.index(entity) if entity in entity_map else -1 for entity in range(num_facets)]

# v = ufl.TestFunction(V_m)

# ds = ufl.Measure("ds", domain=mesh)
# entity_maps = {submesh: mp}
# L = fem.form(ufl.inner(f, v) * ds,
#              entity_maps=entity_maps)
# b = fem.petsc.assemble_vector(L)
# print(b[:])

# f = fem.Function(V_m)
# f.interpolate(lambda x: x[0])
# L = fem.form(ufl.inner(f, v) * ds)
# b = fem.petsc.assemble_vector(L)
# print(b[:])


# u = ufl.TrialFunction(V_sm)
# v = ufl.TestFunction(V_m)
# dx_sm = ufl.Measure("dx", domain=submesh)
# ds_sm = ufl.Measure("ds", domain=submesh)

# entity_maps = {mesh: entity_map}
# a = fem.form(ufl.inner(u, v) * (dx_sm + ds_sm),
#              entity_maps=entity_maps)
# A = fem.petsc.assemble_matrix(a)
# A.assemble()

# print(A[:, :])


# mesh = create_rectangle(
#     MPI.COMM_WORLD, ((0.0, 0.0), (2.0, 1.0)), (2 * n, n),
#     ghost_mode=ghost_mode)
# tdim = mesh.topology.dim
# edim = tdim
# entities = locate_entities(mesh, edim, lambda x: x[0] <= 1.0)
# submesh, entity_map, vertex_map, geom_map = create_submesh(
#     mesh, edim, entities)

# c_to_v = mesh.topology.connectivity(tdim, 0)
# num_cells = c_to_v.num_nodes
# entities_mt = create_adjacencylist([c_to_v.links(c)
#                                     for c in range(num_cells)])
# values = np.zeros((num_cells), dtype=np.int32)
# values[entities] = 1
# mt = meshtags_from_entities(mesh, tdim, entities_mt, values)

# V_sm = fem.FunctionSpace(submesh, (space, k))
# v = ufl.TestFunction(V_sm)
# dx = ufl.Measure("dx", domain=mesh, subdomain_data=mt)

# mp = [entity_map.index(entity) if entity in entity_map else -1
#       for entity in range(num_cells)]

# entity_maps = {submesh: mp}
# L = fem.form(v * dx(1),
#              entity_maps=entity_maps)
# b = fem.petsc.assemble_vector(L)
# b.ghostUpdate(addv=PETSc.InsertMode.ADD,
#               mode=PETSc.ScatterMode.REVERSE)
# print(b[:])
# norm_0 = b.norm()
# print(norm_0)
