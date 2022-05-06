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
                          locate_entities, locate_entities_boundary)

from mpi4py import MPI
from petsc4py import PETSc


def assemble(mesh, space, k):
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

    A_mesh_0, b_mesh_0, s_mesh_0 = assemble(mesh_0, space, k)

    edim = mesh_1.topology.dim
    entities = locate_entities(mesh_1, edim, lambda x: x[0] <= 1.0)
    submesh = create_submesh(mesh_1, edim, entities)[0]
    A_submesh, b_submesh, s_submesh = assemble(submesh, space, k)

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

    A_submesh, b_submesh, s_submesh = assemble(submesh, space, k)

    square_mesh = create_unit_square(
        MPI.COMM_WORLD, n, n, ghost_mode=ghost_mode)
    A_square_mesh, b_square_mesh, s_square_mesh = assemble(
        square_mesh, space, k)

    assert(np.isclose(A_submesh.norm(), A_square_mesh.norm()))
    assert(np.isclose(b_submesh.norm(), b_square_mesh.norm()))
    assert(np.isclose(s_submesh, s_square_mesh))


np.set_printoptions(linewidth=200)
n = 2
ghost_mode = GhostMode.none
space = "Lagrange"
k = 1

mesh = create_rectangle(
    MPI.COMM_WORLD, ((0.0, 0.0), (2.0, 1.0)), (2 * n, n),
    ghost_mode=ghost_mode)
edim = mesh.topology.dim
entities = locate_entities(mesh, edim, lambda x: x[0] <= 1.0)
submesh, entity_map, vertex_map, geom_map = create_submesh(
    mesh, edim, entities)

element = (space, k)
V_m = fem.FunctionSpace(mesh, element)
V_sm = fem.FunctionSpace(submesh, element)

u = ufl.TrialFunction(V_sm)
v = ufl.TestFunction(V_m)
dx_sm = ufl.Measure("dx", domain=submesh)
ds_sm = ufl.Measure("ds", domain=submesh)

entity_maps = {mesh: entity_map}
a = fem.form(ufl.inner(u, v) * (dx_sm + ds_sm),
             entity_maps=entity_maps)
A = fem.petsc.assemble_matrix(a)
A.assemble()

print(A[:, :])
