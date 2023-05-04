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
from dolfinx.la import ScatterMode
from dolfinx.fem import assemble_vector, assemble_matrix, apply_lifting, set_bc
from dolfinx.mesh import (GhostMode, create_box, create_rectangle,
                          create_submesh, create_unit_cube, create_unit_square,
                          locate_entities, locate_entities_boundary)

from mpi4py import MPI
from dolfinx import default_scalar_type

def assemble(mesh, space, k):
    V = fem.FunctionSpace(mesh, (space, k))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    dx = ufl.Measure("dx", domain=mesh)
    ds = ufl.Measure("ds", domain=mesh)

    c = fem.Constant(mesh, default_scalar_type(0.75))
    a = fem.form(ufl.inner(c * u, v) * (dx + ds))

    facet_dim = mesh.topology.dim - 1
    facets = locate_entities_boundary(mesh, facet_dim, lambda x: np.isclose(x[0], 1))
    dofs = fem.locate_dofs_topological(V, facet_dim, facets)

    bc_func = fem.Function(V)
    # TODO Interpolate when issue #2126 has been resolved
    bc_func.x.array[:] = 1.0

    bc = fem.dirichletbc(bc_func, dofs)

    A = assemble_matrix(a, bcs=[bc])
    A.finalize()

    # TODO Test assembly with fem.Function
    x = ufl.SpatialCoordinate(mesh)
    f = 1.5 + x[0]
    L = fem.form(ufl.inner(c * f, v) * (dx + ds))
    b = assemble_vector(L)
    apply_lifting(b.array, [a], bcs=[[bc]])
    b.scatter_reverse(ScatterMode.add)
    set_bc(b.array, [bc])
    s = mesh.comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(c * f, f) * (dx + ds))), op=MPI.SUM)
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
        mesh_0 = create_unit_square(MPI.COMM_WORLD, n, n, ghost_mode=ghost_mode)
        mesh_1 = create_rectangle(MPI.COMM_WORLD, ((0.0, 0.0), (2.0, 1.0)), (2 * n, n), ghost_mode=ghost_mode)
    else:
        mesh_0 = create_unit_cube(MPI.COMM_WORLD, n, n, n, ghost_mode=ghost_mode)
        mesh_1 = create_box(MPI.COMM_WORLD, ((0.0, 0.0, 0.0), (2.0, 1.0, 1.0)),
                            (2 * n, n, n), ghost_mode=ghost_mode)

    A_mesh_0, b_mesh_0, s_mesh_0 = assemble(mesh_0, space, k)

    edim = mesh_1.topology.dim
    entities = locate_entities(mesh_1, edim, lambda x: x[0] <= 1.0)
    submesh = create_submesh(mesh_1, edim, entities)[0]
    A_submesh, b_submesh, s_submesh = assemble(submesh, space, k)

    A0_norm, A_norm, b0_norm, b_norm, s0, s1 = \
        MPI.COMM_WORLD.allreduce(np.array([A_mesh_0.norm_squared(),
                                           A_submesh.norm_squared(),
                                           b_mesh_0.squared_norm(),
                                           b_submesh.squared_norm(),
                                           s_mesh_0, s_submesh]), op=MPI.SUM)
    assert np.isclose(A0_norm, A_norm)
    assert np.isclose(b0_norm, b_norm)
    assert np.isclose(s0, s1)


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

    A0_norm, A_norm, b0_norm, b_norm, s0, s1 = \
        MPI.COMM_WORLD.allreduce(np.array([A_square_mesh.norm_squared(),
                                           A_submesh.norm_squared(),
                                           b_square_mesh.squared_norm(),
                                           b_submesh.squared_norm(),
                                           s_submesh, s_square_mesh]), op=MPI.SUM)
    assert np.isclose(A0_norm, A_norm)
    assert np.isclose(b0_norm, b_norm)
    assert np.isclose(s0, s1)
