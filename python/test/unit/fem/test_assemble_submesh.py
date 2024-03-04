# Copyright (C) 2022 Joseph P. Dean
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
from dolfinx.mesh import (
    GhostMode,
    create_box,
    create_rectangle,
    create_submesh,
    create_unit_cube,
    create_unit_square,
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
    fem.set_bc(b.array, [bc])
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
    assert b_mesh_0.norm() == pytest.approx(b_submesh.norm(), rel=1.0e-4)
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
    assert b_submesh.norm() == pytest.approx(b_square_mesh.norm())
    assert np.isclose(s_submesh, s_square_mesh)


@pytest.mark.parametrize("n", [4, 6])
@pytest.mark.parametrize("k", [1, 3])
@pytest.mark.parametrize("space", ["Lagrange", "Discontinuous Lagrange"])
def test_mixed_dom_codim_0(n, k, space):
    """Test assembling a form where the trial and test functions
    are defined on different meshes"""

    def create_meshtags(msh, dim, tagged_entities):
        values = []
        for tag, entities in tagged_entities.items():
            values.append(np.full_like(entities, tag, dtype=np.intc))
        entities = np.hstack([entities for entities in tagged_entities.values()])
        perm = np.argsort(entities)
        values = np.hstack(values)
        return meshtags(msh, dim, entities[perm], values[perm])

    # Define some boundary markers
    def boundary_0_marker(x):
        return np.isclose(x[0], 0.0)

    def boundary_1_marker(x):
        return np.isclose(x[1], 0.0) & (x[0] <= 1.0)

    def interior_marker(x):
        dist = 1 / (2 * n)
        return (x[0] > dist) & (x[0] < 1 - dist) & (x[1] > dist) & (x[1] < 1 - dist)

    msh = create_rectangle(
        MPI.COMM_WORLD, ((0.0, 0.0), (2.0, 1.0)), (2 * n, n), ghost_mode=GhostMode.shared_facet
    )

    # Dictionary for tags
    markers = {"cells": 8, "bndry_0": 1, "interior": 4, "bndry_1": 7}

    # Locate cells in left half of mesh and create mesh tags
    tdim = msh.topology.dim
    ct = create_meshtags(
        msh, tdim, {markers["cells"]: locate_entities(msh, tdim, lambda x: x[0] <= 1.0)}
    )

    # Create mesh tags for facets
    fdim = tdim - 1
    ft = create_meshtags(
        msh,
        fdim,
        {
            markers["bndry_0"]: locate_entities_boundary(msh, fdim, boundary_0_marker),
            markers["bndry_1"]: locate_entities_boundary(msh, fdim, boundary_1_marker),
            markers["interior"]: locate_entities(msh, fdim, interior_marker),
        },
    )

    # Create integration measures on the mesh
    dx_msh = ufl.Measure("dx", domain=msh, subdomain_data=ct)
    ds_msh = ufl.Measure("ds", domain=msh, subdomain_data=ft)
    dS_msh = ufl.Measure("dS", domain=msh, subdomain_data=ft)

    # Create a submesh of the left half of the mesh
    smsh, smsh_to_msh = create_submesh(msh, tdim, ct.find(markers["cells"]))[:2]

    # Create some integration measures on the submesh
    ft_smsh = create_meshtags(
        smsh,
        fdim,
        {
            markers["bndry_0"]: locate_entities_boundary(smsh, fdim, boundary_0_marker),
            markers["bndry_1"]: locate_entities_boundary(smsh, fdim, boundary_1_marker),
            markers["interior"]: locate_entities(smsh, fdim, interior_marker),
        },
    )

    # Integration measures on the submesh
    ds_smsh = ufl.Measure("ds", domain=smsh, subdomain_data=ft_smsh)
    dS_smsh = ufl.Measure("dS", domain=smsh, subdomain_data=ft_smsh)

    # Define function spaces over the mesh and submesh
    V = fem.functionspace(msh, (space, k))
    W = fem.functionspace(msh, (space, k))
    Q = fem.functionspace(smsh, (space, k))

    # Trial and test functions on the mesh
    u = ufl.TrialFunction(V)
    w = ufl.TestFunction(W)

    # Test function on the submesh
    q = ufl.TestFunction(Q)

    # Create a Dirichlet boundary condition
    u_bc = fem.Function(V)
    u_bc.interpolate(lambda x: np.sin(np.pi * x[0]))
    dirichlet_dofs = fem.locate_dofs_topological(V, fdim, ft.find(markers["bndry_0"]))
    bc = fem.dirichletbc(u_bc, dirichlet_dofs)

    # Define UFL forms
    def ufl_form_a(u, v, dx, ds, dS):
        return ufl.inner(u, v) * dx + ufl.inner(u, v) * ds + ufl.inner(u("+"), v("-")) * dS

    def ufl_form_L(v, dx, ds, dS):
        return ufl.inner(2.5, v) * dx + ufl.inner(0.5, v) * ds + ufl.inner(0.1, v("-")) * dS

    # Single-domain assembly over msh as a reference
    a = fem.form(
        ufl_form_a(
            u, w, dx_msh(markers["cells"]), ds_msh(markers["bndry_0"]), dS_msh(markers["interior"])
        )
    )
    A = fem.assemble_matrix(a, bcs=[bc])
    A.scatter_reverse()

    L = fem.form(
        ufl_form_L(
            w, dx_msh(markers["cells"]), ds_msh(markers["bndry_0"]), dS_msh(markers["interior"])
        )
    )
    b = fem.assemble_vector(L)
    fem.apply_lifting(b.array, [a], bcs=[[bc]])
    b.scatter_reverse(la.InsertMode.add)

    # Assemble a mixed-domain form, taking smsh to be the integration domain
    # Entity maps must map cells in smsh (the integration domain mesh) to
    # cells in msh
    entity_maps = {msh._cpp_object: np.array(smsh_to_msh, dtype=np.int32)}
    a0 = fem.form(
        ufl_form_a(
            u,
            q,
            ufl.dx(smsh),
            ds_smsh(markers["bndry_0"]),
            dS_smsh(markers["interior"]),
        ),
        entity_maps=entity_maps,
    )
    A0 = fem.assemble_matrix(a0, bcs=[bc])
    A0.scatter_reverse()
    assert np.isclose(A0.squared_norm(), A.squared_norm())

    # Now assemble a mixed-domain form using msh as integration domain
    # Entity maps must map cells in msh (the integration domain mesh) to
    # cells in smsh
    cell_imap = msh.topology.index_map(tdim)
    num_cells = cell_imap.size_local + cell_imap.num_ghosts
    msh_to_smsh = np.full(num_cells, -1)
    msh_to_smsh[smsh_to_msh] = np.arange(len(smsh_to_msh))
    entity_maps = {smsh._cpp_object: np.array(msh_to_smsh, dtype=np.int32)}

    a1 = fem.form(
        ufl_form_a(
            u, q, dx_msh(markers["cells"]), ds_msh(markers["bndry_0"]), dS_msh(markers["interior"])
        ),
        entity_maps=entity_maps,
    )
    A1 = fem.assemble_matrix(a1, bcs=[bc])
    A1.scatter_reverse()
    assert np.isclose(A1.squared_norm(), A.squared_norm())

    L1 = fem.form(
        ufl_form_L(
            q, dx_msh(markers["cells"]), ds_msh(markers["bndry_0"]), dS_msh(markers["interior"])
        ),
        entity_maps=entity_maps,
    )
    b1 = fem.assemble_vector(L1)
    fem.apply_lifting(b1.array, [a1], bcs=[[bc]])
    b1.scatter_reverse(la.InsertMode.add)
    assert np.isclose(b1.norm(), b.norm())
