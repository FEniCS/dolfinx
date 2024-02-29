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


@pytest.mark.parametrize("n", [2, 6])
@pytest.mark.parametrize("k", [1, 4])
@pytest.mark.parametrize("space", ["Lagrange", "Discontinuous Lagrange"])
@pytest.mark.parametrize("ghost_mode", [GhostMode.none, GhostMode.shared_facet])
def test_mixed_dom_codim_0(n, k, space, ghost_mode):
    """Test assembling a form where the trial and test functions
    are defined on different meshes"""

    msh = create_rectangle(
        MPI.COMM_WORLD, ((0.0, 0.0), (2.0, 1.0)), (2 * n, n), ghost_mode=ghost_mode
    )

    # Locate cells in left half of mesh and create mesh tags
    tdim = msh.topology.dim
    cells = locate_entities(msh, tdim, lambda x: x[0] <= 1.0)
    perm = np.argsort(cells)
    tag = 1
    values = np.full_like(cells, tag, dtype=np.intc)
    ct = meshtags(msh, tdim, cells[perm], values[perm])

    # Locate facets on left boundary and create mesh tags
    def boundary_marker(x): return np.isclose(x[0], 0.0)

    fdim = tdim - 1
    facets = locate_entities(msh, fdim, boundary_marker)
    facet_perm = np.argsort(facets)
    facet_values = np.full_like(facets, tag, dtype=np.intc)
    ft = meshtags(msh, fdim, facets[facet_perm], facet_values[facet_perm])

    # Create integration measures on the mesh
    dx_msh = ufl.Measure("dx", domain=msh, subdomain_data=ct)
    ds_msh = ufl.Measure("ds", domain=msh, subdomain_data=ft)

    # Create a submesh of the left half of the mesh
    smsh, smsh_to_msh = create_submesh(msh, tdim, cells)[:2]

    # Define function spaces over the mesh and submesh
    V_msh = fem.functionspace(msh, (space, k))
    V_smsh = fem.functionspace(smsh, (space, k))

    # Trial and test functions on the mesh
    u, v = ufl.TrialFunction(V_msh), ufl.TestFunction(V_msh)

    # Test function on the submesh
    w = ufl.TestFunction(V_smsh)

    def ufl_form(u, v, dx, ds):
        return ufl.inner(u, v) * dx + ufl.inner(u, v) * ds

    # Define form to compare to and assemble
    a = fem.form(ufl_form(u, v, dx_msh(tag), ds_msh(tag)))
    A = fem.assemble_matrix(a)
    A.scatter_reverse()

    # Assemble a mixed-domain form, taking smsh to be the integration domain
    # Entity maps must map cells in smsh (the integration domain mesh) to
    # cells in msh
    facets_smsh = locate_entities(smsh, fdim, boundary_marker)
    facet_perm_smsh = np.argsort(facets_smsh)
    facet_values_smsh = np.full_like(facets_smsh, tag, dtype=np.intc)
    ft_smsh = meshtags(smsh, fdim, facets_smsh[facet_perm_smsh], facet_values_smsh[facet_perm_smsh])
    ds_smsh = ufl.Measure("ds", domain=smsh, subdomain_data=ft_smsh)

    entity_maps = {msh._cpp_object: np.array(smsh_to_msh, dtype=np.int32)}
    a0 = fem.form(ufl_form(u, w, ufl.dx(smsh), ds_smsh(tag)), entity_maps=entity_maps)
    A0 = fem.assemble_matrix(a0)
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

    a1 = fem.form(ufl_form(u, w, dx_msh(tag), ds_msh(tag)), entity_maps=entity_maps)
    A1 = fem.assemble_matrix(a1)
    A1.scatter_reverse()
    assert np.isclose(A1.squared_norm(), A.squared_norm())
