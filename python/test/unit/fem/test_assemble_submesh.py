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
        cells = locate_entities(msh, msh.topology.dim, lambda x: x[0] <= 0.5)
        mt = create_meshtags(msh, tdim, cells)
    elif integral_type == "ds":
        facets = locate_entities_boundary(
            msh, msh.topology.dim - 1, lambda x: np.isclose(x[1], 0.0) & (x[0] <= 0.5)
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
    smsh, smsh_to_msh = create_submesh(msh, tdim, cells)[:2]

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
    u_bc.interpolate(lambda x: np.sin(np.pi * x[0]))
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

    # Assemble a mixed-domain form using msh as integration domain.
    # Entity maps must map cells in msh (the integration domain mesh,
    # defined by the integration measure) to cells in smsh.
    cell_imap = msh.topology.index_map(tdim)
    num_cells = cell_imap.size_local + cell_imap.num_ghosts
    msh_to_smsh = np.full(num_cells, -1)
    msh_to_smsh[smsh_to_msh] = np.arange(len(smsh_to_msh))
    entity_maps = {smsh._cpp_object: np.array(msh_to_smsh, dtype=np.int32)}
    a1 = fem.form(a_ufl(u, q, f, g, measure_msh), entity_maps=entity_maps)
    A1 = fem.assemble_matrix(a1, bcs=[bc])
    A1.scatter_reverse()
    assert np.isclose(A1.squared_norm(), A.squared_norm())

    L1 = fem.form(L_ufl(q, f, g, measure_msh), entity_maps=entity_maps)
    b1 = fem.assemble_vector(L1)
    fem.apply_lifting(b1.array, [a1], bcs=[[bc]])
    b1.scatter_reverse(la.InsertMode.add)
    assert np.isclose(b1.norm(), b.norm())

    # Now assemble a mixed-domain form taking smsh to be the integration
    # domain.

    # Create the measure (this time defined over the submesh)
    measure_smsh = create_measure(smsh, integral_type)

    # Entity maps must map cells in smsh (the integration domain mesh) to
    # cells in msh
    entity_maps = {msh._cpp_object: np.array(smsh_to_msh, dtype=np.int32)}
    a0 = fem.form(a_ufl(u, q, f, g, measure_smsh), entity_maps=entity_maps)
    A0 = fem.assemble_matrix(a0, bcs=[bc])
    A0.scatter_reverse()
    assert np.isclose(A0.squared_norm(), A.squared_norm())

    L0 = fem.form(L_ufl(q, f, g, measure_smsh), entity_maps=entity_maps)
    b0 = fem.assemble_vector(L0)
    fem.apply_lifting(b0.array, [a0], bcs=[[bc]])
    b0.scatter_reverse(la.InsertMode.add)
    assert np.isclose(b0.norm(), b.norm())

    # TODO Rename
    # TODO Scalar
