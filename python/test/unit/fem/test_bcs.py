# Copyright (C) 2020-2021 Joseph P. Dean, Massimiliano Leoni and Jørgen S. Dokken
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from mpi4py import MPI

import numpy as np
import pytest

import ufl
from basix.ufl import element, mixed_element
from dolfinx import default_real_type, default_scalar_type, la
from dolfinx.fem import (
    Constant,
    Function,
    apply_lifting,
    assemble_matrix,
    assemble_vector,
    create_matrix,
    create_vector,
    dirichletbc,
    form,
    functionspace,
    locate_dofs_geometrical,
    locate_dofs_topological,
    set_bc,
)
from dolfinx.mesh import CellType, create_unit_cube, create_unit_square, exterior_facet_indices
from ufl import dx, inner


def test_locate_dofs_geometrical():
    """Test that locate_dofs_geometrical, when passed two function
    spaces, returns the correct degrees of freedom in each space"""
    mesh = create_unit_square(MPI.COMM_WORLD, 4, 8)
    p0, p1 = 1, 2
    P0 = element("Lagrange", mesh.basix_cell(), p0, dtype=default_real_type)
    P1 = element("Lagrange", mesh.basix_cell(), p1, dtype=default_real_type)

    W = functionspace(mesh, mixed_element([P0, P1]))
    V = W.sub(0).collapse()[0]

    with pytest.raises(RuntimeError):
        locate_dofs_geometrical(W, lambda x: np.isclose(x.T, [0, 0, 0]).all(axis=1))

    dofs = locate_dofs_geometrical((W.sub(0), V), lambda x: np.isclose(x.T, [0, 0, 0]).all(axis=1))

    # Collect dofs (global indices) from all processes
    dofs0_global = W.sub(0).dofmap.index_map.local_to_global(dofs[0])
    dofs1_global = V.dofmap.index_map.local_to_global(dofs[1])
    all_dofs0 = set(np.concatenate(MPI.COMM_WORLD.allgather(dofs0_global)))
    all_dofs1 = set(np.concatenate(MPI.COMM_WORLD.allgather(dofs1_global)))

    # Check only one dof pair is found globally
    assert len(all_dofs0) == 1
    assert len(all_dofs1) == 1

    # On process with the dof pair
    if len(dofs) == 1:
        # Check correct dof returned in W
        coords_W = W.tabulate_dof_coordinates()
        assert np.isclose(coords_W[dofs[0][0]], [0, 0, 0]).all()
        # Check correct dof returned in V
        coords_V = V.tabulate_dof_coordinates()
        assert np.isclose(coords_V[dofs[0][1]], [0, 0, 0]).all()


def test_overlapping_bcs():
    """Test that, when boundaries condition overlap, the last provided
    boundary condition is applied"""
    n = 23
    mesh = create_unit_square(MPI.COMM_WORLD, n, n)
    V = functionspace(mesh, ("Lagrange", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = form(inner(u, v) * dx)
    L = form(inner(1, v) * dx)

    dofs_left = locate_dofs_geometrical(V, lambda x: x[0] < 1.0 / (2.0 * n))
    dofs_top = locate_dofs_geometrical(V, lambda x: x[1] > 1.0 - 1.0 / (2.0 * n))
    dof_corner = np.array(list(set(dofs_left).intersection(set(dofs_top))), dtype=np.int64)

    # Check only one dof pair is found globally
    assert len(set(np.concatenate(MPI.COMM_WORLD.allgather(dof_corner)))) == 1

    bcs = [
        dirichletbc(default_scalar_type(0), dofs_left, V),
        dirichletbc(default_scalar_type(123.456), dofs_top, V),
    ]

    A, b = create_matrix(a), create_vector(L)
    assemble_matrix(A, a, bcs=bcs)
    A.scatter_reverse()

    # Check the diagonal (only on the rank that owns the row)
    As = A.to_scipy(ghosted=True)
    d = As.diagonal()
    if len(dof_corner) > 0 and dof_corner[0] < V.dofmap.index_map.size_local:
        assert d[dof_corner[0]] == 1.0  # /NOSONAR

    b.array[:] = 0
    assemble_vector(b.array, L)
    apply_lifting(b.array, [a], [bcs])
    b.scatter_reverse(la.InsertMode.add)
    set_bc(b.array, bcs)
    b.scatter_forward()

    if len(dof_corner) > 0:
        assert b.array[dof_corner[0]] == default_real_type(123.456)


def test_constant_bc_constructions():
    """Test construction from constant values"""
    msh = create_unit_square(MPI.COMM_WORLD, 4, 4, dtype=default_real_type)
    gdim = msh.geometry.dim
    V0 = functionspace(msh, ("Lagrange", 1))
    V1 = functionspace(msh, ("Lagrange", 1, (gdim,)))
    V2 = functionspace(msh, ("Lagrange", 1, (gdim, gdim)))

    tdim = msh.topology.dim
    msh.topology.create_connectivity(1, 2)
    boundary_facets = exterior_facet_indices(msh.topology)
    boundary_dofs0 = locate_dofs_topological(V0, tdim - 1, boundary_facets)
    boundary_dofs1 = locate_dofs_topological(V1, tdim - 1, boundary_facets)
    boundary_dofs2 = locate_dofs_topological(V2, tdim - 1, boundary_facets)

    if default_real_type == np.float64:
        dtype = np.complex128
    else:
        dtype = np.complex64

    bc0 = dirichletbc(dtype(1.0 + 2.2j), boundary_dofs0, V0)
    assert bc0.g.value.dtype == dtype
    assert bc0.g.value.shape == tuple()
    assert bc0.g.value == dtype(1.0 + 2.2j)

    bc1 = dirichletbc(np.array([1.0 + 2.2j, 3.0 + 2.2j], dtype=dtype), boundary_dofs1, V1)
    assert bc1.g.value.dtype == dtype
    assert bc1.g.value.shape == (tdim,)
    assert (bc1.g.value == [dtype(1.0 + 2.2j), dtype(3.0 + 2.2j)]).all()

    bc2 = dirichletbc(
        np.array([[1.0, 3.0], [3.0, -2.0]], dtype=default_real_type), boundary_dofs2, V2
    )
    assert bc2.g.value.dtype == default_real_type
    assert bc2.g.value.shape == (tdim, tdim)
    assert (bc2.g.value == [[1.0, 3.0], [3.0, -2.0]]).all()


@pytest.mark.parametrize(
    "mesh_factory",
    [
        (create_unit_square, (MPI.COMM_WORLD, 4, 4)),
        (create_unit_square, (MPI.COMM_WORLD, 8, 8, CellType.quadrilateral)),
        (create_unit_cube, (MPI.COMM_WORLD, 3, 3, 3)),
        (create_unit_cube, (MPI.COMM_WORLD, 3, 3, 3, CellType.hexahedron)),
    ],
)
def test_constant_bc(mesh_factory):
    """Test that setting a dirichletbc with a constant yields the same
    result as setting it with a function"""
    func, args = mesh_factory
    mesh = func(*args)
    V = functionspace(mesh, ("Lagrange", 1))
    c = default_scalar_type(2)
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim - 1, tdim)
    boundary_facets = exterior_facet_indices(mesh.topology)
    boundary_dofs = locate_dofs_topological(V, tdim - 1, boundary_facets)

    u_bc = Function(V)
    u_bc.x.array[:] = c

    bc_f = dirichletbc(u_bc, boundary_dofs)
    bc_c = dirichletbc(c, boundary_dofs, V)

    u_f = Function(V)
    set_bc(u_f.x.array, [bc_f])

    u_c = Function(V)
    set_bc(u_c.x.array, [bc_c])
    assert np.allclose(u_f.x.array, u_c.x.array)


@pytest.mark.parametrize(
    "mesh_factory",
    [
        (create_unit_square, (MPI.COMM_WORLD, 4, 4)),
        (create_unit_square, (MPI.COMM_WORLD, 8, 8, CellType.quadrilateral)),
        (create_unit_cube, (MPI.COMM_WORLD, 3, 3, 3)),
        (create_unit_cube, (MPI.COMM_WORLD, 3, 3, 3, CellType.hexahedron)),
    ],
)
def test_vector_constant_bc(mesh_factory):
    """Test that setting a dirichletbc with a vector valued constant
    yields the same result as setting it with a function"""
    func, args = mesh_factory
    mesh = func(*args)
    tdim = mesh.topology.dim
    gdim = mesh.geometry.dim
    V = functionspace(mesh, ("Lagrange", 1, (gdim,)))
    assert V.num_sub_spaces == gdim
    c = np.arange(1, mesh.geometry.dim + 1, dtype=default_scalar_type)
    mesh.topology.create_connectivity(tdim - 1, tdim)
    boundary_facets = exterior_facet_indices(mesh.topology)

    # Set using sub-functions
    Vs = [V.sub(i).collapse()[0] for i in range(V.num_sub_spaces)]
    boundary_dofs = [
        locate_dofs_topological((V.sub(i), Vs[i]), tdim - 1, boundary_facets)
        for i in range(len(Vs))
    ]
    u_bcs = [Function(Vs[i]) for i in range(len(Vs))]
    bcs_f = []
    for i, u in enumerate(u_bcs):
        u_bcs[i].x.array[:] = c[i]
        bcs_f.append(dirichletbc(u_bcs[i], boundary_dofs[i], V.sub(i)))
    u_f = Function(V)
    set_bc(u_f.x.array, bcs_f)

    # Set using constant
    boundary_dofs = locate_dofs_topological(V, tdim - 1, boundary_facets)
    bc_c = dirichletbc(c, boundary_dofs, V)
    u_c = Function(V)
    u_c.x.array[:] = 0.0
    set_bc(u_c.x.array, [bc_c])

    assert np.allclose(u_f.x.array, u_c.x.array)


@pytest.mark.parametrize(
    "mesh_factory",
    [
        (create_unit_square, (MPI.COMM_WORLD, 4, 4)),
        (create_unit_square, (MPI.COMM_WORLD, 8, 8, CellType.quadrilateral)),
        (create_unit_cube, (MPI.COMM_WORLD, 3, 3, 3)),
        (create_unit_cube, (MPI.COMM_WORLD, 3, 3, 3, CellType.hexahedron)),
    ],
)
def test_sub_constant_bc(mesh_factory):
    """Test that setting a dirichletbc with on a component of a vector
    valued function yields the same result as setting it with a
    function"""
    func, args = mesh_factory
    mesh = func(*args)
    gdim = mesh.geometry.dim
    V = functionspace(mesh, ("Lagrange", 1, (gdim,)))
    c = Constant(mesh, default_scalar_type(3.14))
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim - 1, tdim)
    boundary_facets = exterior_facet_indices(mesh.topology)

    for i in range(V.num_sub_spaces):
        Vi = V.sub(i).collapse()[0]
        u_bci = Function(Vi)
        u_bci.x.array[:] = default_scalar_type(c.value)

        boundary_dofsi = locate_dofs_topological((V.sub(i), Vi), tdim - 1, boundary_facets)
        bc_fi = dirichletbc(u_bci, boundary_dofsi, V.sub(i))
        boundary_dofs = locate_dofs_topological(V.sub(i), tdim - 1, boundary_facets)
        bc_c = dirichletbc(c, boundary_dofs, V.sub(i))

        u_f = Function(V)
        set_bc(u_f.x.array, [bc_fi])
        u_c = Function(V)
        set_bc(u_c.x.array, [bc_c])
        assert np.allclose(u_f.x.array, u_c.x.array)


@pytest.mark.parametrize(
    "mesh_factory",
    [
        (create_unit_square, (MPI.COMM_WORLD, 4, 4)),
        (create_unit_square, (MPI.COMM_WORLD, 8, 8, CellType.quadrilateral)),
        (create_unit_cube, (MPI.COMM_WORLD, 3, 3, 3)),
        (create_unit_cube, (MPI.COMM_WORLD, 3, 3, 3, CellType.hexahedron)),
    ],
)
def test_mixed_constant_bc(mesh_factory):
    """Test that setting a dirichletbc with on a component of a mixed
    function yields the same result as setting it with a function"""
    func, args = mesh_factory
    mesh = func(*args)
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim - 1, tdim)
    boundary_facets = exterior_facet_indices(mesh.topology)
    TH = mixed_element(
        [
            element("Lagrange", mesh.basix_cell(), 1, dtype=default_real_type),
            element("Lagrange", mesh.basix_cell(), 2, dtype=default_real_type),
        ]
    )
    W = functionspace(mesh, TH)
    u = Function(W)

    bc_val = default_scalar_type(3)
    c = Constant(mesh, bc_val)
    u_func = Function(W)
    for i in range(2):
        u_func.x.array[:] = 0
        u.x.array[:] = 0

        # Apply BC to scalar component of a mixed space using a Constant
        dofs = locate_dofs_topological(W.sub(i), tdim - 1, boundary_facets)
        bc = dirichletbc(c, dofs, W.sub(i))
        set_bc(u.x.array, [bc])

        # Apply BC to scalar component of a mixed space using a Function
        ubc = u.sub(i).collapse()
        ubc.interpolate(lambda x: np.full(x.shape[1], bc_val))
        dofs_both = locate_dofs_topological(
            (W.sub(i), ubc.function_space), tdim - 1, boundary_facets
        )
        bc_func = dirichletbc(ubc, dofs_both, W.sub(i))
        set_bc(u_func.x.array, [bc_func])

        # Check that both approaches yield the same vector
        assert np.allclose(u.x.array, u_func.x.array)


def test_mixed_blocked_constant():
    """Check that mixed space with blocked component cannot have
    Dirichlet BC based on a vector valued Constant."""
    mesh = create_unit_square(MPI.COMM_WORLD, 4, 4)
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim - 1, tdim)
    boundary_facets = exterior_facet_indices(mesh.topology)

    TH = mixed_element(
        [
            element("Lagrange", mesh.basix_cell(), 1, dtype=default_real_type),
            element(
                "Lagrange",
                mesh.basix_cell(),
                2,
                shape=(mesh.geometry.dim,),
                dtype=default_real_type,
            ),
        ]
    )
    W = functionspace(mesh, TH)
    u = Function(W)
    c0 = default_scalar_type(3)
    dofs0 = locate_dofs_topological(W.sub(0), tdim - 1, boundary_facets)
    bc0 = dirichletbc(c0, dofs0, W.sub(0))
    set_bc(u.x.array, [bc0])

    # Apply BC to scalar component of a mixed space using a Function
    ubc = u.sub(0).collapse()
    ubc.interpolate(lambda x: np.full(x.shape[1], c0))
    dofs_both = locate_dofs_topological((W.sub(0), ubc.function_space), tdim - 1, boundary_facets)
    bc_func = dirichletbc(ubc, dofs_both, W.sub(0))
    u_func = Function(W)
    set_bc(u_func.x.array, [bc_func])
    assert np.allclose(u.x.array, u_func.x.array)

    # Check that vector space throws error
    c1 = default_scalar_type((5, 7))
    with pytest.raises(RuntimeError):
        dofs1 = locate_dofs_topological(W.sub(1), tdim - 1, boundary_facets)
        dirichletbc(c1, dofs1, W.sub(1))
