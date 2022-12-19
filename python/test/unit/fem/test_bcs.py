# Copyright (C) 2020-2021 Joseph P. Dean, Massimiliano Leoni and Jørgen S. Dokken
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import numpy as np
import pytest

import ufl
from dolfinx.fem import (Constant, Function, FunctionSpace,
                         TensorFunctionSpace, VectorFunctionSpace, dirichletbc,
                         form, locate_dofs_geometrical,
                         locate_dofs_topological)
from dolfinx.fem.petsc import (apply_lifting, assemble_matrix, assemble_vector,
                               create_matrix, create_vector, set_bc)
from dolfinx.mesh import (CellType, create_unit_cube, create_unit_square,
                          locate_entities_boundary)
from ufl import dx, inner

from mpi4py import MPI
from petsc4py import PETSc


def test_locate_dofs_geometrical():
    """Test that locate_dofs_geometrical, when passed two function
    spaces, returns the correct degrees of freedom in each space"""
    mesh = create_unit_square(MPI.COMM_WORLD, 4, 8)
    p0, p1 = 1, 2
    P0 = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), p0)
    P1 = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), p1)

    W = FunctionSpace(mesh, P0 * P1)
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
    V = FunctionSpace(mesh, ("Lagrange", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = form(inner(u, v) * dx)
    L = form(inner(1, v) * dx)

    dofs_left = locate_dofs_geometrical(V, lambda x: x[0] < 1.0 / (2.0 * n))
    dofs_top = locate_dofs_geometrical(V, lambda x: x[1] > 1.0 - 1.0 / (2.0 * n))
    dof_corner = np.array(list(set(dofs_left).intersection(set(dofs_top))), dtype=np.int64)

    # Check only one dof pair is found globally
    assert len(set(np.concatenate(MPI.COMM_WORLD.allgather(dof_corner)))) == 1

    bcs = [dirichletbc(PETSc.ScalarType(0), dofs_left, V),
           dirichletbc(PETSc.ScalarType(123.456), dofs_top, V)]

    A, b = create_matrix(a), create_vector(L)
    assemble_matrix(A, a, bcs=bcs)
    A.assemble()

    # Check the diagonal (only on the rank that owns the row)
    d = A.getDiagonal()
    if len(dof_corner) > 0 and dof_corner[0] < V.dofmap.index_map.size_local:
        assert np.isclose(d.array_r[dof_corner[0]], 1.0)

    with b.localForm() as b_loc:
        b_loc.set(0)
    assemble_vector(b, L)
    apply_lifting(b, [a], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, bcs)
    b.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    if len(dof_corner) > 0:
        with b.localForm() as b_loc:
            assert b_loc[dof_corner[0]] == 123.456


def test_constant_bc_constructions():
    """Test construction from constant values"""
    msh = create_unit_square(MPI.COMM_WORLD, 4, 4)
    V0 = FunctionSpace(msh, ("Lagrange", 1))
    V1 = VectorFunctionSpace(msh, ("Lagrange", 1))
    V2 = TensorFunctionSpace(msh, ("Lagrange", 1))

    tdim = msh.topology.dim
    boundary_facets = locate_entities_boundary(msh, tdim - 1, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs0 = locate_dofs_topological(V0, tdim - 1, boundary_facets)
    boundary_dofs1 = locate_dofs_topological(V1, tdim - 1, boundary_facets)
    boundary_dofs2 = locate_dofs_topological(V2, tdim - 1, boundary_facets)

    bc0 = dirichletbc(1.0 + 2.2j, boundary_dofs0, V0)
    assert bc0.value.value.dtype == np.complex128
    assert bc0.value.value.shape == tuple()
    assert bc0.value.value == 1.0 + 2.2j

    bc1 = dirichletbc(np.array([1.0 + 2.2j, 3.0 + 2.2j], dtype=np.complex128), boundary_dofs1, V1)
    assert bc1.value.value.dtype == np.complex128
    assert bc1.value.value.shape == (tdim,)
    assert (bc1.value.value == [1.0 + 2.2j, 3.0 + 2.2j]).all()

    bc2 = dirichletbc(np.array([[1.0, 3.0], [3.0, -2.0]], dtype=np.float32), boundary_dofs2, V2)
    assert bc2.value.value.dtype == np.float32
    assert bc2.value.value.shape == (tdim, tdim)
    assert (bc2.value.value == [[1.0, 3.0], [3.0, -2.0]]).all()


@pytest.mark.parametrize('mesh_factory',
                         [
                             (create_unit_square, (MPI.COMM_WORLD, 4, 4)),
                             (create_unit_square, (MPI.COMM_WORLD, 8, 8, CellType.quadrilateral)),
                             (create_unit_cube, (MPI.COMM_WORLD, 3, 3, 3)),
                             (create_unit_cube, (MPI.COMM_WORLD, 3, 3, 3, CellType.hexahedron))
                         ])
def test_constant_bc(mesh_factory):
    """Test that setting a dirichletbc with a constant yields the same
    result as setting it with a function"""
    func, args = mesh_factory
    mesh = func(*args)
    V = FunctionSpace(mesh, ("Lagrange", 1))
    c = PETSc.ScalarType(2)
    tdim = mesh.topology.dim
    boundary_facets = locate_entities_boundary(mesh, tdim - 1, lambda x: np.ones(x.shape[1], dtype=bool))

    boundary_dofs = locate_dofs_topological(V, tdim - 1, boundary_facets)

    u_bc = Function(V)
    u_bc.x.array[:] = c

    bc_f = dirichletbc(u_bc, boundary_dofs)
    bc_c = dirichletbc(c, boundary_dofs, V)

    u_f = Function(V)
    set_bc(u_f.vector, [bc_f])

    u_c = Function(V)
    set_bc(u_c.vector, [bc_c])
    assert np.allclose(u_f.vector.array, u_c.vector.array)


@pytest.mark.parametrize(
    'mesh_factory', [
        (create_unit_square, (MPI.COMM_WORLD, 4, 4)),
        (create_unit_square, (MPI.COMM_WORLD, 8, 8, CellType.quadrilateral)),
        (create_unit_cube, (MPI.COMM_WORLD, 3, 3, 3)),
        (create_unit_cube, (MPI.COMM_WORLD, 3, 3, 3, CellType.hexahedron))
    ])
def test_vector_constant_bc(mesh_factory):
    """Test that setting a dirichletbc with a vector valued constant
    yields the same result as setting it with a function"""
    func, args = mesh_factory
    mesh = func(*args)
    tdim = mesh.topology.dim
    V = VectorFunctionSpace(mesh, ("Lagrange", 1))
    assert V.num_sub_spaces == mesh.geometry.dim
    c = np.arange(1, mesh.geometry.dim + 1, dtype=PETSc.ScalarType)
    boundary_facets = locate_entities_boundary(mesh, tdim - 1, lambda x: np.ones(x.shape[1], dtype=bool))

    # Set using sub-functions
    Vs = [V.sub(i).collapse()[0] for i in range(V.num_sub_spaces)]
    boundary_dofs = [locate_dofs_topological((V.sub(i), Vs[i]), tdim - 1, boundary_facets)
                     for i in range(len(Vs))]
    u_bcs = [Function(Vs[i]) for i in range(len(Vs))]
    bcs_f = []
    for i, u in enumerate(u_bcs):
        u_bcs[i].x.array[:] = c[i]
        bcs_f.append(dirichletbc(u_bcs[i], boundary_dofs[i], V.sub(i)))
    u_f = Function(V)
    set_bc(u_f.vector, bcs_f)

    # Set using constant
    boundary_dofs = locate_dofs_topological(V, tdim - 1, boundary_facets)
    bc_c = dirichletbc(c, boundary_dofs, V)
    u_c = Function(V)
    u_c.x.array[:] = 0.0
    set_bc(u_c.vector, [bc_c])

    assert np.allclose(u_f.x.array, u_c.x.array)


@pytest.mark.parametrize(
    'mesh_factory', [
        (create_unit_square, (MPI.COMM_WORLD, 4, 4)),
        (create_unit_square, (MPI.COMM_WORLD, 8, 8, CellType.quadrilateral)),
        (create_unit_cube, (MPI.COMM_WORLD, 3, 3, 3)),
        (create_unit_cube, (MPI.COMM_WORLD, 3, 3, 3, CellType.hexahedron))
    ])
def test_sub_constant_bc(mesh_factory):
    """Test that setting a dirichletbc with on a component of a vector
    valued function yields the same result as setting it with a
    function"""
    func, args = mesh_factory
    mesh = func(*args)
    tdim = mesh.topology.dim
    V = VectorFunctionSpace(mesh, ("Lagrange", 1))
    c = Constant(mesh, PETSc.ScalarType(3.14))
    boundary_facets = locate_entities_boundary(mesh, tdim - 1, lambda x: np.ones(x.shape[1], dtype=bool))

    for i in range(V.num_sub_spaces):
        Vi = V.sub(i).collapse()[0]
        u_bci = Function(Vi)
        u_bci.x.array[:] = PETSc.ScalarType(c.value)

        boundary_dofsi = locate_dofs_topological((V.sub(i), Vi), tdim - 1, boundary_facets)
        bc_fi = dirichletbc(u_bci, boundary_dofsi, V.sub(i))
        boundary_dofs = locate_dofs_topological(V.sub(i), tdim - 1, boundary_facets)
        bc_c = dirichletbc(c, boundary_dofs, V.sub(i))

        u_f = Function(V)
        set_bc(u_f.vector, [bc_fi])
        u_c = Function(V)
        set_bc(u_c.vector, [bc_c])
        assert np.allclose(u_f.vector.array, u_c.vector.array)


@pytest.mark.parametrize(
    'mesh_factory', [
        (create_unit_square, (MPI.COMM_WORLD, 4, 4)),
        (create_unit_square, (MPI.COMM_WORLD, 8, 8, CellType.quadrilateral)),
        (create_unit_cube, (MPI.COMM_WORLD, 3, 3, 3)),
        (create_unit_cube, (MPI.COMM_WORLD, 3, 3, 3, CellType.hexahedron))
    ])
def test_mixed_constant_bc(mesh_factory):
    """Test that setting a dirichletbc with on a component of a mixed
    function yields the same result as setting it with a function"""
    func, args = mesh_factory
    mesh = func(*args)
    tdim = mesh.topology.dim
    boundary_facets = locate_entities_boundary(mesh, tdim - 1, lambda x: np.ones(x.shape[1], dtype=bool))
    TH = ufl.MixedElement([ufl.FiniteElement("Lagrange", mesh.ufl_cell(), 1),
                          ufl.FiniteElement("Lagrange", mesh.ufl_cell(), 2)])
    W = FunctionSpace(mesh, TH)
    u = Function(W)

    bc_val = PETSc.ScalarType(3)
    c = Constant(mesh, bc_val)
    u_func = Function(W)
    for i in range(2):
        u_func.x.array[:] = 0
        u.x.array[:] = 0

        # Apply BC to scalar component of a mixed space using a Constant
        dofs = locate_dofs_topological(W.sub(i), tdim - 1, boundary_facets)
        bc = dirichletbc(c, dofs, W.sub(i))
        set_bc(u.vector, [bc])

        # Apply BC to scalar component of a mixed space using a Function
        ubc = u.sub(i).collapse()
        ubc.interpolate(lambda x: np.full(x.shape[1], bc_val))
        dofs_both = locate_dofs_topological((W.sub(i), ubc.function_space), tdim - 1, boundary_facets)
        bc_func = dirichletbc(ubc, dofs_both, W.sub(i))
        set_bc(u_func.vector, [bc_func])

        # Check that both approaches yield the same vector
        assert np.allclose(u.x.array, u_func.x.array)


def test_mixed_blocked_constant():
    """Check that mixed space with blocked component cannot have
    Dirichlet BC based on a vector valued Constant."""
    mesh = create_unit_square(MPI.COMM_WORLD, 4, 4)

    tdim = mesh.topology.dim
    boundary_facets = locate_entities_boundary(mesh, tdim - 1, lambda x: np.ones(x.shape[1], dtype=bool))

    TH = ufl.MixedElement([ufl.FiniteElement("Lagrange", mesh.ufl_cell(), 1),
                           ufl.VectorElement("Lagrange", mesh.ufl_cell(), 2)])
    W = FunctionSpace(mesh, TH)
    u = Function(W)
    c0 = PETSc.ScalarType(3)
    dofs0 = locate_dofs_topological(W.sub(0), tdim - 1, boundary_facets)
    bc0 = dirichletbc(c0, dofs0, W.sub(0))
    set_bc(u.vector, [bc0])

    # Apply BC to scalar component of a mixed space using a Function
    ubc = u.sub(0).collapse()
    ubc.interpolate(lambda x: np.full(x.shape[1], c0))
    dofs_both = locate_dofs_topological((W.sub(0), ubc.function_space), tdim - 1, boundary_facets)
    bc_func = dirichletbc(ubc, dofs_both, W.sub(0))
    u_func = Function(W)
    set_bc(u_func.vector, [bc_func])
    assert np.allclose(u.x.array, u_func.x.array)

    # Check that vector space throws error
    c1 = PETSc.ScalarType((5, 7))
    with pytest.raises(RuntimeError):
        dofs1 = locate_dofs_topological(W.sub(1), tdim - 1, boundary_facets)
        dirichletbc(c1, dofs1, W.sub(1))
