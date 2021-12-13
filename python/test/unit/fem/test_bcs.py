# Copyright (C) 2020-2021 Joseph P. Dean, Massimiliano Leoni and Jørgen S. Dokken
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import numpy as np
import pytest

import ufl
from dolfinx.fem import (Constant, DirichletBC, Function, FunctionSpace,
                         VectorFunctionSpace, apply_lifting, assemble_matrix,
                         assemble_vector, create_matrix, create_vector,
                         locate_dofs_geometrical, locate_dofs_topological,
                         set_bc)
from dolfinx.generation import UnitCubeMesh, UnitSquareMesh
from dolfinx.mesh import CellType, locate_entities_boundary
from ufl import dx, inner

from mpi4py import MPI
from petsc4py import PETSc


def test_locate_dofs_geometrical():
    """Test that locate_dofs_geometrical, when passed two function
    spaces, returns the correct degrees of freedom in each space.
    """
    mesh = UnitSquareMesh(MPI.COMM_WORLD, 4, 8)
    p0, p1 = 1, 2
    P0 = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), p0)
    P1 = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), p1)

    W = FunctionSpace(mesh, P0 * P1)
    V = W.sub(0).collapse()

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
    boundary condition is applied.
    """
    n = 23
    mesh = UnitSquareMesh(MPI.COMM_WORLD, n, n)
    V = FunctionSpace(mesh, ("Lagrange", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = inner(u, v) * dx
    L = inner(1, v) * dx

    dofs_left = locate_dofs_geometrical(V, lambda x: x[0] < 1.0 / (2.0 * n))
    dofs_top = locate_dofs_geometrical(V, lambda x: x[1] > 1.0 - 1.0 / (2.0 * n))
    dof_corner = np.array(list(set(dofs_left).intersection(set(dofs_top))), dtype=np.int64)

    # Check only one dof pair is found globally
    assert len(set(np.concatenate(MPI.COMM_WORLD.allgather(dof_corner)))) == 1

    u0, u1 = Function(V), Function(V)
    with u0.vector.localForm() as u0_loc:
        u0_loc.set(0)
    with u1.vector.localForm() as u1_loc:
        u1_loc.set(123.456)
    bcs = [DirichletBC(u0, dofs_left), DirichletBC(u1, dofs_top)]

    A, b = create_matrix(a), create_vector(L)
    assemble_matrix(A, a, bcs=bcs)
    A.assemble()

    # Check the diagonal (only on the rank that owns the row)
    d = A.getDiagonal()
    if len(dof_corner) > 0 and dof_corner[0] < V.dofmap.index_map.size_local:
        d.array_r[dof_corner[0]] == 1.0

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


@pytest.mark.parametrize(
    'mesh_factory', [(UnitSquareMesh, (MPI.COMM_WORLD, 4, 4)),
                     (UnitSquareMesh,
                      (MPI.COMM_WORLD, 4, 4, CellType.quadrilateral)),
                     (UnitCubeMesh, (MPI.COMM_WORLD, 3, 3, 3),
                      ), (UnitCubeMesh, (MPI.COMM_WORLD, 3, 3, 3, CellType.hexahedron))])
def test_constant_bc(mesh_factory):
    """
    Test that setting a DirichletBC with a constant yields the same result as setting it with a function.
    """
    func, args = mesh_factory
    mesh = func(*args)
    V = FunctionSpace(mesh, ("CG", 1))
    c = Constant(mesh, PETSc.ScalarType(2))

    def on_boundary(x):
        return np.ones(x.shape[1], dtype=bool)

    tdim = mesh.topology.dim
    boundary_facets = locate_entities_boundary(mesh, tdim - 1, on_boundary)
    boundary_dofs = locate_dofs_topological(V, tdim - 1, boundary_facets)

    u_bc = Function(V)
    u_bc.x.array[:] = PETSc.ScalarType(c.value)

    bc_f = DirichletBC(u_bc, boundary_dofs)
    bc_c = DirichletBC(c, boundary_dofs, V)

    u_f = Function(V)
    set_bc(u_f.vector, [bc_f])

    u_c = Function(V)
    set_bc(u_c.vector, [bc_c])
    assert(np.allclose(u_f.vector.array, u_c.vector.array))


@pytest.mark.parametrize(
    'mesh_factory', [(UnitSquareMesh, (MPI.COMM_WORLD, 4, 4)),
                     (UnitSquareMesh,
                      (MPI.COMM_WORLD, 4, 4, CellType.quadrilateral)),
                     (UnitCubeMesh, (MPI.COMM_WORLD, 3, 3, 3),
                      ), (UnitCubeMesh, (MPI.COMM_WORLD, 3, 3, 3, CellType.hexahedron))])
def test_vector_constant_bc(mesh_factory):
    """
    Test that setting a DirichletBC with a vector valued constant yields the same result as setting it with a function.
    """
    func, args = mesh_factory
    mesh = func(*args)
    V = VectorFunctionSpace(mesh, ("CG", 1))
    assert(V.num_sub_spaces() == mesh.geometry.dim)
    vals = np.arange(mesh.geometry.dim, dtype=np.float64)
    c = Constant(mesh, PETSc.ScalarType(vals))

    def on_boundary(x):
        return np.ones(x.shape[1], dtype=bool)

    tdim = mesh.topology.dim
    Vs = [V.sub(i).collapse() for i in range(V.num_sub_spaces())]
    boundary_facets = locate_entities_boundary(mesh, tdim - 1, on_boundary)
    boundary_dofs = [locate_dofs_topological((V.sub(i), Vs[i]), tdim - 1, boundary_facets)
                     for i in range(V.num_sub_spaces())]

    u_bcs = [Function(Vs[i]) for i in range(V.num_sub_spaces())]
    bcs_f = []
    for i, u in enumerate(u_bcs):
        u_bcs[i].x.array[:] = PETSc.ScalarType(c.value[i])
        bcs_f.append(DirichletBC(u_bcs[i], boundary_dofs[i], V.sub(i)))

    boundary_dofs = locate_dofs_topological(V, tdim - 1, boundary_facets)
    bc_c = DirichletBC(c, boundary_dofs, V)

    u_f = Function(V)
    set_bc(u_f.vector, bcs_f)

    u_c = Function(V)
    set_bc(u_c.vector, [bc_c])
    assert(np.allclose(u_f.vector.array, u_c.vector.array))


@pytest.mark.parametrize(
    'mesh_factory', [(UnitSquareMesh, (MPI.COMM_WORLD, 4, 4)),
                     (UnitSquareMesh,
                      (MPI.COMM_WORLD, 4, 4, CellType.quadrilateral)),
                     (UnitCubeMesh, (MPI.COMM_WORLD, 3, 3, 3),
                      ), (UnitCubeMesh, (MPI.COMM_WORLD, 3, 3, 3, CellType.hexahedron))])
def test_sub_constant_bc(mesh_factory):
    """
    Test that setting a DirichletBC with on a component of a vector valued function
    yields the same result as setting it with a function.
    """
    func, args = mesh_factory
    mesh = func(*args)

    V = VectorFunctionSpace(mesh, ("CG", 1))
    c = Constant(mesh, PETSc.ScalarType(3.14))

    def on_boundary(x):
        return np.ones(x.shape[1], dtype=bool)

    tdim = mesh.topology.dim
    boundary_facets = locate_entities_boundary(mesh, tdim - 1, on_boundary)

    for i in range(V.num_sub_spaces()):
        Vi = V.sub(i).collapse()
        boundary_dofsi = locate_dofs_topological((V.sub(i), Vi), tdim - 1, boundary_facets)

        u_bci = Function(Vi)
        u_bci.x.array[:] = PETSc.ScalarType(c.value)

        bc_fi = DirichletBC(u_bci, boundary_dofsi, V.sub(i))

        boundary_dofs = locate_dofs_topological(V.sub(i), tdim - 1, boundary_facets)
        bc_c = DirichletBC(c, boundary_dofs, V.sub(i))

        u_f = Function(V)
        set_bc(u_f.vector, [bc_fi])

        u_c = Function(V)
        set_bc(u_c.vector, [bc_c])
        assert(np.allclose(u_f.vector.array, u_c.vector.array))
