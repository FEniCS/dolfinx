# Copyright (C) 2019 Jorgen Dokken and Garth N. Wells
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import os

import numpy as np
import pytest
from petsc4py import PETSc

import ufl
from dolfinx import MPI, DirichletBC, Function, FunctionSpace, geometry
from dolfinx.cpp.mesh import GhostMode
from dolfinx.fem import (apply_lifting, assemble_matrix, assemble_scalar,
                         assemble_vector, set_bc)
from dolfinx.io import XDMFFile
from dolfinx_utils.test.skips import skip_in_parallel
from ufl import (SpatialCoordinate, TestFunction, TrialFunction, div, dx, grad,
                 inner)


@pytest.mark.parametrize("filename", ["UnitSquareMesh_triangle.xdmf",
                                      "UnitCubeMesh_tetra.xdmf",
                                      "UnitCubeMesh_hexahedron.xdmf",
                                      "UnitSquareMesh_quad.xdmf"])
# @pytest.mark.parametrize("degree", [2, 3, 4])
@pytest.mark.parametrize("degree", [2, 3])
def test_manufactured_poisson(degree, filename, datadir):
    """ Manufactured Poisson problem, solving u = x[i]**p, where p is the
    degree of the Lagrange function space.

    """

    with XDMFFile(MPI.comm_world, os.path.join(datadir, filename)) as xdmf:
        mesh = xdmf.read_mesh(GhostMode.none)

    V = FunctionSpace(mesh, ("Lagrange", degree))
    u, v = TrialFunction(V), TestFunction(V)
    a = inner(grad(u), grad(v)) * dx

    for component in range(1):
        # Exact solution
        x = SpatialCoordinate(mesh)
        u_exact = x[component]**degree

        # Source term
        f = - div(grad(u_exact))
        L = inner(f, v) * dx

        u_bc = Function(V)
        u_bc.interpolate(lambda x: x[component]**degree)
        bc = DirichletBC(V, u_bc, lambda x: np.full(x.shape[1], True))

        b = assemble_vector(L)
        apply_lifting(b, [a], [[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b, [bc])

        A = assemble_matrix(a, [bc])
        A.assemble()

        # Create LU linear solver
        solver = PETSc.KSP().create(MPI.comm_world)
        solver.setType(PETSc.KSP.Type.PREONLY)
        solver.getPC().setType(PETSc.PC.Type.LU)
        solver.setOperators(A)

        # Solve
        uh = Function(V)
        solver.solve(b, uh.vector)
        uh.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        error = assemble_scalar((u_exact - uh)**2 * dx)
        error = MPI.sum(mesh.mpi_comm(), error)
        assert np.absolute(error) < 1.0e-14


# @pytest.mark.parametrize("filename", ["UnitSquareMesh_triangle.xdmf",
#                                       #   "UnitCubeMesh_tetra.xdmf",
#                                       #   "UnitCubeMesh_hexahedron.xdmf",
#                                       #   "UnitSquareMesh_quad.xdmf"
#                                       ])
# @pytest.mark.parametrize("degree", [1])
# def test_manufactured_poisson_mixed(degree, filename, datadir):
@skip_in_parallel
def test_manufactured_poisson_mixed(datadir):
    """ Manufactured Poisson problem, solving u = x[i]**p, where p is the
    degree of the Lagrange function space.

    """

    # mesh = dolfinx.UnitSquareMesh(dolfinx.MPI.comm_world, 16, 16)
    filename = "UnitSquareMesh_triangle.xdmf"
    with XDMFFile(MPI.comm_world, os.path.join(datadir, filename)) as xdmf:
        mesh = xdmf.read_mesh(GhostMode.none)

    degree = 1
    V = FunctionSpace(mesh, ("BDM", degree))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = inner(u, v) * dx

    xp = np.array([0.51, 0.51, 0.0])
    tree = geometry.BoundingBoxTree(mesh, mesh.geometry.dim)
    cells = geometry.compute_first_entity_collision(tree, mesh, xp)

    # Source term
    x = SpatialCoordinate(mesh)
    u_ref = x[0]**degree
    L = inner(u_ref, v[0]) * dx

    b = assemble_vector(L)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    A = assemble_matrix(a)
    A.assemble()

    # Create LU linear solver (Note: need to use a solver that
    # re-orders to handle pivots, e.g. not the PETSc built-in LU
    # solver)
    solver = PETSc.KSP().create(MPI.comm_world)
    solver.setType("preonly")
    solver.getPC().setType('lu')
    # solver.getPC().setFactorSolverType('umfpack')
    solver.setOperators(A)

    # Solve
    uh = Function(V)
    solver.solve(b, uh.vector)
    uh.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    up = uh.eval(xp, cells[0])
    print("test0:", up)
    print("test1:", xp[0]**degree)

    u_exact = np.zeros(mesh.geometry.dim)
    u_exact[0] = xp[0]**degree
    assert np.allclose(up, u_exact)
