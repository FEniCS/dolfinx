# Copyright (C) 2019 Jorgen Dokken and Garth N. Wells
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import os

import numpy as np
import pytest
from petsc4py import PETSc


from dolfin import (DirichletBC, fem, Function, FunctionSpace,
                    MPI, UnitIntervalMesh, UnitSquareMesh, UnitCubeMesh)
from dolfin.cpp.mesh import GhostMode, CellType
from dolfin.fem import (apply_lifting, assemble_matrix, assemble_scalar,
                        assemble_vector, set_bc)
from dolfin.io import XDMFFile
from dolfin.cpp.refinement import refine
from ufl import (SpatialCoordinate, TestFunction, TrialFunction, div, dx, grad,
                 inner)


@pytest.mark.parametrize("n", [2, 3, 4])
@pytest.mark.parametrize("component", [0, 1, 2])
@pytest.mark.parametrize("filename", ["UnitCubeMesh_hexahedron.xdmf",
                                      "UnitCubeMesh_tetra.xdmf",
                                      "UnitSquareMesh_quad.xdmf",
                                      "UnitSquareMesh_triangle.xdmf"])
def test_manufactured_poisson(n, filename, component, datadir):
    """ Manufactured Poisson problem, solving u = x[component]**p, where p is the
    degree of the Lagrange function space.

    """
    with XDMFFile(MPI.comm_world, os.path.join(datadir, filename)) as xdmf:
        mesh = xdmf.read_mesh(GhostMode.none)

    if component >= mesh.geometry.dim:
        return

    V = FunctionSpace(mesh, ("Lagrange", n))
    u, v = TrialFunction(V), TestFunction(V)

    # Exact solution
    x = SpatialCoordinate(mesh)
    u_exact = x[component]**n

    # Source term
    f = - div(grad(u_exact))

    a = inner(grad(u), grad(v)) * dx
    L = inner(f, v) * dx

    u_bc = Function(V)
    u_bc.interpolate(lambda x: x[component]**n)
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


@pytest.mark.parametrize("n", [1, 2, 3])
@pytest.mark.parametrize("filename", ["UnitCubeMesh_tetra.xdmf",
                                      "UnitSquareMesh_triangle.xdmf"])
def test_convergence_rate_poisson_simplices(n, filename, datadir):
    """ Manufactured Poisson problem, solving u = Pi_{i=0}^gdim sin(pi*x_i) """
    with XDMFFile(MPI.comm_world, os.path.join(datadir, filename)) as xdmf:
        mesh = xdmf.read_mesh(GhostMode.none)

    cmap = fem.create_coordinate_map(mesh.ufl_domain())

    refs = 3
    errors = np.zeros(refs)
    for i in range(refs):
        mesh = refine(mesh, False)
        mesh.geometry.coord_mapping = cmap

        V = FunctionSpace(mesh, ("Lagrange", n))

        u, v = TrialFunction(V), TestFunction(V)

        def u_exact(x):
            u = 1
            for component in range(mesh.geometry.dim):
                u *= np.sin(np.pi * x[component])
            return u

        # Exact solution
        Vh = FunctionSpace(mesh, ("Lagrange", n + 2))
        u_ex = Function(Vh)
        u_ex.interpolate(u_exact)
        # Source term
        f = - div(grad(u_ex))

        a = inner(grad(u), grad(v)) * dx
        L = inner(f, v) * dx

        u_bc = Function(V)
        u_bc.interpolate(lambda x: np.zeros(x.shape[1]))
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
        solver.getPC().setFactorSolverType("mumps")

        # Solve
        uh = Function(V)
        solver.solve(b, uh.vector)

        uh.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        error = assemble_scalar((u_ex - uh)**2 * dx)
        error = MPI.sum(mesh.mpi_comm(), error)
        errors[i] = np.sqrt(error)

    # Compute convergence rate
    rate = np.log(errors[1:] / errors[:-1]) / np.log(0.5)
    print(rate)
    assert rate[1] > n + 0.85
    assert rate[1] > n + 0.9


@pytest.mark.parametrize("n", [1, 2, 3])
@pytest.mark.parametrize("cell", [CellType.interval, CellType.quadrilateral, CellType.hexahedron])
def test_convergence_rate_poisson_non_simplices(n, cell):
    """ Manufactured Poisson problem, solving $u = Pi_{i=0}^gdim sin(pi*x_i)$ """
    if cell == CellType.interval:
        gdim = 1
    elif cell == CellType.quadrilateral:
        gdim = 2
    elif cell == CellType.hexahedron:
        gdim = 3

    def u_exact(x):
        u = 1
        for component in range(gdim):
            u *= np.sin(np.pi * x[component])
        return u

    refs = 3
    errors = np.zeros(refs)
    for i in range(refs):
        if (gdim == 3):
            N = 2**(i + 1)
        else:
            N = 2**(i + 3)
        if cell == CellType.interval:
            mesh = UnitIntervalMesh(MPI.comm_world, N)
        elif cell == CellType.quadrilateral:
            mesh = UnitSquareMesh(MPI.comm_world, N, N, cell)
        elif cell == CellType.hexahedron:
            mesh = UnitCubeMesh(MPI.comm_world, N, N, N, CellType.hexahedron)

        cmap = fem.create_coordinate_map(mesh.ufl_domain())
        mesh.geometry.coord_mapping = cmap
        V = FunctionSpace(mesh, ("Lagrange", n))
        u, v = TrialFunction(V), TestFunction(V)

        # Exact solution
        Vh = FunctionSpace(mesh, ("Lagrange", n + 2))
        u_ex = Function(Vh)
        u_ex.interpolate(u_exact)

        # Source term
        f = - div(grad(u_ex))

        a = inner(grad(u), grad(v)) * dx
        L = inner(f, v) * dx

        u_bc = Function(V)
        u_bc.interpolate(lambda x: np.zeros(x.shape[1]))
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

        error = assemble_scalar((u_ex - uh)**2 * dx)
        error = MPI.sum(mesh.mpi_comm(), error)
        errors[i] = np.sqrt(error)

    # Compute convergence rate
    rate = np.log(errors[1:] / errors[:-1]) / np.log(0.5)
    assert rate[0] > n + 0.8
    assert rate[1] > n + 0.95
