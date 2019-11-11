# Copyright (C) 2019 Jorgen Dokken and Garth N. Wells
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import numpy as np
import pytest
from petsc4py import PETSc

from dolfin import (MPI, DirichletBC, fem, Function, FunctionSpace, TestFunction,
                    TrialFunction, UnitCubeMesh, UnitIntervalMesh,
                    UnitSquareMesh)
from dolfin.cpp.mesh import CellType
from dolfin.cpp.refinement import refine
from dolfin.fem import (apply_lifting, assemble_matrix, assemble_scalar,
                        assemble_vector, set_bc)
from ufl import SpatialCoordinate, div, dx, grad, inner


@pytest.mark.parametrize("n", [2, 3, 4])
@pytest.mark.parametrize("component", [0, 1, 2])
@pytest.mark.parametrize("mesh", [
    UnitIntervalMesh(MPI.comm_world, 10),
    UnitSquareMesh(MPI.comm_world, 3, 4, CellType.triangle),
    UnitSquareMesh(MPI.comm_world, 3, 4, CellType.quadrilateral),
    UnitCubeMesh(MPI.comm_world, 2, 3, 2, CellType.tetrahedron),
    UnitCubeMesh(MPI.comm_world, 2, 3, 2, CellType.hexahedron)
])
def test_manufactured_poisson(n, mesh, component):
    """ Manufactured Poisson problem, solving u = x[component]**p, where p is the
    degree of the Lagrange function space.

    """
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
    u_bc.interpolate(lambda x: x[:, component]**n)
    bc = DirichletBC(V, u_bc, lambda x: np.full(x.shape[0], True))

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


@pytest.mark.parametrize("n", [1, 2, 3, 4])
@pytest.mark.parametrize("cell", [CellType.triangle, CellType.tetrahedron])
def test_convergence_rate_poisson_simplices(n, cell):
    """ Manufactured Poisson problem, solving u = Pi_{i=0}^gdim sin(pi*x_i) """
    if cell == CellType.triangle:
        mesh = UnitSquareMesh(MPI.comm_world, 4, 4)
    elif cell == CellType.tetrahedron:
        if n >= 2:
            N0 = 3
        else:
            N0 = 4
        mesh = UnitCubeMesh(MPI.comm_world, N0, N0, CellType.tetrahedron)

    refs = 5 if mesh.geometric_dimension() == 2 else 3
    errors = np.zeros(refs)
    for i in range(refs):
        cmap = fem.create_coordinate_map(mesh.ufl_domain())
        mesh.geometry.coord_mapping = cmap
        V = FunctionSpace(mesh, ("Lagrange", n))
        u, v = TrialFunction(V), TestFunction(V)


        def u_exact(x):
            u_exact = 1
            for component in range(mesh.geometry.dim):
                u_exact *= np.sin(np.pi * x[:, component])
            return u_exact

        # Exact solution
        Vh = FunctionSpace(mesh, ("Lagrange", n + 3))
        u_ex = Function(Vh)
        u_ex.interpolate(u_exact)

        # Source term
        f = - div(grad(u_ex))

        a = inner(grad(u), grad(v)) * dx
        L = inner(f, v) * dx

        u_bc = Function(V)
        u_bc.interpolate(lambda x: np.zeros(x.shape[0]))
        bc = DirichletBC(V, u_bc, lambda x: np.full(x.shape[0], True))

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

        mesh = refine(mesh, True)

    # Compute convergence rate
    rate = np.log(errors[1:] / errors[:-1]) / np.log(0.5)
    # Exclude first mesh
    refined_rates = rate[1:]
    assert min(refined_rates) > n + 0.9


@pytest.mark.parametrize("n", [1, 2, 3, 4])
@pytest.mark.parametrize("cell", [CellType.interval, CellType.quadrilateral, CellType.hexahedron])
def test_convergence_rate_poisson_non_simplices(n, cell):
    """ Manufactured Poisson problem, solving $u = Pi_{i=0}^gdim sin(pi*x_i)$ """
    N0 = 3
    if cell == CellType.interval:
        mesh = UnitIntervalMesh(MPI.comm_world, N0)
    elif cell == CellType.quadrilateral:
        mesh = UnitSquareMesh(MPI.comm_world, N0, N0, cell)
    elif cell == CellType.hexahedron:
        mesh = UnitCubeMesh(MPI.comm_world, N0, N0, N0, CellType.hexahedron)

    refs = 5 if mesh.geometric_dimension() == 2 else 3
    errors = np.zeros(refs)
    for i in range(refs):
        N = N0 * 2**i
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

        def u_exact(x):
            u_exact = 1
            for component in range(mesh.geometry.dim):
                u_exact *= np.sin(np.pi * x[:, component])
            return u_exact

        # Exact solution
        Vh = FunctionSpace(mesh, ("Lagrange", n + 3))
        u_ex = Function(Vh)
        u_ex.interpolate(u_exact)

        # Source term
        f = - div(grad(u_ex))

        a = inner(grad(u), grad(v)) * dx
        L = inner(f, v) * dx

        u_bc = Function(V)
        u_bc.interpolate(lambda x: np.zeros(x.shape[0]))
        bc = DirichletBC(V, u_bc, lambda x: np.full(x.shape[0], True))

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

    # Exclude first mesh
    refined_rates = rate[1:]
    assert min(refined_rates) > n + 0.9
