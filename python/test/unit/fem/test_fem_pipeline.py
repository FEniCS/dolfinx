# Copyright (C) 2019 Jorgen Dokken
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import numpy as np
import pytest
from petsc4py import PETSc

from dolfin import (MPI, DirichletBC, Function, FunctionSpace, TestFunction,
                    TrialFunction, UnitCubeMesh, UnitIntervalMesh,
                    UnitSquareMesh)
from dolfin.cpp.mesh import CellType
from dolfin.fem import (apply_lifting, assemble_matrix, assemble_scalar,
                        assemble_vector, set_bc)
from ufl import dx, grad, inner


@pytest.mark.parametrize("n", [2, 3, 4])
@pytest.mark.parametrize("mesh", [UnitIntervalMesh(MPI.comm_world, 10),
                                  UnitSquareMesh(MPI.comm_world, 3, 4, CellType.triangle),
                                  UnitSquareMesh(MPI.comm_world, 3, 4, CellType.quadrilateral),
                                  UnitCubeMesh(MPI.comm_world, 2, 3, 2, CellType.tetrahedron),
                                  UnitCubeMesh(MPI.comm_world, 2, 3, 2, CellType.hexahedron)])
def test_manufactured_poisson(n, mesh):
    """ Manufactured Poisson problem, solving u = x[0]**p, where p is the
    degree of the Lagrange function space.

    """

    V = FunctionSpace(mesh, ("Lagrange", n))
    V_f = FunctionSpace(mesh, ("Lagrange", max(n - 2, 1)))
    u, v = TrialFunction(V), TestFunction(V)

    # Exact solution
    u_exact = Function(V)
    u_exact.interpolate(lambda x: x[:, 0]**n)

    # Source term
    f = Function(V_f)
    f.interpolate(lambda x: -n * (n - 1) * x[:, 0]**(n - 2))

    a = inner(grad(u), grad(v)) * dx
    L = inner(f, v) * dx

    u_bc = Function(V)
    u_bc.interpolate(lambda x: x[:, 0]**n)
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
    uh.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                          mode=PETSc.ScatterMode.FORWARD)

    error = assemble_scalar((u_exact - uh)**2 * dx)
    error = MPI.sum(mesh.mpi_comm(), error)
    assert error < 1.0e-14
