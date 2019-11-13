# Copyright (C) 2019 Jorgen Dokken and Garth N. Wells
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import numpy as np
import pytest
from petsc4py import PETSc

from dolfinx import (MPI, DirichletBC, Function, FunctionSpace, UnitCubeMesh,
                    UnitIntervalMesh, UnitSquareMesh)
from dolfinx.cpp.mesh import CellType
from dolfinx.fem import (apply_lifting, assemble_matrix, assemble_scalar,
                        assemble_vector, set_bc)
from ufl import (SpatialCoordinate, TestFunction, TrialFunction, div, dx, grad,
                 inner)


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
