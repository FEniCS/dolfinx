# Copyright (C) 2019 Joe Dean, Jorgen Dokken, and Garth N. Wells
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import numpy as np
import pytest
from petsc4py import PETSc
from dolfin import (MPI, Function, FunctionSpace, UnitCubeMesh, UnitIntervalMesh,
                    UnitSquareMesh, FacetNormal, CellDiameter)
from dolfin.cpp.mesh import CellType, Ordering
from dolfin.fem import (assemble_matrix, assemble_scalar, assemble_vector)
from ufl import (SpatialCoordinate, div, dx, grad, inner, ds, dS, avg, jump,
                 TestFunction, TrialFunction)


@pytest.mark.parametrize("p", [2, 3, 4])
@pytest.mark.parametrize("component", [0, 1, 2])
@pytest.mark.parametrize("mesh", [
    UnitIntervalMesh(MPI.comm_world, 10),
    UnitSquareMesh(MPI.comm_world, 3, 4, CellType.triangle),
    UnitSquareMesh(MPI.comm_world, 3, 4, CellType.quadrilateral),
    UnitCubeMesh(MPI.comm_world, 2, 3, 2, CellType.tetrahedron),
    UnitCubeMesh(MPI.comm_world, 2, 3, 2, CellType.hexahedron)
])
def test_manufactured_poisson(p, mesh, component):
    """ Manufactured Poisson problem, solving u = x[component]**p, where p is the
    degree of the Lagrange function space.
    """

    if component >= mesh.geometry.dim:
        return

    Ordering.order_simplex(mesh)

    V = FunctionSpace(mesh, ("DG", p))
    u, v = TrialFunction(V), TestFunction(V)

    # Exact solution
    x = SpatialCoordinate(mesh)
    u_exact = x[component]**p

    # Coefficient
    k = Function(V)
    k.vector.set(2.0)

    # Source term
    f = - div(k * grad(u_exact))

    # Mesh normals and element size
    n = FacetNormal(mesh)
    h = CellDiameter(mesh)
    h_avg = (h("+") + h("-")) / 2.0

    # Penalty parameter
    alpha = 32

    a = inner(k * grad(u), grad(v)) * dx \
        - inner(k("+") * avg(grad(u)), jump(v, n)) * dS \
        - inner(k("+") * avg(grad(v)), jump(u, n)) * dS \
        + (alpha / h_avg) * inner(k("+") * jump(u, n), jump(v, n)) * dS \
        - inner(k * grad(u), v * n) * ds \
        - inner(k * grad(v), u * n) * ds \
        + (alpha / h) * inner(k * u, v) * ds
    L = inner(f, v) * dx - inner(grad(v), k * u_exact * n) * ds + \
        (alpha / h) * inner(k * u_exact, v) * ds

    b = assemble_vector(L)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    A = assemble_matrix(a, [])
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

    assert np.absolute(error) < 1.0e-14
