# Copyright (C) 2019 Jorgen Dokken
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import numpy as np
import pytest
import sympy as sp
from petsc4py import PETSc
from dolfin import (MPI, DirichletBC, Function, FunctionSpace, TestFunction,
                    TrialFunction, UnitCubeMesh, UnitIntervalMesh,
                    UnitSquareMesh, solve)
from dolfin.fem import assemble_scalar
from dolfin.cpp.mesh import CellType
from ufl import SpatialCoordinate, dx, grad, inner, div


def ManufacturedSolution(degree, gdim):
    """
    Generate manufactured solution as function of x,y,z and the corresponding Laplacian
    """
    xvec = sp.symbols("x[0] x[1] x[2]")
    x, y, z = xvec
    u = (x)**(degree - 1) * (1 - x)

    if gdim > 1:
        u *= (y)**(degree - 1) * (1 - y)
    if gdim > 2:
        u *= (z)**(degree - 1) * (1 - z)

    return u


def boundary(x):
    """Boundary marker """
    condition = np.logical_or(x[:, 0] < 1.0e-6, x[:, 0] > 1.0 - 1.0e-6)
    for dim in range(1, x.shape[1]):
        c_dim = np.logical_or(x[:, dim] < 1.0e-6, x[:, dim] > 1.0 - 1.0e-6)
        condition = np.logical_or(condition, c_dim)
    return condition


@pytest.mark.parametrize("degree", [2, 3, 4])
@pytest.mark.parametrize("cell_type, gdim", [(CellType.interval, 1), (CellType.triangle, 2),
                                             (CellType.quadrilateral, 2), (CellType.tetrahedron, 3),
                                             (CellType.hexahedron, 3)])
def test_manufactured_poisson(degree, cell_type, gdim):
    """ Manufactured Poisson problem, solving u = Pi_{i=0}^gdim (1 - x[i]) * x[i]^(p - 1)
    where p is the degree of the Lagrange function space. Solved on the
    gdim-dimensional UnitMesh, with homogeneous Dirichlet boundary
    conditions.

    Compares values at each dof with the analytical solution.

    """
    if gdim == 1:
        mesh = UnitIntervalMesh(MPI.comm_world, 10)
    if gdim == 2:
        mesh = UnitSquareMesh(MPI.comm_world, 15, 15, cell_type)
    elif gdim == 3:
        mesh = UnitCubeMesh(MPI.comm_world, 5, 5, 5, cell_type)
    V = FunctionSpace(mesh, ("CG", degree))
    u, v = TrialFunction(V), TestFunction(V)

    x = SpatialCoordinate(mesh)  # noqa: F841
    u_exact = eval(str(ManufacturedSolution(degree, gdim)))

    f = -div(grad(u_exact))
    a = inner(grad(u), grad(v)) * dx(metadata={'quadrature_degree': degree + 3})
    L = inner(f, v) * dx(metadata={'quadrature_degree': degree + 3})

    u_bc = Function(V)
    with u_bc.vector.localForm() as u_local:
        u_local.set(0.0)
    bc = DirichletBC(V, u_bc, boundary)

    uh = Function(V)
    solve(a == L, uh, bc)
    uh.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    error = assemble_scalar((u_exact - uh)**2 * dx(metadata={'quadrature_degree': degree + 3}))
    error = MPI.sum(mesh.mpi_comm(), error)

    if cell_type == CellType.tetrahedron:
        assert np.sqrt(error) < 2e-4
    else:
        assert np.sqrt(error) < 5e-6
