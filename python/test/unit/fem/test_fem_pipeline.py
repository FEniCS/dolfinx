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
                    UnitSquareMesh)
from dolfin.cpp.mesh import CellType
from dolfin.fem import (apply_lifting, assemble_matrix, assemble_scalar,
                        assemble_vector, set_bc)
from ufl import SpatialCoordinate, div, dx, grad, inner


def ManufacturedSolution(degree, gdim):
    """Generate manufactured solution as function of x,y,z and the
    corresponding Laplacian.
    """
    xvec = sp.symbols("x[0] x[1] x[2]")
    x, y, z = xvec
    u = (x)**(degree - 1) * (1 - x)
    if gdim > 1:
        u *= (y)**(degree - 1) * (1 - y)
    if gdim > 2:
        u *= (z)**(degree - 1) * (1 - z)

    u_dolfin = u
    for i in range(gdim):
        x_d = sp.Symbol("x[:,{0:d}]".format(i))
        u_dolfin = u_dolfin.subs(xvec[i], x_d)
    return u, u_dolfin


def boundary(x):
    """Boundary marker """
    return np.full(x.shape[0], True)


@pytest.mark.parametrize("degree", [2, 3, 4])
@pytest.mark.parametrize("cell_type, gdim", [(CellType.interval, 1),
                                             (CellType.triangle, 2),
                                             (CellType.quadrilateral, 2),
                                             (CellType.tetrahedron, 3),
                                             (CellType.hexahedron, 3)])
def test_manufactured_poisson(degree, cell_type, gdim):
    """ Manufactured Poisson problem, solving u = Pi_{i=0}^gdim (1 - x[i]) * x[i]^(p - 1)
    where p is the degree of the Lagrange function space. Solved on the
    gdim-dimensional UnitMesh, with homogeneous Dirichlet boundary
    conditions.
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
    u_exact, u_dolfin = ManufacturedSolution(degree, gdim)
    u_exact = eval(str(u_exact))
    u_ex = Function(V)
    u_ex.interpolate(lambda x: eval(str(u_dolfin)))

    f = -div(grad(u_exact))
    a = inner(grad(u), grad(v)) * dx(metadata={'quadrature_degree': 2 * degree})
    L = inner(f, v) * dx(metadata={'quadrature_degree': 2 * degree})

    u_bc = Function(V)
    with u_bc.vector.localForm() as u_local:
        u_local.set(0.0)
    bc = DirichletBC(V, u_bc, boundary)

    uh = Function(V)
    b = assemble_vector(L)
    apply_lifting(b, [a], [[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, [bc])

    A = assemble_matrix(a, [bc])
    A.assemble()

    # Create CG Krylov solver and turn convergence monitoring on
    opts = PETSc.Options()
    opts["ksp_type"] = "preonly"
    opts["pc_type"] = "lu"
    solver = PETSc.KSP().create(MPI.comm_world)
    solver.setFromOptions()
    solver.setOperators(A)
    solver.solve(b, uh.vector)

    uh.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    error = assemble_scalar((u_ex - uh)**2 * dx(metadata={'quadrature_degree': 2 * degree}))
    error = MPI.sum(mesh.mpi_comm(), error)
    assert np.sqrt(error) < 5e-5
