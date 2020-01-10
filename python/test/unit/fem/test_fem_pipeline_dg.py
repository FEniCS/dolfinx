# Copyright (C) 2019 Joe Dean, Jorgen Dokken, and Garth N. Wells
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import os

import numpy as np
import pytest
from petsc4py import PETSc
from dolfin import MPI, Function, FunctionSpace, FacetNormal, CellDiameter
from dolfin.cpp.mesh import GhostMode
from dolfin.fem import assemble_matrix, assemble_scalar, assemble_vector
from dolfin.io import XDMFFile
from ufl import (SpatialCoordinate, div, dx, grad, inner, ds, dS, avg, jump,
                 TestFunction, TrialFunction)


@pytest.mark.parametrize("n", [2, 3, 4])
@pytest.mark.parametrize("component", [0, 1, 2])
@pytest.mark.parametrize("filename", ["UnitCubeMesh_hexahedron.xdmf",
                                      "UnitCubeMesh_tetra.xdmf",
                                      "UnitSquareMesh_quad.xdmf",
                                      "UnitSquareMesh_triangle.xdmf"])
def test_manufactured_poisson_dg(n, filename, component, datadir):
    """ Manufactured Poisson problem, solving u = x[component]**n, where n is the
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
    u_exact = x[component] ** n

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
