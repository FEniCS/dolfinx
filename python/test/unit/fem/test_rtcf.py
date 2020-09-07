# Copyright (C) 2019-2020 Matthew Scroggs, Jorgen Dokken and Garth N. Wells
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from math import log
from random import shuffle, choice
import numpy as np
import pytest
import ufl
from dolfinx import Function, FunctionSpace, VectorFunctionSpace, fem
from dolfinx.fem import assemble_matrix, assemble_scalar, assemble_vector
from dolfinx.mesh import create_mesh
from dolfinx_utils.test.skips import skip_in_parallel, skip_if_complex
from mpi4py import MPI
from petsc4py import PETSc
from ufl import SpatialCoordinate, dx, inner, div


@skip_in_parallel
@pytest.mark.parametrize("family", ["RTCF"])
@pytest.mark.parametrize("degree", [1])
def test_manufactured_vector(family, degree):
    """Projection into H(div/curl) spaces"""

    N = 20
    points = np.array([[i / N, j / N] for i in range(N + 1) for j in range(N + 1)])
    order = [i for i, _ in enumerate(points)]
    shuffle(order)
    ordered_points = np.zeros_like(points)
    for i, j in enumerate(order):
        ordered_points[j] = points[i]
    cells = []
    for i in range(N):
        for j in range(N):
            start = i * (N + 1) + j
            cell = [start, start + 1, start + N + 1, start + N + 2]
            cells.append([order[cell[k]]
                          for k in choice([[0, 1, 2, 3], [2, 0, 3, 1],
                                           [3, 2, 1, 0], [1, 3, 0, 2]])])
    cells = np.array(cells)

    domain = ufl.Mesh(ufl.VectorElement("Lagrange", "quadrilateral", 1))
    mesh = create_mesh(MPI.COMM_WORLD, cells, ordered_points, domain)
    mesh.topology.create_connectivity_all()

    V = FunctionSpace(mesh, (family, degree))
    W = VectorFunctionSpace(mesh, ("CG", degree))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = inner(u, v) * dx

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
    solver = PETSc.KSP().create(MPI.COMM_WORLD)
    solver.setType("preonly")
    solver.getPC().setType('lu')
    solver.setOperators(A)

    # Solve
    uh = Function(V)
    solver.solve(b, uh.vector)
    uh.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    u_exact = Function(W)
    u_exact.interpolate(lambda x: np.array(
        [x[0]**degree if i == 0 else 0 * x[0] for i in range(mesh.topology.dim)]))

    M = inner(uh - u_exact, uh - u_exact) * dx
    M = fem.Form(M)
    error = mesh.mpi_comm().allreduce(assemble_scalar(M), op=MPI.SUM)

    assert np.absolute(error) < 1.0e-14


@skip_in_parallel
@skip_if_complex
def test_div():
    points = np.array([[0., 0.], [0., 1.], [1., 0.], [2., 1.]])
    cells = np.array([[0, 1, 2, 3]])

    domain = ufl.Mesh(ufl.VectorElement("Lagrange", "quadrilateral", 1))
    mesh = create_mesh(MPI.COMM_WORLD, cells, points, domain)
    mesh.topology.create_connectivity_all()

    RT = FunctionSpace(mesh, ("RTCF", 1))
    tau = ufl.TestFunction(RT)
    a = div(tau) * dx
    v = assemble_vector(a)

    v = sorted(list(v[:]))

    # Assert that these values match those computed elsewhere using sympy
    actual = [-1.0, 1 / 2 - 2 * log(2), 1, 1 / 2 + log(2)]
    for a, b in zip(v, actual):
        assert abs(a - b) < 0.01
