# Copyright (C) 2019 Jorgen Dokken and Garth N. Wells
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import os

import numpy as np
import pytest
from petsc4py import PETSc

from dolfinx import MPI, DirichletBC, Function, FunctionSpace
from dolfinx.cpp.mesh import GhostMode
from dolfinx.fem import (apply_lifting, assemble_matrix, assemble_scalar,
                         assemble_vector, set_bc)
from dolfinx.io import XDMFFile
from ufl import (SpatialCoordinate, TestFunction, TrialFunction, div, dx, grad,
                 inner)


@pytest.mark.parametrize("filename", ["UnitCubeMesh_hexahedron.xdmf",
                                      "UnitCubeMesh_tetra.xdmf",
                                      "UnitSquareMesh_quad.xdmf",
                                      "UnitSquareMesh_triangle.xdmf"])
@pytest.mark.parametrize("degree", [2, 3, 4])
def test_manufactured_poisson(degree, filename, datadir):
    """ Manufactured Poisson problem, solving u = x[i]**p, where p is the
    degree of the Lagrange function space.

    """

    with XDMFFile(MPI.comm_world, os.path.join(datadir, filename)) as xdmf:
        mesh = xdmf.read_mesh(GhostMode.none)

    V = FunctionSpace(mesh, ("Lagrange", degree))
    u, v = TrialFunction(V), TestFunction(V)

    for component in range(mesh.geometry.dim):
        # Exact solution
        x = SpatialCoordinate(mesh)
        u_exact = x[component]**degree

        # Source term
        f = - div(grad(u_exact))

        a = inner(grad(u), grad(v)) * dx
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
