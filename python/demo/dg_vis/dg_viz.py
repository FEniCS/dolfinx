# Copyright (C) 2019 Jorgen Dokken, Matthew Scroggs and Garth N. Wells
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from dolfinx import MeshTags
from dolfinx import VectorFunctionSpace
import os

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

import ufl
import dolfinx.io
from dolfinx import (Function, FunctionSpace, fem, UnitSquareMesh,
                     common)
from dolfinx.fem import (assemble_matrix, assemble_scalar,
                         assemble_vector)
from dolfinx.cpp.mesh import CellType
from ufl import (SpatialCoordinate, TestFunction, TrialFunction, div, dx, grad,
                 inner, ds, dS, avg, jump, FacetNormal, CellDiameter)


def run_dg_test(mesh, V, degree):
    """ Manufactured Poisson problem, solving u = x[component]**n, where n is the
    degree of the Lagrange function space.
    """
    u, v = TrialFunction(V), TestFunction(V)

    # Exact solution
    x = SpatialCoordinate(mesh)
    u_exact = x[1] ** degree

    # Coefficient
    k = Function(V)
    k.vector.set(2.0)
    k.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    # Source term
    f = - div(k * grad(u_exact))

    # Mesh normals and element size
    n = FacetNormal(mesh)
    h = CellDiameter(mesh)
    h_avg = (h("+") + h("-")) / 2.0

    # Penalty parameter
    alpha = 32

    dx_ = dx(metadata={"quadrature_degree": -1})
    ds_ = ds(metadata={"quadrature_degree": -1})
    dS_ = dS(metadata={"quadrature_degree": -1})

    with common.Timer("Compile forms"):
        a = inner(k * grad(u), grad(v)) * dx_ \
            - k("+") * inner(avg(grad(u)), jump(v, n)) * dS_ \
            - k("+") * inner(jump(u, n), avg(grad(v))) * dS_ \
            + k("+") * (alpha / h_avg) * inner(jump(u, n), jump(v, n)) * dS_ \
            - inner(k * grad(u), v * n) * ds_ \
            - inner(u * n, k * grad(v)) * ds_ \
            + (alpha / h) * inner(k * u, v) * ds_
        L = inner(f, v) * dx_ - inner(k * u_exact * n, grad(v)) * ds_ \
            + (alpha / h) * inner(k * u_exact, v) * ds_

    for integral in a.integrals():
        integral.metadata()["quadrature_degree"] = ufl.algorithms.estimate_total_polynomial_degree(a)
    for integral in L.integrals():
        integral.metadata()["quadrature_degree"] = ufl.algorithms.estimate_total_polynomial_degree(L)

    with common.Timer("Assemble vector"):
        b = assemble_vector(L)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    with common.Timer("Assemble matrix"):
        A = assemble_matrix(a, [])
        A.assemble()

    with common.Timer("Solve"):
        # Create LU linear solver
        solver = PETSc.KSP().create(MPI.COMM_WORLD)
        solver.setType(PETSc.KSP.Type.PREONLY)
        solver.getPC().setType(PETSc.PC.Type.LU)
        solver.setOperators(A)

        # Solve
        uh = Function(V)
        solver.solve(b, uh.vector)
        uh.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                              mode=PETSc.ScatterMode.FORWARD)
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "u.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(uh)
    vtk = dolfinx.io.VTKFile("u.pvd")
    vtk.write(uh)
    # uh.vector[:] *= 2
    # vtk.write(uh)

    with common.Timer("Error functional compile"):
        # Calculate error
        M = (u_exact - uh)**2 * dx
        M = fem.Form(M)

    with common.Timer("Error assembly"):
        error = mesh.mpi_comm().allreduce(assemble_scalar(M), op=MPI.SUM)

    #common.list_timings(MPI.COMM_WORLD, [common.TimingType.wall])
    assert np.absolute(error) < 1.0e-14


# family = "DG"
# degree = 2
# cell_type = CellType.triangle
# mesh = UnitSquareMesh(MPI.COMM_WORLD, 3, 2)
# V = FunctionSpace(mesh, (family, degree))
# run_dg_test(mesh, V, degree)

mesh = UnitSquareMesh(MPI.COMM_WORLD, 15, 15)
V = VectorFunctionSpace(mesh, ("CG", 1))
u = Function(V)
mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim - 1)
num_facets = mesh.topology.index_map(mesh.topology.dim - 1).size_local
indices = np.arange(num_facets)
values = np.copy(indices)
ft = dolfinx.MeshTags(mesh, mesh.topology.dim - 1, indices, values)


def expr(x):
    vals = np.zeros((2, x.shape[1]))
    vals[0] = x[0]
    vals[1] = x[1]
    return vals


u.interpolate(expr)
vtk = dolfinx.io.VTKFile("u.pvd")
vtk.write(u)
