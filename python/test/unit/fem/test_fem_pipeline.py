# Copyright (C) 2019 Jorgen Dokken and Garth N. Wells
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import os
import time

import numpy as np
import pytest
import ufl
from dolfinx import (DirichletBC, Function, FunctionSpace, VectorFunctionSpace,
                     cpp, fem)
from dolfinx.fem import (apply_lifting, assemble_matrix, assemble_scalar,
                         assemble_vector, locate_dofs_topological, set_bc)
from dolfinx.io import XDMFFile
from dolfinx_utils.test.skips import skip_if_complex
from mpi4py import MPI
from petsc4py import PETSc
from ufl import (SpatialCoordinate, TestFunction, TrialFunction, div, dx, grad,
                 inner)


@pytest.mark.parametrize("filename", [
    "UnitCubeMesh_hexahedron.xdmf",
    "UnitCubeMesh_tetra.xdmf",
    "UnitSquareMesh_quad.xdmf",
    "UnitSquareMesh_triangle.xdmf"
])
@pytest.mark.parametrize("degree", [2, 3, 4])
def test_manufactured_poisson(degree, filename, datadir):
    """ Manufactured Poisson problem, solving u = x[1]**p, where p is the
    degree of the Lagrange function space.

    """

    with XDMFFile(MPI.COMM_WORLD, os.path.join(datadir, filename), "r", encoding=XDMFFile.Encoding.ASCII) as xdmf:
        mesh = xdmf.read_mesh(name="Grid")

    V = FunctionSpace(mesh, ("Lagrange", degree))
    u, v = TrialFunction(V), TestFunction(V)
    a = inner(grad(u), grad(v)) * dx

    # Get quadrature degree for bilinear form integrand (ignores effect
    # of non-affine map)
    a = inner(grad(u), grad(v)) * dx(metadata={"quadrature_degree": -1})
    a.integrals()[0].metadata()["quadrature_degree"] = ufl.algorithms.estimate_total_polynomial_degree(a)

    # Source term
    x = SpatialCoordinate(mesh)
    u_exact = x[1]**degree
    f = - div(grad(u_exact))

    # Set quadrature degree for linear form integrand (ignores effect of
    # non-affine map)
    L = inner(f, v) * dx(metadata={"quadrature_degree": -1})
    L.integrals()[0].metadata()["quadrature_degree"] = ufl.algorithms.estimate_total_polynomial_degree(L)

    t0 = time.time()
    L = fem.Form(L)
    t1 = time.time()
    print("Linear form compile time:", t1 - t0)

    u_bc = Function(V)
    u_bc.interpolate(lambda x: x[1]**degree)

    # Create Dirichlet boundary condition
    mesh.topology.create_connectivity_all()
    facetdim = mesh.topology.dim - 1
    bndry_facets = np.where(np.array(cpp.mesh.compute_boundary_facets(mesh.topology)) == 1)[0]
    bdofs = locate_dofs_topological(V, facetdim, bndry_facets)
    assert(len(bdofs) < V.dim)
    bc = DirichletBC(u_bc, bdofs)

    t0 = time.time()
    b = assemble_vector(L)
    apply_lifting(b, [a], [[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, [bc])
    t1 = time.time()
    print("Vector assembly time:", t1 - t0)

    t0 = time.time()
    a = fem.Form(a)
    t1 = time.time()
    print("Bilinear form compile time:", t1 - t0)

    t0 = time.time()
    A = assemble_matrix(a, [bc])
    A.assemble()
    t1 = time.time()
    print("Matrix assembly time:", t1 - t0)

    # Create LU linear solver
    solver = PETSc.KSP().create(MPI.COMM_WORLD)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)
    solver.setOperators(A)
    # Solve
    t0 = time.time()
    uh = Function(V)
    solver.solve(b, uh.vector)
    uh.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    t1 = time.time()
    print("Linear solver time:", t1 - t0)

    M = (u_exact - uh)**2 * dx
    t0 = time.time()
    M = fem.Form(M)
    t1 = time.time()
    print("Error functional compile time:", t1 - t0)

    t0 = time.time()
    error = mesh.mpi_comm().allreduce(assemble_scalar(M), op=MPI.SUM)

    t1 = time.time()

    print("Error assembly time:", t1 - t0)
    assert np.absolute(error) < 1.0e-14


@pytest.mark.parametrize("filename", [
    "UnitSquareMesh_triangle.xdmf",
    "UnitCubeMesh_tetra.xdmf",
    # "UnitSquareMesh_quad.xdmf",
    # "UnitCubeMesh_hexahedron.xdmf"
])
@pytest.mark.parametrize("family",
                         [
                             "BDM",
                             "N2curl"
                         ])
@pytest.mark.parametrize("degree", [1, 2, 3])
def test_manufactured_vector1(family, degree, filename, datadir):
    """Projection into H(div/curl) spaces"""

    with XDMFFile(MPI.COMM_WORLD, os.path.join(datadir, filename), "r", encoding=XDMFFile.Encoding.ASCII) as xdmf:
        mesh = xdmf.read_mesh(name="Grid")

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


# This is skipped in complex as it is very slow and causes CircleCI to time out
@skip_if_complex
@pytest.mark.parametrize("filename", [
    "UnitSquareMesh_triangle.xdmf",
    "UnitCubeMesh_tetra.xdmf",
    # "UnitSquareMesh_quad.xdmf",
    # "UnitCubeMesh_hexahedron.xdmf"
])
@pytest.mark.parametrize("family",
                         [
                             "RT",
                             "N1curl",
                         ])
@pytest.mark.parametrize("degree", [1, 2])
def test_manufactured_vector2(family, degree, filename, datadir):
    """Projection into H(div/curl) spaces"""

    with XDMFFile(MPI.COMM_WORLD, os.path.join(datadir, filename), "r", XDMFFile.Encoding.ASCII) as xdmf:
        mesh = xdmf.read_mesh(name="Grid")

    V = FunctionSpace(mesh, (family, degree + 1))
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


@pytest.mark.parametrize("filename", [
    "UnitSquareMesh_triangle.xdmf",
    "UnitCubeMesh_tetra.xdmf",
    "UnitSquareMesh_quad.xdmf",
    "UnitCubeMesh_hexahedron.xdmf"
])
@pytest.mark.parametrize("degree", [1, 2, 3])
def test_manufactured_vector_function_space(degree, filename, datadir):
    """Projection into H(div/curl) spaces"""

    with XDMFFile(MPI.COMM_WORLD, os.path.join(datadir, filename), "r", encoding=XDMFFile.Encoding.ASCII) as xdmf:
        mesh = xdmf.read_mesh(name="Grid")

    V = VectorFunctionSpace(mesh, ("Lagrange", degree))
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

    u_exact = Function(V)
    u_exact.interpolate(lambda x: np.array(
        [x[0]**degree if i == 0 else 0 * x[0] for i in range(mesh.topology.dim)]))

    M = inner(uh - u_exact, uh - u_exact) * dx
    M = fem.Form(M)
    error = mesh.mpi_comm().allreduce(assemble_scalar(M), op=MPI.SUM)

    assert np.absolute(error) < 1.0e-14
