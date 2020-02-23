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
from dolfinx import MPI, DirichletBC, Function, FunctionSpace, fem, geometry
from dolfinx.cpp.mesh import GhostMode, Ordering
from dolfinx.fem import (apply_lifting, assemble_matrix, assemble_scalar,
                         assemble_vector, locate_dofs_topological, set_bc)
from dolfinx.io import XDMFFile
from dolfinx_utils.test.skips import skip_if_complex, skip_in_parallel
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

    with XDMFFile(MPI.comm_world, os.path.join(datadir, filename)) as xdmf:
        mesh = xdmf.read_mesh(GhostMode.none)

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
    mesh.create_connectivity_all()
    facetdim = mesh.topology.dim - 1
    bndry_facets = np.where(np.array(
        mesh.topology.on_boundary(facetdim)) == 1)[0]
    bdofs = locate_dofs_topological(V, facetdim, bndry_facets)
    assert(len(bdofs) < V.dim())
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
    solver = PETSc.KSP().create(MPI.comm_world)
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
    error = assemble_scalar(M)
    error = MPI.sum(mesh.mpi_comm(), error)
    t1 = time.time()

    print("Error assembly time:", t1 - t0)
    assert np.absolute(error) < 1.0e-14


@skip_in_parallel
@pytest.mark.parametrize("filename", [
    "UnitSquareMesh_triangle.xdmf",
    "UnitCubeMesh_tetra.xdmf",
    # "UnitSquareMesh_quad.xdmf",
    # "UnitCubeMesh_hexahedron.xdmf"
])
@pytest.mark.parametrize("family",
                         [
                             ("BDM", 0),
                             ("RT", 1),
                             ("N2curl", 0),
                             ("N1curl", 1),
                         ])
@pytest.mark.parametrize("degree", [1, 2])
def test_manufactured_vector1(family, degree, filename, datadir):
    """Projection into H(div/curl) spaces"""

    with XDMFFile(MPI.comm_world, os.path.join(datadir, filename)) as xdmf:
        mesh = xdmf.read_mesh(GhostMode.none)

    # FIXME: these test are currently failing on unordered meshes
    if "tetra" in filename:
        if family[0] == "N1curl":
            Ordering.order_simplex(mesh)

    V = FunctionSpace(mesh, (family[0], degree + family[1]))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = inner(u, v) * dx

    xp = np.array([0.33, 0.33, 0.0])
    tree = geometry.BoundingBoxTree(mesh, mesh.geometry.dim)
    cells = geometry.compute_first_entity_collision(tree, mesh, xp)

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
    solver = PETSc.KSP().create(MPI.comm_world)
    solver.setType("preonly")
    solver.getPC().setType('lu')
    solver.setOperators(A)

    # Solve
    uh = Function(V)
    solver.solve(b, uh.vector)
    uh.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    up = uh.eval(xp, cells[0])
    print("test0:", up)
    print("test1:", xp[0]**degree)

    u_exact = np.zeros(mesh.geometry.dim)
    u_exact[0] = xp[0]**degree
    assert np.allclose(up, u_exact)


@skip_if_complex
@skip_in_parallel
@pytest.mark.parametrize("filename", ["UnitSquareMesh_triangle.xdmf",
                                      "UnitCubeMesh_tetra.xdmf",
                                      # "UnitSquareMesh_quad.xdmf",
                                      # "UnitCubeMesh_hexahedron.xdmf"
                                      ])
@pytest.mark.parametrize("family",
                         [
                             "RT",
                             "N1curl",
                         ])
@pytest.mark.parametrize("degree", [1, 2, 3])
def test_manufactured_vector2(family, degree, filename, datadir):
    """Projection into H(div/curl) spaces"""

    # Skip slowest tests
    if "tetra" in filename and degree > 2:
        return

    with XDMFFile(MPI.comm_world, os.path.join(datadir, filename)) as xdmf:
        mesh = xdmf.read_mesh(GhostMode.none)

    # FIXME: these test are currently failing on unordered meshes
    if "tetra" in filename:
        if family == "N1curl":
            Ordering.order_simplex(mesh)

    V = FunctionSpace(mesh, (family, degree + 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = inner(u, v) * dx

    xp = np.array([0.33, 0.33, 0.0])
    tree = geometry.BoundingBoxTree(mesh, mesh.geometry.dim)
    cells = geometry.compute_first_entity_collision(tree, mesh, xp)

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
    solver = PETSc.KSP().create(MPI.comm_world)
    solver.setType("preonly")
    solver.getPC().setType('lu')
    solver.setOperators(A)

    # Solve
    uh = Function(V)
    solver.solve(b, uh.vector)
    uh.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    up = uh.eval(xp, cells[0])
    print("test0:", up)
    print("test1:", xp[0]**degree)

    u_exact = np.zeros(mesh.geometry.dim)
    u_exact[0] = xp[0]**degree
    assert np.allclose(up, u_exact)
