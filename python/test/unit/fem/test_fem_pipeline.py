# Copyright (C) 2019 Jorgen Dokken and Garth N. Wells
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import os
import time

import numpy as np
import pytest
from petsc4py import PETSc

import ufl
from dolfinx import (MPI, DirichletBC, Function, FunctionSpace, fem, geometry,
                     FacetNormal, CellDiameter, UnitSquareMesh, UnitCubeMesh)
from dolfinx.fem import (apply_lifting, assemble_matrix, assemble_scalar,
                         assemble_vector, locate_dofs_topological, set_bc)
from dolfinx.io import XDMFFile
from dolfinx.cpp.mesh import CellType
from dolfinx_utils.test.skips import skip_in_parallel
from ufl import (SpatialCoordinate, TestFunction, TrialFunction, div, dx, grad,
                 inner, ds, dS, avg, jump)


def get_mesh(cell_type, datadir):
    if MPI.size(MPI.comm_world) == 1:
        if cell_type == CellType.triangle or cell_type == CellType.quadrilateral:
            mesh = UnitSquareMesh(MPI.comm_world, 2, 1, cell_type)
        else:
            mesh = UnitCubeMesh(MPI.comm_world, 2, 1, 1, cell_type)
        return mesh
    else:
        if cell_type == CellType.triangle:
            filename = "UnitSquareMesh_triangle.xdmf"
        elif cell_type == CellType.quadrilateral:
            filename = "UnitSquareMesh_quad.xdmf"
        elif cell_type == CellType.tetrahedron:
            filename = "UnitCubeMesh_tetra.xdmf"
        elif cell_type == CellType.hexahedron:
            filename = "UnitCubeMesh_hexahedron.xdmf"
        with XDMFFile(MPI.comm_world, os.path.join(datadir, filename), "r", encoding=XDMFFile.Encoding.ASCII) as xdmf:
            return xdmf.read_mesh(name="Grid")


parametrize_cell_types = pytest.mark.parametrize(
    "cell_type", [CellType.triangle, CellType.quadrilateral,
                  CellType.tetrahedron, CellType.hexahedron])
parametrize_cell_types_simplex = pytest.mark.parametrize(
    "cell_type", [CellType.triangle, CellType.tetrahedron])
parametrize_cell_types_tp = pytest.mark.parametrize(
    "cell_type", [CellType.quadrilateral, CellType.hexahedron])
parametrize_cell_types_quad = pytest.mark.parametrize(
    "cell_type", [CellType.quadrilateral])
parametrize_cell_types_hex = pytest.mark.parametrize(
    "cell_type", [CellType.hexahedron])


def run_scalar_test(mesh, V, degree):
    """ Manufactured Poisson problem, solving u = x[1]**p, where p is the
    degree of the Lagrange function space.

    """
    u, v = TrialFunction(V), TestFunction(V)
    a = inner(grad(u), grad(v)) * dx

    # Get quadrature degree for bilinear form integrand (ignores effect
    # of non-affine map)
    a = inner(grad(u), grad(v)) * dx(metadata={"quadrature_degree": -1})
    a.integrals()[0].metadata()["quadrature_degree"] = ufl.algorithms.estimate_total_polynomial_degree(a)

    # Source term
    x = SpatialCoordinate(mesh)
    u_exact = x[1] ** degree
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
    u_bc.interpolate(lambda x: x[1] ** degree)

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

    M = (u_exact - uh) ** 2 * dx
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


def run_vector_test(mesh, V, degree):
    """Projection into H(div/curl) spaces"""
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = inner(u, v) * dx

    # Source term
    x = SpatialCoordinate(mesh)
    u_ref = x[0] ** degree
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

    xp = np.array([0.33, 0.33, 0.0])
    tree = geometry.BoundingBoxTree(mesh, mesh.geometry.dim)
    cells = geometry.compute_first_entity_collision(tree, mesh, xp)

    up = uh.eval(xp, cells[0])
    print("test0:", up)
    print("test1:", xp[0] ** degree)

    u_exact = np.zeros(mesh.geometry.dim)
    u_exact[0] = xp[0] ** degree
    assert np.allclose(up, u_exact)


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

    error = assemble_scalar((u_exact - uh) ** 2 * dx)
    error = MPI.sum(mesh.mpi_comm(), error)

    assert np.absolute(error) < 1.0e-14


# Run tests on all spaces in periodic table on triangles and tetrahedra
@parametrize_cell_types_simplex
@pytest.mark.parametrize("family", ["Lagrange"])
@pytest.mark.parametrize("degree", [2, 3, 4])
def test_P_simplex(family, degree, cell_type, datadir):
    mesh = get_mesh(cell_type, datadir)
    V = FunctionSpace(mesh, (family, degree))
    run_scalar_test(mesh, V, degree)


# TODO: turn this test back on in parallel once ghosting is fixed
@skip_in_parallel
@parametrize_cell_types_simplex
@pytest.mark.parametrize("family", ["DG"])
@pytest.mark.parametrize("degree", [2, 3, 4])
def test_dP_simplex(family, degree, cell_type, datadir):
    mesh = get_mesh(cell_type, datadir)
    V = FunctionSpace(mesh, (family, degree))
    run_dg_test(mesh, V, degree)


# TODO: turn this test back on in parallel once ghosting is fixed
@skip_in_parallel
@parametrize_cell_types_simplex
@pytest.mark.parametrize("family", ["RT", "N1curl"])
@pytest.mark.parametrize("degree", [1, 2, 3])
def test_RT_N1curl_simplex(family, degree, cell_type, datadir):
    mesh = get_mesh(cell_type, datadir)
    V = FunctionSpace(mesh, (family, degree + 1))
    run_vector_test(mesh, V, degree)


# TODO: turn this test back on in parallel once ghosting is fixed
@skip_in_parallel
@parametrize_cell_types_simplex
@pytest.mark.parametrize("family", ["BDM", "N2curl"])
@pytest.mark.parametrize("degree", [1, 2, 3])
def test_BDM_N2curl_simplex(family, degree, cell_type, datadir):
    mesh = get_mesh(cell_type, datadir)
    V = FunctionSpace(mesh, (family, degree))
    run_vector_test(mesh, V, degree)


# Run tests on all spaces in periodic table on quadrilaterals and hexahedra
@parametrize_cell_types_tp
@pytest.mark.parametrize("family", ["Q"])
@pytest.mark.parametrize("degree", [2, 3, 4])
def test_P_tp(family, degree, cell_type, datadir):
    mesh = get_mesh(cell_type, datadir)
    V = FunctionSpace(mesh, (family, degree))
    run_scalar_test(mesh, V, degree)


# TODO: turn this test back on in parallel once ghosting is fixed
@skip_in_parallel
# TODO: Implement DPC spaces
@parametrize_cell_types_tp
@pytest.mark.parametrize("family", ["DQ"])
# @pytest.mark.parametrize("family", ["DQ", "DPC"])
@pytest.mark.parametrize("degree", [2, 3, 4])
def test_dP_tp(family, degree, cell_type, datadir):
    mesh = get_mesh(cell_type, datadir)
    V = FunctionSpace(mesh, (family, degree))
    run_dg_test(mesh, V, degree)


# TODO: turn this test back on in parallel once ghosting is fixed
@skip_in_parallel
# TODO: Implement RTCE and higher order RTCE spaces
@parametrize_cell_types_quad
# @pytest.mark.parametrize("family", ["RTCE", "RTCF"])
# @pytest.mark.parametrize("degree", [1, 2, 3])
@pytest.mark.parametrize("family", ["RTCF"])
@pytest.mark.parametrize("degree", [1])
def test_RTC_quad(family, degree, cell_type, datadir):
    mesh = get_mesh(cell_type, datadir)
    V = FunctionSpace(mesh, (family, degree))
    run_vector_test(mesh, V, degree)


# TODO: Implement NC spaces
@parametrize_cell_types_hex
@pytest.mark.parametrize("family", ["NCE", "NCF"])
@pytest.mark.parametrize("degree", [1, 2, 3])
def xtest_NC_hex(family, degree, cell_type, datadir):
    mesh = get_mesh(cell_type, datadir)
    V = FunctionSpace(mesh, (family, degree + 1))
    run_vector_test(mesh, V, degree)


# TODO: Implement BDMC spaces
@parametrize_cell_types_quad
@pytest.mark.parametrize("family", ["BDMCE", "BDMCF"])
@pytest.mark.parametrize("degree", [1, 2, 3])
def xtest_BDM_quad(family, degree, cell_type, datadir):
    mesh = get_mesh(cell_type, datadir)
    V = FunctionSpace(mesh, (family, degree))
    run_vector_test(mesh, V, degree)


# TODO: Implement AA spaces
@parametrize_cell_types_hex
@pytest.mark.parametrize("family", ["AAE", "AAF"])
@pytest.mark.parametrize("degree", [1, 2, 3])
def xtest_AA_hex(family, degree, cell_type, datadir):
    mesh = get_mesh(cell_type, datadir)
    V = FunctionSpace(mesh, (family, degree))
    run_vector_test(mesh, V, degree)
