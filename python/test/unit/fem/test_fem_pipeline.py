# Copyright (C) 2019 Jorgen Dokken, Matthew Scroggs and Garth N. Wells
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import os

import numpy as np
import pytest
from mpi4py import MPI
from petsc4py import PETSc

import ufl
from dolfinx import (DirichletBC, Function, FunctionSpace, fem, cpp,
                     UnitCubeMesh, UnitSquareMesh, VectorFunctionSpace,
                     common)
from dolfinx.fem import (apply_lifting, assemble_matrix, assemble_scalar,
                         assemble_vector, locate_dofs_topological, set_bc)
from dolfinx.cpp.mesh import CellType
from dolfinx.io import XDMFFile
from dolfinx_utils.test.skips import skip_if_complex
from ufl import (SpatialCoordinate, TestFunction, TrialFunction, div, dx, grad,
                 inner, ds, dS, avg, jump, FacetNormal, CellDiameter)


def get_mesh(cell_type, datadir):
    if MPI.COMM_WORLD.size == 1:
        # If running in serial, use a small mesh
        if cell_type in [CellType.triangle, CellType.quadrilateral]:
            return UnitSquareMesh(MPI.COMM_WORLD, 2, 1, cell_type)
        else:
            return UnitCubeMesh(MPI.COMM_WORLD, 2, 1, 1, cell_type)
    else:
        # In parallel, use larger meshes
        if cell_type == CellType.triangle:
            filename = "UnitSquareMesh_triangle.xdmf"
        elif cell_type == CellType.quadrilateral:
            filename = "UnitSquareMesh_quad.xdmf"
        elif cell_type == CellType.tetrahedron:
            filename = "UnitCubeMesh_tetra.xdmf"
        elif cell_type == CellType.hexahedron:
            filename = "UnitCubeMesh_hexahedron.xdmf"
        with XDMFFile(MPI.COMM_WORLD, os.path.join(datadir, filename), "r", encoding=XDMFFile.Encoding.ASCII) as xdmf:
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

    # Get quadrature degree for bilinear form integrand (ignores effect of non-affine map)
    a = inner(grad(u), grad(v)) * dx(metadata={"quadrature_degree": -1})
    a.integrals()[0].metadata()["quadrature_degree"] = ufl.algorithms.estimate_total_polynomial_degree(a)

    # Source term
    x = SpatialCoordinate(mesh)
    u_exact = x[1]**degree
    f = - div(grad(u_exact))

    # Set quadrature degree for linear form integrand (ignores effect of non-affine map)
    L = inner(f, v) * dx(metadata={"quadrature_degree": -1})
    L.integrals()[0].metadata()["quadrature_degree"] = ufl.algorithms.estimate_total_polynomial_degree(L)

    with common.Timer("Linear form compile"):
        L = fem.Form(L)

    with common.Timer("Function interpolation"):
        u_bc = Function(V)
        u_bc.interpolate(lambda x: x[1]**degree)

    # Create Dirichlet boundary condition
    mesh.topology.create_connectivity_all()
    facetdim = mesh.topology.dim - 1
    bndry_facets = np.where(np.array(cpp.mesh.compute_boundary_facets(mesh.topology)) == 1)[0]
    bdofs = locate_dofs_topological(V, facetdim, bndry_facets)
    assert(len(bdofs) < V.dim)
    bc = DirichletBC(u_bc, bdofs)

    with common.Timer("Vector assembly"):
        b = assemble_vector(L)
        apply_lifting(b, [a], [[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b, [bc])

    with common.Timer("Bilinear form compile"):
        a = fem.Form(a)

    with common.Timer("Matrix assembly"):
        A = assemble_matrix(a, [bc])
        A.assemble()

    # Create LU linear solver
    solver = PETSc.KSP().create(MPI.COMM_WORLD)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)
    solver.setOperators(A)

    with common.Timer("Solve"):
        uh = Function(V)
        solver.solve(b, uh.vector)
        uh.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    with common.Timer("Error functional compile"):
        M = (u_exact - uh)**2 * dx
        M = fem.Form(M)

    with common.Timer("Error assembly"):
        error = mesh.mpi_comm().allreduce(assemble_scalar(M), op=MPI.SUM)

    common.list_timings(MPI.COMM_WORLD, [common.TimingType.wall])
    assert np.absolute(error) < 1.0e-14


def run_vector_test(mesh, V, degree):
    """Projection into H(div/curl) spaces"""
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = inner(u, v) * dx

    # Source term
    x = SpatialCoordinate(mesh)
    u_exact = x[0] ** degree
    L = inner(u_exact, v[0]) * dx

    with common.Timer("Assemble vector"):
        b = assemble_vector(L)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    with common.Timer("Assemble matrix"):
        A = assemble_matrix(a)
        A.assemble()

    with common.Timer("Solve"):
        # Create LU linear solver (Note: need to use a solver that
        # re-orders to handle pivots, e.g. not the PETSc built-in LU solver)
        solver = PETSc.KSP().create(MPI.COMM_WORLD)
        solver.setType("preonly")
        solver.getPC().setType('lu')
        solver.setOperators(A)

        # Solve
        uh = Function(V)
        solver.solve(b, uh.vector)
        uh.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    with common.Timer("Error functional compile"):
        # Calculate error
        M = (u_exact - uh[0])**2 * dx
        for i in range(1, mesh.topology.dim):
            M += uh[i]**2 * dx
        M = fem.Form(M)

    with common.Timer("Error assembly"):
        error = mesh.mpi_comm().allreduce(assemble_scalar(M), op=MPI.SUM)

    common.list_timings(MPI.COMM_WORLD, [common.TimingType.wall])
    assert np.absolute(error) < 1.0e-14


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

    with common.Timer("Error functional compile"):
        # Calculate error
        M = (u_exact - uh)**2 * dx
        M = fem.Form(M)

    with common.Timer("Error assembly"):
        error = mesh.mpi_comm().allreduce(assemble_scalar(M), op=MPI.SUM)

    common.list_timings(MPI.COMM_WORLD, [common.TimingType.wall])
    assert np.absolute(error) < 1.0e-14


# Run tests on all spaces in periodic table on triangles and tetrahedra
@parametrize_cell_types_simplex
@pytest.mark.parametrize("family", ["Lagrange"])
@pytest.mark.parametrize("degree", [2, 3, 4])
def test_P_simplex(family, degree, cell_type, datadir):
    mesh = get_mesh(cell_type, datadir)
    V = FunctionSpace(mesh, (family, degree))
    run_scalar_test(mesh, V, degree)


@parametrize_cell_types_simplex
@pytest.mark.parametrize("family", ["Lagrange"])
@pytest.mark.parametrize("degree", [2, 3, 4])
def test_vector_P_simplex(family, degree, cell_type, datadir):
    mesh = get_mesh(cell_type, datadir)
    V = VectorFunctionSpace(mesh, (family, degree))
    run_vector_test(mesh, V, degree)


@parametrize_cell_types_simplex
@pytest.mark.parametrize("family", ["DG"])
@pytest.mark.parametrize("degree", [2, 3])
def test_dP_simplex(family, degree, cell_type, datadir):
    mesh = get_mesh(cell_type, datadir)
    V = FunctionSpace(mesh, (family, degree))
    run_dg_test(mesh, V, degree)


@parametrize_cell_types_simplex
@pytest.mark.parametrize("family", ["RT", "N1curl"])
@pytest.mark.parametrize("degree", [1, 2, 3, 4])
def test_RT_N1curl_simplex(family, degree, cell_type, datadir):
    mesh = get_mesh(cell_type, datadir)
    V = FunctionSpace(mesh, (family, degree))
    run_vector_test(mesh, V, degree - 1)


@parametrize_cell_types_simplex
@pytest.mark.parametrize("family", ["BDM", "N2curl"])
@pytest.mark.parametrize("degree", [1, 2])
def test_BDM_N2curl_simplex(family, degree, cell_type, datadir):
    mesh = get_mesh(cell_type, datadir)
    V = FunctionSpace(mesh, (family, degree))
    run_vector_test(mesh, V, degree)


# Skip slowest test in complex to stop CI timing out
@skip_if_complex
@parametrize_cell_types_simplex
@pytest.mark.parametrize("family", ["BDM", "N2curl"])
@pytest.mark.parametrize("degree", [3])
def test_BDM_N2curl_simplex_highest_order(family, degree, cell_type, datadir):
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


@parametrize_cell_types_tp
@pytest.mark.parametrize("family", ["Q"])
@pytest.mark.parametrize("degree", [2, 3, 4])
def test_vector_P_tp(family, degree, cell_type, datadir):
    mesh = get_mesh(cell_type, datadir)
    V = VectorFunctionSpace(mesh, (family, degree))
    run_vector_test(mesh, V, degree)


# TODO: Implement DPC spaces
@parametrize_cell_types_quad
@pytest.mark.parametrize("family", ["DQ"])
# @pytest.mark.parametrize("family", ["DQ", "DPC"])
@pytest.mark.parametrize("degree", [1, 2, 3])
def test_dP_quad(family, degree, cell_type, datadir):
    mesh = get_mesh(cell_type, datadir)
    V = FunctionSpace(mesh, (family, degree))
    run_dg_test(mesh, V, degree)


@parametrize_cell_types_hex
@pytest.mark.parametrize("family", ["DQ"])
# @pytest.mark.parametrize("family", ["DQ", "DPC"])
@pytest.mark.parametrize("degree", [1, 2])
def test_dP_hex(family, degree, cell_type, datadir):
    mesh = get_mesh(cell_type, datadir)
    V = FunctionSpace(mesh, (family, degree))
    run_dg_test(mesh, V, degree)


@parametrize_cell_types_quad
@pytest.mark.parametrize("family", ["RTCE", "RTCF"])
@pytest.mark.parametrize("degree", [1, 2, 3])
def test_RTC_quad(family, degree, cell_type, datadir):
    mesh = get_mesh(cell_type, datadir)
    V = FunctionSpace(mesh, (family, degree))
    run_vector_test(mesh, V, degree - 1)


@parametrize_cell_types_hex
@pytest.mark.parametrize("family", ["NCE", "NCF"])
@pytest.mark.parametrize("degree", [1, 2, 3])
def test_NC_hex(family, degree, cell_type, datadir):
    # TODO: Implement higher order NCE/NCF spaces
    if family == "NCE" and degree >= 3:
        return
    if family == "NCF" and degree >= 2:
        return
    mesh = get_mesh(cell_type, datadir)
    V = FunctionSpace(mesh, (family, degree))
    run_vector_test(mesh, V, degree - 1)


# TODO: Implement BDMCE spaces
# Note: These are currently not supported in FIAT
@parametrize_cell_types_quad
@pytest.mark.parametrize("family", ["BDMCE", "BDMCF"])
@pytest.mark.parametrize("degree", [1, 2, 3])
def xtest_BDM_quad(family, degree, cell_type, datadir):
    mesh = get_mesh(cell_type, datadir)
    V = FunctionSpace(mesh, (family, degree))
    run_vector_test(mesh, V, degree)


# TODO: Implement AA spaces
# Note: These are currently not supported in FIAT
@parametrize_cell_types_hex
@pytest.mark.parametrize("family", ["AAE", "AAF"])
@pytest.mark.parametrize("degree", [1, 2, 3])
def xtest_AA_hex(family, degree, cell_type, datadir):
    mesh = get_mesh(cell_type, datadir)
    V = FunctionSpace(mesh, (family, degree))
    run_vector_test(mesh, V, degree)
