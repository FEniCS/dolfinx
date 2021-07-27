# Copyright (C) 2021 Jorgen Dokken, Jack S. Hale, Matthew Scroggs and Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
import os
import numpy as np
import pytest
import ufl
from dolfinx import (DirichletBC, Function, FunctionSpace,
                     VectorFunctionSpace, RectangleMesh, cpp, fem, UnitCubeMesh, UnitSquareMesh)
from dolfinx.cpp.mesh import CellType, locate_entities_boundary
from dolfinx.fem import (LinearProblem, apply_lifting, assemble_matrix, assemble_scalar,
                         assemble_vector, locate_dofs_topological, set_bc)
from dolfinx.io import XDMFFile
from dolfinx_utils.test.skips import skip_if_complex
from mpi4py import MPI
from petsc4py import PETSc
from ufl import (CellDiameter, FacetNormal, SpatialCoordinate, TestFunction,
                 TrialFunction, avg, div, ds, dS, dx, grad, inner, jump)


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
    L = fem.Form(L)

    u_bc = Function(V)
    u_bc.interpolate(lambda x: x[1]**degree)

    # Create Dirichlet boundary condition
    mesh.topology.create_connectivity_all()
    facetdim = mesh.topology.dim - 1
    bndry_facets = np.where(np.array(cpp.mesh.compute_boundary_facets(mesh.topology)) == 1)[0]
    bdofs = locate_dofs_topological(V, facetdim, bndry_facets)
    bc = DirichletBC(u_bc, bdofs)

    b = assemble_vector(L)
    apply_lifting(b, [a], [[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, [bc])

    a = fem.Form(a)
    A = assemble_matrix(a, [bc])
    A.assemble()

    # Create LU linear solver
    solver = PETSc.KSP().create(MPI.COMM_WORLD)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)
    solver.setOperators(A)

    uh = Function(V)
    solver.solve(b, uh.vector)
    uh.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    M = (u_exact - uh)**2 * dx
    M = fem.Form(M)

    error = mesh.mpi_comm().allreduce(assemble_scalar(M), op=MPI.SUM)
    assert np.absolute(error) < 1.0e-14


def run_vector_test(mesh, V, degree):
    """Projection into H(div/curl) spaces"""
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = inner(u, v) * dx

    # Source term
    x = SpatialCoordinate(mesh)
    u_exact = x[0] ** degree
    L = inner(u_exact, v[0]) * dx

    b = assemble_vector(L)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    A = assemble_matrix(a)
    A.assemble()

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

    # Calculate error
    M = (u_exact - uh[0])**2 * dx
    for i in range(1, mesh.topology.dim):
        M += uh[i]**2 * dx
    M = fem.Form(M)

    error = mesh.mpi_comm().allreduce(assemble_scalar(M), op=MPI.SUM)
    assert np.absolute(error) < 1.0e-14


def run_dg_test(mesh, V, degree):
    """Manufactured Poisson problem, solving u = x[component]**n, where n is the
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
    solver = PETSc.KSP().create(MPI.COMM_WORLD)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)
    solver.setOperators(A)

    # Solve
    uh = Function(V)
    solver.solve(b, uh.vector)
    uh.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                          mode=PETSc.ScatterMode.FORWARD)

    # Calculate error
    M = (u_exact - uh)**2 * dx
    M = fem.Form(M)

    error = mesh.mpi_comm().allreduce(assemble_scalar(M), op=MPI.SUM)
    assert np.absolute(error) < 1.0e-14


def test_biharmonic():
    """Manufactured biharmonic problem.

    Solved using rotated Regge mixed finite element method. This is equivalent
    to the Helan-Herrmann-Johnson (HHJ) finite element method in
    two-dimensions. Solving w(x) = (1 - cos(2*pi*x))*(1 - cos(4*pi*y)) on
    Omega = [-1, 1]^2 with w in H^2_0(Omega)."""
    # TODO: Move and scale domain. Extend to 3D. Boundary conditions. Solve.
    # TODO: Possible to do 'patch test' like other tests here?
    mesh = RectangleMesh(MPI.COMM_WORLD, [np.array([-1.0, -1.0, 0.0]),
                                          np.array([1.0, 1.0, 0.0])], [32, 32], CellType.triangle)

    element = ufl.MixedElement([ufl.FiniteElement("Regge", ufl.triangle, 1),
                                ufl.FiniteElement("Lagrange", ufl.triangle, 2)])

    V = FunctionSpace(mesh, element)
    sigma, u = ufl.TrialFunctions(V)
    tau, v = ufl.TestFunctions(V)

    x = ufl.SpatialCoordinate(mesh)
    u_exact = (1.0 - ufl.cos(2.0 * ufl.pi * x[0])) * (1 - ufl.cos(4.0 * ufl.pi * x[1]))
    f_exact = div(grad(div(grad(u_exact))))

    # sigma and tau are tangential-tangential continuous according to the
    # H(curl curl) continuity of the Regge space. However, for the biharmonic
    # problem we require normal-normal continuity H (div div). Theorem 4.2 of
    # Lizao Li's PhD thesis shows that the latter space can be constructed by
    # the former through the action of the operator S:
    def S(tau):
        return tau - ufl.Identity(2) * ufl.tr(tau)

    sigma_S = S(sigma)
    tau_S = S(tau)

    # Normal-normal component of tensor field
    n = FacetNormal(mesh)

    # Discrete duality inner product eq. 4.5 Lizao Li's PhD thesis
    # Exterior facet term dropped due to w \in H_0^2(\Omega).
    def b(tau_S, v):
        return inner(tau_S, grad(grad(v))) * dx \
            - ufl.dot(ufl.dot(tau_S('+'), n('+')), n('+')) * jump(grad(v), n) * dS \
            - ufl.dot(ufl.dot(tau_S, n), n) * ufl.dot(grad(v), n) * ds

    # Symmetric formulation
    a = inner(sigma_S, tau_S) * dx + b(tau_S, u) - b(sigma_S, v)
    L = inner(f_exact, v) * dx

    # Strong (Dirichlet) boundary condition
    boundary_facets = locate_entities_boundary(mesh, mesh.topology.dim - 1, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = locate_dofs_topological([V.sub(1), V.sub(1)], mesh.topology.dim - 1, boundary_facets)

    V_1 = V.sub(1).collapse()
    zero_u = Function(V_1)
    with zero_u.vector.localForm() as zero_u_local:
        zero_u_local.set(0.0)

    bcs = [DirichletBC(zero_u, boundary_dofs, V.sub(1))]

    A = assemble_matrix(a, bcs=bcs)
    A.assemble()
    b = assemble_vector(L)
    apply_lifting(b, [a], [bcs])

    # Solve
    solver = PETSc.KSP().create(MPI.COMM_WORLD)
    PETSc.Options()["ksp_type"] = "preonly"
    PETSc.Options()["ksp_view"] = ""
    PETSc.Options()["pc_type"] = "lu"
    PETSc.Options()["pc_factor_mat_solver_type"] = "mumps"
    solver.setFromOptions()
    solver.setOperators(A)

    x_h = Function(V)
    solver.solve(b, x_h.vector)
    x_h.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                           mode=PETSc.ScatterMode.FORWARD)
    sigma_h, u_h = x_h.split()
    print(u_h.vector.norm())

    with XDMFFile(MPI.COMM_WORLD, "u_h.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(u_h)

    u = ufl.TrialFunction(V_1)
    v = ufl.TestFunction(V_1)

    a = inner(u, v) * dx
    L = inner(u_exact, v) * dx

    problem = LinearProblem(a, L, petsc_options={"ksp_type": "preonly",
                                                 "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"})
    u_exact_h = problem.solve()

    with XDMFFile(MPI.COMM_WORLD, "u_exact_h.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(u_exact_h)


def get_mesh(cell_type, datadir):
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


# Run tests on all spaces in periodic table on triangles and tetrahedra
@parametrize_cell_types_simplex
@pytest.mark.parametrize("family", ["Lagrange"])
@pytest.mark.parametrize("degree", [2, 3, 4])
def test_P_simplex(family, degree, cell_type, datadir):
    if cell_type == CellType.tetrahedron and degree == 4:
        pytest.skip("Skip expensive test on tetrahedron")
    mesh = get_mesh(cell_type, datadir)
    V = FunctionSpace(mesh, (family, degree))
    run_scalar_test(mesh, V, degree)


@parametrize_cell_types_simplex
@pytest.mark.parametrize("family", ["Lagrange"])
@pytest.mark.parametrize("degree", [2, 3, 4])
def test_P_simplex_built_in(family, degree, cell_type, datadir):
    if cell_type == CellType.tetrahedron:
        mesh = UnitCubeMesh(MPI.COMM_WORLD, 5, 5, 5)
    elif cell_type == CellType.triangle:
        mesh = UnitSquareMesh(MPI.COMM_WORLD, 5, 5)
    V = FunctionSpace(mesh, (family, degree))
    run_scalar_test(mesh, V, degree)


@parametrize_cell_types_simplex
@pytest.mark.parametrize("family", ["Lagrange"])
@pytest.mark.parametrize("degree", [2, 3, 4])
def test_vector_P_simplex(family, degree, cell_type, datadir):
    if cell_type == CellType.tetrahedron and degree == 4:
        pytest.skip("Skip expensive test on tetrahedron")
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
    if cell_type == CellType.tetrahedron and degree == 4:
        pytest.skip("Skip expensive test on tetrahedron")
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


# Run tests on all spaces in periodic table on quadrilaterals and
# hexahedra
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
def test_P_tp_built_in_mesh(family, degree, cell_type, datadir):
    if cell_type == CellType.hexahedron:
        mesh = UnitCubeMesh(MPI.COMM_WORLD, 5, 5, 5, cell_type)
    elif cell_type == CellType.quadrilateral:
        mesh = UnitSquareMesh(MPI.COMM_WORLD, 5, 5, cell_type)
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


@parametrize_cell_types_quad
@pytest.mark.parametrize("family", ["DQ", "DPC"])
@pytest.mark.parametrize("degree", [1, 2, 3])
def test_dP_quad(family, degree, cell_type, datadir):
    if family == "DPC":
        pytest.skip("DPC space currently not implemented in basix.")
    mesh = get_mesh(cell_type, datadir)
    V = FunctionSpace(mesh, (family, degree))
    run_dg_test(mesh, V, degree)


@parametrize_cell_types_hex
@pytest.mark.parametrize("family", ["DQ", "DPC"])
@pytest.mark.parametrize("degree", [1, 2])
def test_dP_hex(family, degree, cell_type, datadir):
    if family == "DPC":
        pytest.skip("DPC space currently not implemented in basix.")
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
    mesh = get_mesh(cell_type, datadir)
    V = FunctionSpace(mesh, (family, degree))
    run_vector_test(mesh, V, degree - 1)


@parametrize_cell_types_quad
@pytest.mark.parametrize("family", ["BDMCE", "BDMCE"])
@pytest.mark.parametrize("degree", [1, 2, 3])
def test_BDM_quad(family, degree, cell_type, datadir):
    pytest.skip("BDMCE and BDMCE spaces currently not implemented in basix")
    mesh = get_mesh(cell_type, datadir)
    V = FunctionSpace(mesh, (family, degree))
    run_vector_test(mesh, V, degree)


@parametrize_cell_types_hex
@pytest.mark.parametrize("family", ["AAE", "AAE"])
@pytest.mark.parametrize("degree", [1, 2, 3])
def test_AA_hex(family, degree, cell_type, datadir):
    pytest.skip("AAE and AAEÎ spaces currently not implemented in basix")
    mesh = get_mesh(cell_type, datadir)
    V = FunctionSpace(mesh, (family, degree))
    run_vector_test(mesh, V, degree)
