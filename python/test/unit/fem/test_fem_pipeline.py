# Copyright (C) 2021 Jorgen Dokken, Jack S. Hale, Matthew Scroggs and Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from pathlib import Path

from mpi4py import MPI

import numpy as np
import pytest

import basix
import dolfinx
import ufl
from basix.ufl import element, mixed_element
from dolfinx import default_real_type, la
from dolfinx.fem import (
    Function,
    apply_lifting,
    assemble_matrix,
    assemble_scalar,
    assemble_vector,
    dirichletbc,
    form,
    functionspace,
    locate_dofs_topological,
    set_bc,
)
from dolfinx.io import XDMFFile
from dolfinx.mesh import (
    CellType,
    create_rectangle,
    create_unit_cube,
    create_unit_square,
    exterior_facet_indices,
    locate_entities_boundary,
)
from ufl import (
    CellDiameter,
    FacetNormal,
    SpatialCoordinate,
    TestFunction,
    TrialFunction,
    avg,
    div,
    dS,
    ds,
    dx,
    grad,
    inner,
    jump,
)


def run_scalar_test(mesh, V, degree, cg_solver):
    """Manufactured Poisson problem, solving u = x[1]**p, where p is the
    degree of the Lagrange function space.
    """
    dtype = mesh.geometry.x.dtype
    u, v = TrialFunction(V), TestFunction(V)
    a = inner(grad(u), grad(v)) * dx

    # Get quadrature degree for bilinear form integrand (ignores effect of non-affine map)
    a = inner(grad(u), grad(v)) * dx(metadata={"quadrature_degree": -1})
    a.integrals()[0].metadata()["quadrature_degree"] = (
        ufl.algorithms.estimate_total_polynomial_degree(a)
    )
    a = form(a, dtype=dtype)

    # Source term
    x = SpatialCoordinate(mesh)
    u_exact = x[1] ** degree
    f = -div(grad(u_exact))

    # Set quadrature degree for linear form integrand (ignores effect of non-affine map)
    L = inner(f, v) * dx(metadata={"quadrature_degree": -1})
    L.integrals()[0].metadata()["quadrature_degree"] = (
        ufl.algorithms.estimate_total_polynomial_degree(L)
    )
    L = form(L, dtype=dtype)

    u_bc = Function(V, dtype=dtype)
    u_bc.interpolate(lambda x: x[1] ** degree)

    # Create Dirichlet boundary condition
    facetdim = mesh.topology.dim - 1
    mesh.topology.create_connectivity(facetdim, mesh.topology.dim)
    bndry_facets = exterior_facet_indices(mesh.topology)
    bdofs = locate_dofs_topological(V, facetdim, bndry_facets)
    bc = dirichletbc(u_bc, bdofs)

    b = assemble_vector(L)
    apply_lifting(b.array, [a], bcs=[[bc]])
    b.scatter_reverse(la.InsertMode.add)
    set_bc(b.array, [bc])

    a = form(a, dtype=dtype)
    A = assemble_matrix(a, bcs=[bc])
    A.scatter_reverse()

    uh = Function(V, dtype=dtype)
    cg_solver(mesh.comm, A, b, uh.x)
    uh.x.scatter_forward()

    M = (u_exact - uh) ** 2 * dx
    M = form(M, dtype=dtype)
    error = mesh.comm.allreduce(assemble_scalar(M), op=MPI.SUM)
    eps = np.sqrt(np.finfo(dtype).eps)
    assert np.isclose(error, 0.0, atol=eps)


def run_vector_test(mesh, V, degree, cg_solver, maxit=500, rtol=None):
    """Projection into H(div/curl) spaces."""
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = form(inner(u, v) * dx)

    # Source term
    x = SpatialCoordinate(mesh)
    u_exact = x[0] ** degree
    L = form(inner(u_exact, v[0]) * dx)

    b = assemble_vector(L)
    b.scatter_reverse(la.InsertMode.add)

    A = assemble_matrix(a)
    A.scatter_reverse()

    # Solve
    uh = Function(V)
    cg_solver(mesh.comm, A, b, uh.x, maxit=maxit, rtol=rtol)
    uh.x.scatter_forward()

    # Calculate error
    M = (u_exact - uh[0]) ** 2 * dx
    for i in range(1, mesh.topology.dim):
        M += uh[i] ** 2 * dx
    M = form(M)

    error = mesh.comm.allreduce(assemble_scalar(M), op=MPI.SUM)
    assert np.isclose(error, 0.0, atol=1e-07)


def run_dg_test(mesh, V, degree, cg_solver):
    """Manufactured Poisson problem, solving u = x[component]**n, where
    n is the degree of the Lagrange function space."""
    u, v = TrialFunction(V), TestFunction(V)

    # Exact solution
    x = SpatialCoordinate(mesh)
    u_exact = x[1] ** degree

    # Coefficient
    k = Function(V)
    k.x.array[:] = 2.0

    # Source term
    f = -div(k * grad(u_exact))

    # Mesh normals and element size
    n = FacetNormal(mesh)
    h = CellDiameter(mesh)
    h_avg = (h("+") + h("-")) / 2.0

    # Penalty parameter
    alpha = 32

    dx_ = dx(metadata={"quadrature_degree": -1})
    ds_ = ds(metadata={"quadrature_degree": -1})
    dS_ = dS(metadata={"quadrature_degree": -1})

    a = (
        inner(k * grad(u), grad(v)) * dx_
        - k("+") * inner(avg(grad(u)), jump(v, n)) * dS_
        - k("+") * inner(jump(u, n), avg(grad(v))) * dS_
        + k("+") * (alpha / h_avg) * inner(jump(u, n), jump(v, n)) * dS_
        - inner(k * grad(u), v * n) * ds_
        - inner(u * n, k * grad(v)) * ds_
        + (alpha / h) * inner(k * u, v) * ds_
    )
    L = (
        inner(f, v) * dx_
        - inner(k * u_exact * n, grad(v)) * ds_
        + (alpha / h) * inner(k * u_exact, v) * ds_
    )

    for integral in a.integrals():
        integral.metadata()["quadrature_degree"] = ufl.algorithms.estimate_total_polynomial_degree(
            a
        )
    for integral in L.integrals():
        integral.metadata()["quadrature_degree"] = ufl.algorithms.estimate_total_polynomial_degree(
            L
        )

    a, L = form(a), form(L)

    b = assemble_vector(L)
    b.scatter_reverse(la.InsertMode.add)

    A = assemble_matrix(a, [])
    A.scatter_reverse()

    # Solve
    uh = Function(V)
    cg_solver(mesh.comm, A, b, uh.x)
    uh.x.scatter_forward()

    # Calculate error
    M = (u_exact - uh) ** 2 * dx
    M = form(M)

    error = mesh.comm.allreduce(assemble_scalar(M), op=MPI.SUM)
    assert np.isclose(error, 0.0)


@pytest.mark.parametrize("family", ["N1curl", "N2curl"])
@pytest.mark.parametrize("order", [1])
def test_petsc_curl_curl_eigenvalue(family, order):
    """curl-curl eigenvalue problem.

    Solved using H(curl)-conforming finite element method.
    See https://www-users.cse.umn.edu/~arnold/papers/icm2002.pdf for details.
    """
    if not dolfinx.cpp.common.has_petsc:
        return

    petsc4py = pytest.importorskip("petsc4py")  # noqa: F841
    from petsc4py import PETSc

    from dolfinx.fem.petsc import assemble_matrix as petsc_assemble_matrix

    slepc4py = pytest.importorskip("slepc4py")  # noqa: F841
    from slepc4py import SLEPc

    mesh = create_rectangle(
        MPI.COMM_WORLD,
        [np.array([0.0, 0.0]), np.array([np.pi, np.pi])],
        [24, 24],
        CellType.triangle,
    )

    e = element(family, basix.CellType.triangle, order, dtype=default_real_type)
    V = functionspace(mesh, e)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = inner(ufl.curl(u), ufl.curl(v)) * dx
    b = inner(u, v) * dx

    boundary_facets = locate_entities_boundary(mesh, mesh.topology.dim - 1)
    boundary_dofs = locate_dofs_topological(V, mesh.topology.dim - 1, boundary_facets)

    zero_u = Function(V)
    zero_u.x.array[:] = 0
    bcs = [dirichletbc(zero_u, boundary_dofs)]

    a, b = form(a), form(b)
    A = petsc_assemble_matrix(a, bcs=bcs)
    A.assemble()
    B = petsc_assemble_matrix(b, bcs=bcs, diagonal=0.01)
    B.assemble()

    eps = SLEPc.EPS().create()
    eps.setOperators(A, B)
    PETSc.Options()["eps_type"] = "krylovschur"
    PETSc.Options()["eps_gen_hermitian"] = ""
    PETSc.Options()["eps_target_magnitude"] = ""
    PETSc.Options()["eps_target"] = 5.0
    PETSc.Options()["eps_view"] = ""
    PETSc.Options()["eps_nev"] = 12
    eps.setFromOptions()
    eps.solve()

    num_converged = eps.getConverged()
    evlas_unsorted = np.zeros(num_converged, dtype=np.complex128)

    for i in range(0, num_converged):
        evlas_unsorted[i] = eps.getEigenvalue(i)

    assert np.isclose(np.imag(evlas_unsorted), 0.0).all()
    evals_sorted = np.sort(np.real(evlas_unsorted))[:-1]
    evals_sorted = evals_sorted[np.logical_not(evals_sorted < 1e-8)]

    evals_exact = np.array([1.0, 1.0, 2.0, 4.0, 4.0, 5.0, 5.0, 8.0, 9.0])
    assert np.isclose(evals_sorted[0 : evals_exact.shape[0]], evals_exact, rtol=1e-2).all()

    eps.destroy()
    A.destroy()
    B.destroy()


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("family", ["HHJ", "Regge"])
def test_biharmonic(family, dtype):
    """Manufactured biharmonic problem.

    Solved using rotated Regge or the Hellan-Herrmann-Johnson (HHJ)
    mixed finite element method in two-dimensions.

    Runs in serial to use the SciPy sparse solvers (to avoid PETSc
    dependency).
    """
    import scipy

    xtype = np.real(dtype(0)).dtype
    mesh = create_rectangle(
        MPI.COMM_SELF,
        [np.array([0.0, 0.0]), np.array([1.0, 1.0])],
        [16, 16],
        CellType.triangle,
        dtype=xtype,
    )

    e = mixed_element(
        [
            element(family, basix.CellType.triangle, 1, dtype=dtype),
            element(basix.ElementFamily.P, basix.CellType.triangle, 2, dtype=dtype),
        ]
    )

    V = functionspace(mesh, e)
    sigma, u = ufl.TrialFunctions(V)
    tau, v = ufl.TestFunctions(V)

    x = ufl.SpatialCoordinate(mesh)
    u_exact = (
        ufl.sin(ufl.pi * x[0])
        * ufl.sin(ufl.pi * x[0])
        * ufl.sin(ufl.pi * x[1])
        * ufl.sin(ufl.pi * x[1])
    )
    f_exact = div(grad(div(grad(u_exact))))
    sigma_exact = grad(grad(u_exact))

    # sigma and tau are tangential-tangential continuous according to
    # the H(curl curl) continuity of the Regge space. However, for the
    # biharmonic problem we require normal-normal continuity H (div
    # div). Theorem 4.2 of Lizao Li's PhD thesis shows that the latter
    # space can be constructed by the former through the action of the
    # operator S:
    def S(tau):
        return tau - ufl.Identity(2) * ufl.tr(tau)

    if family == "Regge":
        # Apply S if we are working with Regge which is H(curl curl)
        sigma_S = S(sigma)
        tau_S = S(tau)
    elif family == "HHJ":
        # Don't apply S if we are working with HHJ which is already
        # H(div div)
        sigma_S = sigma
        tau_S = tau
    else:
        raise ValueError(f"Family {family} not supported.")

    # Discrete duality inner product eq. 4.5 Lizao Li's PhD thesis
    def b(tau_S, v):
        n = FacetNormal(mesh)
        return (
            inner(tau_S, grad(grad(v))) * dx
            - ufl.dot(ufl.dot(tau_S("+"), n("+")), n("+")) * jump(grad(v), n) * dS
            - ufl.dot(ufl.dot(tau_S, n), n) * ufl.dot(grad(v), n) * ds
        )

    # Non-symmetric formulation
    a = form(inner(sigma_S, tau_S) * dx - b(tau_S, u) + b(sigma_S, v), dtype=dtype)
    L = form(inner(f_exact, v) * dx, dtype=dtype)

    V_1 = V.sub(1).collapse()[0]
    zero_u = Function(V_1, dtype=dtype)
    zero_u.x.array[:] = 0

    # Strong (Dirichlet) boundary condition
    tdim = mesh.topology.dim
    boundary_facets = locate_entities_boundary(mesh, tdim - 1)
    boundary_dofs = locate_dofs_topological((V.sub(1), V_1), tdim - 1, boundary_facets)

    bcs = [dirichletbc(zero_u, boundary_dofs, V.sub(1))]

    A = assemble_matrix(a, bcs=bcs)
    A.scatter_reverse()
    b = assemble_vector(L)
    apply_lifting(b.array, [a], bcs=[bcs])
    b.scatter_reverse(la.InsertMode.add)
    set_bc(b.array, bcs)

    x_h = Function(V, dtype=dtype)
    x_h.x.array[:] = scipy.sparse.linalg.spsolve(A.to_scipy(), b.array)
    x_h.x.scatter_forward()

    # Recall that x_h has flattened indices
    u_error_numerator = np.sqrt(
        mesh.comm.allreduce(
            assemble_scalar(
                form(
                    inner(u_exact - x_h[4], u_exact - x_h[4])
                    * dx(mesh, metadata={"quadrature_degree": 6}),
                    dtype=dtype,
                )
            ),
            op=MPI.SUM,
        )
    )
    u_error_denominator = np.sqrt(
        mesh.comm.allreduce(
            assemble_scalar(
                form(
                    inner(u_exact, u_exact) * dx(mesh, metadata={"quadrature_degree": 6}),
                    dtype=dtype,
                )
            ),
            op=MPI.SUM,
        )
    )
    assert np.abs(u_error_numerator / u_error_denominator) < 0.05

    # Reconstruct tensor from flattened indices.
    # Apply inverse transform. In 2D we have S^{-1} = S.
    if family == "Regge":
        sigma_h = S(ufl.as_tensor([[x_h[0], x_h[1]], [x_h[2], x_h[3]]]))
    elif family == "HHJ":
        sigma_h = ufl.as_tensor([[x_h[0], x_h[1]], [x_h[2], x_h[3]]])
    else:
        raise ValueError(f"Family {family} not supported.")

    sigma_error_numerator = np.sqrt(
        mesh.comm.allreduce(
            assemble_scalar(
                form(
                    inner(sigma_exact - sigma_h, sigma_exact - sigma_h)
                    * dx(mesh, metadata={"quadrature_degree": 6}),
                    dtype=dtype,
                )
            ),
            op=MPI.SUM,
        )
    )
    sigma_error_denominator = np.sqrt(
        mesh.comm.allreduce(
            assemble_scalar(
                form(
                    inner(sigma_exact, sigma_exact) * dx(mesh, metadata={"quadrature_degree": 6}),
                    dtype=dtype,
                )
            ),
            op=MPI.SUM,
        )
    )
    assert np.abs(sigma_error_numerator / sigma_error_denominator) < 0.05


def get_mesh(cell_type, datadir):
    # In parallel, use larger meshes
    if cell_type == CellType.triangle:
        filename = "create_unit_square_triangle.xdmf"
    elif cell_type == CellType.quadrilateral:
        filename = "create_unit_square_quad.xdmf"
    elif cell_type == CellType.tetrahedron:
        filename = "create_unit_cube_tetra.xdmf"
    elif cell_type == CellType.hexahedron:
        filename = "create_unit_cube_hexahedron.xdmf"
    with XDMFFile(
        MPI.COMM_WORLD, Path(datadir, filename), "r", encoding=XDMFFile.Encoding.ASCII
    ) as xdmf:
        return xdmf.read_mesh(name="Grid")


parametrize_cell_types = pytest.mark.parametrize(
    "cell_type",
    [CellType.triangle, CellType.quadrilateral, CellType.tetrahedron, CellType.hexahedron],
)
parametrize_cell_types_simplex = pytest.mark.parametrize(
    "cell_type", [CellType.triangle, CellType.tetrahedron]
)
parametrize_cell_types_tp = pytest.mark.parametrize(
    "cell_type", [CellType.quadrilateral, CellType.hexahedron]
)
parametrize_cell_types_quad = pytest.mark.parametrize("cell_type", [CellType.quadrilateral])
parametrize_cell_types_hex = pytest.mark.parametrize("cell_type", [CellType.hexahedron])


# Run tests on all spaces in periodic table on triangles and tetrahedra
@pytest.mark.skipif(default_real_type != np.float64, reason="float32 not supported yet")
@parametrize_cell_types_simplex
@pytest.mark.parametrize("family", ["Lagrange"])
@pytest.mark.parametrize("degree", [2, 3, 4])
def test_P_simplex(family, degree, cell_type, datadir, cg_solver):
    if cell_type == CellType.tetrahedron and degree == 4:
        pytest.skip("Skip expensive test on tetrahedron")
    mesh = get_mesh(cell_type, datadir)
    V = functionspace(mesh, (family, degree))
    run_scalar_test(mesh, V, degree, cg_solver)


@parametrize_cell_types_simplex
@pytest.mark.parametrize("family", ["Lagrange"])
@pytest.mark.parametrize("degree", [2, 3, 4])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_P_simplex_built_in(family, degree, dtype, cell_type, datadir, cg_solver):
    if cell_type == CellType.tetrahedron:
        mesh = create_unit_cube(MPI.COMM_WORLD, 5, 5, 5, dtype=dtype)
    elif cell_type == CellType.triangle:
        mesh = create_unit_square(MPI.COMM_WORLD, 5, 5, dtype=dtype)
    V = functionspace(mesh, (family, degree))
    run_scalar_test(mesh, V, degree, cg_solver)


@pytest.mark.skipif(default_real_type != np.float64, reason="float32 not supported yet")
@parametrize_cell_types_simplex
@pytest.mark.parametrize("family", ["Lagrange"])
@pytest.mark.parametrize("degree", [2, 3, 4])
def test_vector_P_simplex(family, degree, cell_type, datadir, cg_solver):
    if cell_type == CellType.tetrahedron and degree == 4:
        pytest.skip("Skip expensive test on tetrahedron")
    mesh = get_mesh(cell_type, datadir)
    gdim = mesh.geometry.dim
    V = functionspace(mesh, (family, degree, (gdim,)))
    run_vector_test(mesh, V, degree, cg_solver)


@pytest.mark.skipif(default_real_type != np.float64, reason="float32 not supported yet")
@parametrize_cell_types_simplex
@pytest.mark.parametrize("family", ["DG"])
@pytest.mark.parametrize("degree", [2, 3])
def test_dP_simplex(family, degree, cell_type, datadir, cg_solver):
    mesh = get_mesh(cell_type, datadir)
    V = functionspace(mesh, (family, degree))
    run_dg_test(mesh, V, degree, cg_solver)


@pytest.mark.skipif(default_real_type != np.float64, reason="float32 not supported yet")
@parametrize_cell_types_simplex
@pytest.mark.parametrize("family", ["RT", "N1curl"])
@pytest.mark.parametrize("degree", [1, 2, 3, 4])
def test_RT_N1curl_simplex(family, degree, cell_type, datadir, cg_solver):
    if cell_type == CellType.tetrahedron and degree == 4:
        pytest.skip("Skip expensive test on tetrahedron")
    mesh = get_mesh(cell_type, datadir)
    V = functionspace(mesh, (family, degree))
    run_vector_test(mesh, V, degree - 1, cg_solver)


@pytest.mark.skipif(default_real_type != np.float64, reason="float32 not supported yet")
@parametrize_cell_types_simplex
@pytest.mark.parametrize("family", ["Discontinuous Raviart-Thomas"])
@pytest.mark.parametrize("degree", [1, 2, 3, 4])
def test_discontinuous_RT(family, degree, cell_type, datadir, cg_solver):
    if cell_type == CellType.tetrahedron and degree == 4:
        pytest.skip("Skip expensive test on tetrahedron")
    mesh = get_mesh(cell_type, datadir)
    V = functionspace(mesh, (family, degree))
    run_vector_test(mesh, V, degree - 1, cg_solver)


@pytest.mark.skipif(default_real_type != np.float64, reason="float32 not supported yet")
@parametrize_cell_types_simplex
@pytest.mark.parametrize("family", ["BDM", "N2curl"])
@pytest.mark.parametrize("degree", [1, 2])
def test_BDM_N2curl_simplex(family, degree, cell_type, datadir, cg_solver):
    mesh = get_mesh(cell_type, datadir)
    V = functionspace(mesh, (family, degree))
    run_vector_test(mesh, V, degree, cg_solver)


# Skip slowest test in complex to stop CI timing out
# @skip_if_complex
@pytest.mark.skipif(default_real_type != np.float64, reason="float32 not supported yet")
@parametrize_cell_types_simplex
@pytest.mark.parametrize("family", ["BDM", "N2curl"])
@pytest.mark.parametrize("degree", [3])
def test_BDM_N2curl_simplex_highest_order(family, degree, cell_type, datadir, cg_solver):
    mesh = get_mesh(cell_type, datadir)
    V = functionspace(mesh, (family, degree))
    run_vector_test(mesh, V, degree, cg_solver, maxit=900, rtol=1e-5)


# Run tests on all spaces in periodic table on quadrilaterals and
# hexahedra
@pytest.mark.skipif(default_real_type != np.float64, reason="float32 not supported yet")
@parametrize_cell_types_tp
@pytest.mark.parametrize("family", ["Q"])
@pytest.mark.parametrize("degree", [2, 3, 4])
def test_P_tp(family, degree, cell_type, datadir, cg_solver):
    mesh = get_mesh(cell_type, datadir)
    V = functionspace(mesh, (family, degree))
    run_scalar_test(mesh, V, degree, cg_solver)


@pytest.mark.skipif(default_real_type != np.float64, reason="float32 not supported yet")
@parametrize_cell_types_tp
@pytest.mark.parametrize("family", ["Q"])
@pytest.mark.parametrize("degree", [2, 3, 4])
def test_P_tp_built_in_mesh(family, degree, cell_type, datadir, cg_solver):
    if cell_type == CellType.hexahedron:
        mesh = create_unit_cube(MPI.COMM_WORLD, 5, 5, 5, cell_type)
    elif cell_type == CellType.quadrilateral:
        mesh = create_unit_square(MPI.COMM_WORLD, 5, 5, cell_type)
    mesh = get_mesh(cell_type, datadir)
    V = functionspace(mesh, (family, degree))
    run_scalar_test(mesh, V, degree, cg_solver)


@pytest.mark.skipif(default_real_type != np.float64, reason="float32 not supported yet")
@parametrize_cell_types_tp
@pytest.mark.parametrize("family", ["Q"])
@pytest.mark.parametrize("degree", [2, 3, 4])
def test_vector_P_tp(family, degree, cell_type, datadir, cg_solver):
    if cell_type == CellType.hexahedron and degree == 4:
        pytest.skip("Skip expensive test on hexahedron")
    mesh = get_mesh(cell_type, datadir)
    gdim = mesh.geometry.dim
    V = functionspace(mesh, (family, degree, (gdim,)))
    run_vector_test(mesh, V, degree, cg_solver)


@pytest.mark.skipif(default_real_type != np.float64, reason="float32 not supported yet")
@parametrize_cell_types_quad
@pytest.mark.parametrize("family", ["DQ"])
@pytest.mark.parametrize("degree", [1, 2, 3])
def test_dP_quad(family, degree, cell_type, datadir, cg_solver):
    mesh = get_mesh(cell_type, datadir)
    V = functionspace(mesh, (family, degree))
    run_dg_test(mesh, V, degree, cg_solver)


@pytest.mark.skipif(default_real_type != np.float64, reason="float32 not supported yet")
@parametrize_cell_types_hex
@pytest.mark.parametrize("family", ["DQ"])
@pytest.mark.parametrize("degree", [1, 2])
def test_dP_hex(family, degree, cell_type, datadir, cg_solver):
    mesh = get_mesh(cell_type, datadir)
    V = functionspace(mesh, (family, degree))
    run_dg_test(mesh, V, degree, cg_solver)


@pytest.mark.skipif(default_real_type != np.float64, reason="float32 not supported yet")
@parametrize_cell_types_tp
@pytest.mark.parametrize("family", ["S"])
@pytest.mark.parametrize("degree", [2, 3, 4])
def test_S_tp(family, degree, cell_type, datadir, cg_solver):
    mesh = get_mesh(cell_type, datadir)
    V = functionspace(mesh, (family, degree))
    run_scalar_test(mesh, V, degree // 2, cg_solver)


@pytest.mark.skipif(default_real_type != np.float64, reason="float32 not supported yet")
@parametrize_cell_types_tp
@pytest.mark.parametrize("family", ["S"])
@pytest.mark.parametrize("degree", [2, 3, 4])
def test_S_tp_built_in_mesh(family, degree, cell_type, datadir, cg_solver):
    if cell_type == CellType.hexahedron:
        mesh = create_unit_cube(MPI.COMM_WORLD, 5, 5, 5, cell_type)
    elif cell_type == CellType.quadrilateral:
        mesh = create_unit_square(MPI.COMM_WORLD, 5, 5, cell_type)
    mesh = get_mesh(cell_type, datadir)
    V = functionspace(mesh, (family, degree))
    run_scalar_test(mesh, V, degree // 2, cg_solver)


@pytest.mark.skipif(default_real_type != np.float64, reason="float32 not supported yet")
@parametrize_cell_types_tp
@pytest.mark.parametrize("family", ["S"])
@pytest.mark.parametrize("degree", [2, 3, 4])
def test_vector_S_tp(family, degree, cell_type, datadir, cg_solver):
    if cell_type == CellType.hexahedron and degree == 4:
        pytest.skip("Skip expensive test on hexahedron")
    mesh = get_mesh(cell_type, datadir)
    gdim = mesh.geometry.dim
    V = functionspace(mesh, (family, degree, (gdim,)))
    run_vector_test(mesh, V, degree // 2, cg_solver)


@pytest.mark.skipif(default_real_type != np.float64, reason="float32 not supported yet")
@parametrize_cell_types_quad
@pytest.mark.parametrize("family", ["DPC"])
@pytest.mark.parametrize("degree", [2, 3, 4])
def test_DPC_quad(family, degree, cell_type, datadir, cg_solver):
    mesh = get_mesh(cell_type, datadir)
    V = functionspace(mesh, (family, degree))
    run_dg_test(mesh, V, degree // 2, cg_solver)


@pytest.mark.skipif(default_real_type != np.float64, reason="float32 not supported yet")
@parametrize_cell_types_hex
@pytest.mark.parametrize("family", ["DPC"])
@pytest.mark.parametrize("degree", [2])
def test_DPC_hex(family, degree, cell_type, datadir, cg_solver):
    mesh = get_mesh(cell_type, datadir)
    V = functionspace(mesh, (family, degree))
    run_dg_test(mesh, V, degree // 2, cg_solver)


@pytest.mark.skipif(default_real_type != np.float64, reason="float32 not supported yet")
@parametrize_cell_types_quad
@pytest.mark.parametrize("family", ["RTCE", "RTCF"])
@pytest.mark.parametrize("degree", [1, 2, 3])
def test_RTC_quad(family, degree, cell_type, datadir, cg_solver):
    mesh = get_mesh(cell_type, datadir)
    V = functionspace(mesh, (family, degree))
    run_vector_test(mesh, V, degree - 1, cg_solver)


@pytest.mark.skipif(default_real_type != np.float64, reason="float32 not supported yet")
@parametrize_cell_types_hex
@pytest.mark.parametrize("family", ["NCE", "NCF"])
@pytest.mark.parametrize("degree", [1, 2, 3])
def test_NC_hex(family, degree, cell_type, datadir, cg_solver):
    mesh = get_mesh(cell_type, datadir)
    V = functionspace(mesh, (family, degree))
    run_vector_test(mesh, V, degree - 1, cg_solver, maxit=700, rtol=1e-4)


@pytest.mark.skipif(default_real_type != np.float64, reason="float32 not supported yet")
@parametrize_cell_types_quad
@pytest.mark.parametrize("family", ["BDMCE", "BDMCF"])
@pytest.mark.parametrize("degree", [1, 2, 3, 4])
def test_BDM_quad(family, degree, cell_type, datadir, cg_solver):
    mesh = get_mesh(cell_type, datadir)
    V = functionspace(mesh, (family, degree))
    run_vector_test(mesh, V, (degree - 1) // 2, cg_solver)


@pytest.mark.skipif(default_real_type != np.float64, reason="float32 not supported yet")
@parametrize_cell_types_hex
@pytest.mark.parametrize("family", ["AAE", "AAF"])
@pytest.mark.parametrize("degree", [1, 2, 3])
def test_AA_hex(family, degree, cell_type, datadir, cg_solver):
    mesh = get_mesh(cell_type, datadir)
    V = functionspace(mesh, (family, degree))
    run_vector_test(mesh, V, (degree - 1) // 2, cg_solver)
