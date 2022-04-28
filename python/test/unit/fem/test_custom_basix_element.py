import basix
from basix.ufl_wrapper import BasixElement

import numpy as np
import pytest

import ufl
from dolfinx.fem import (Function, FunctionSpace,
                         assemble_scalar, dirichletbc, form,
                         locate_dofs_topological)
from dolfinx.fem.petsc import (apply_lifting, assemble_matrix, assemble_vector,
                               set_bc)
from dolfinx.mesh import CellType, compute_boundary_facets, create_unit_square
from ufl import SpatialCoordinate, TestFunction, TrialFunction, div, dx, grad, inner

from mpi4py import MPI
from petsc4py import PETSc


def run_scalar_test(V, degree):
    mesh = V.mesh
    u, v = TrialFunction(V), TestFunction(V)
    a = inner(grad(u), grad(v)) * dx

    # Get quadrature degree for bilinear form integrand (ignores effect of non-affine map)
    a = inner(grad(u), grad(v)) * dx(metadata={"quadrature_degree": -1})
    a.integrals()[0].metadata()["quadrature_degree"] = ufl.algorithms.estimate_total_polynomial_degree(a)
    a = form(a)

    # Source term
    x = SpatialCoordinate(mesh)
    u_exact = x[1]**degree
    f = - div(grad(u_exact))

    # Set quadrature degree for linear form integrand (ignores effect of non-affine map)
    L = inner(f, v) * dx(metadata={"quadrature_degree": -1})
    L.integrals()[0].metadata()["quadrature_degree"] = ufl.algorithms.estimate_total_polynomial_degree(L)
    L = form(L)

    u_bc = Function(V)
    u_bc.interpolate(lambda x: x[1]**degree)

    # Create Dirichlet boundary condition
    facetdim = mesh.topology.dim - 1
    mesh.topology.create_connectivity(facetdim, mesh.topology.dim)
    bndry_facets = np.where(np.array(compute_boundary_facets(mesh.topology)) == 1)[0]
    bdofs = locate_dofs_topological(V, facetdim, bndry_facets)
    bc = dirichletbc(u_bc, bdofs)

    b = assemble_vector(L)
    apply_lifting(b, [a], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, [bc])

    a = form(a)
    A = assemble_matrix(a, bcs=[bc])
    A.assemble()

    # Create LU linear solver
    solver = PETSc.KSP().create(MPI.COMM_WORLD)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)
    solver.setOperators(A)

    uh = Function(V)
    solver.solve(b, uh.vector)
    uh.x.scatter_forward()

    M = (u_exact - uh)**2 * dx
    M = form(M)
    error = mesh.comm.allreduce(assemble_scalar(M), op=MPI.SUM)

    assert np.absolute(error) < 1.0e-14


@pytest.mark.parametrize("degree", range(1, 6))
def test_basix_element_wrapper(degree):
    e = basix.create_element(basix.ElementFamily.P, basix.CellType.triangle, degree, basix.LagrangeVariant.gll_isaac)
    ufl_element = BasixElement(e)

    mesh = create_unit_square(MPI.COMM_WORLD, 10, 10)
    V = FunctionSpace(mesh, ufl_element)

    run_scalar_test(V, degree)


def test_custom_element_triangle_degree1():
    wcoeffs = np.eye(3)
    z = np.zeros((0, 2))
    x = [[np.array([[0., 0.]]), np.array([[1., 0.]]), np.array([[0., 1.]])],
         [z, z, z], [z], []]
    z = np.zeros((0, 1, 0))
    M = [[np.array([[[1.]]]), np.array([[[1.]]]), np.array([[[1.]]])],
         [z, z, z], [z], []]

    e = basix.create_custom_element(
        basix.CellType.triangle, [], wcoeffs,
        x, M, basix.MapType.identity, False, 1, 1)
    ufl_element = BasixElement(e)

    mesh = create_unit_square(MPI.COMM_WORLD, 10, 10)
    V = FunctionSpace(mesh, ufl_element)

    run_scalar_test(V, 1)


def test_custom_element_triangle_degree4():
    wcoeffs = np.eye(15)
    x = [[np.array([[0., 0.]]), np.array([[1., 0.]]), np.array([[0., 1.]])],
         [np.array([[.75, .25], [.5, .5], [.25, .75]]), np.array([[0., .25], [0., .5], [0., .75]]),
          np.array([[.25, 0.], [.5, 0.], [.75, 0.]])],
         [np.array([[.25, .25], [.5, .25], [.25, .5]])], []]
    id = np.array([[[1., 0., 0.]], [[0., 1., 0.]], [[0., 0., 1.]]])
    M = [[np.array([[[1.]]]), np.array([[[1.]]]), np.array([[[1.]]])],
         [id, id, id], [id], []]

    e = basix.create_custom_element(
        basix.CellType.triangle, [], wcoeffs,
        x, M, basix.MapType.identity, False, 4, 4)
    ufl_element = BasixElement(e)

    mesh = create_unit_square(MPI.COMM_WORLD, 10, 10)
    V = FunctionSpace(mesh, ufl_element)

    run_scalar_test(V, 4)


def test_custom_element_quadrilateral_degree1():
    wcoeffs = np.eye(4)
    z = np.zeros((0, 2))
    x = [[np.array([[0., 0.]]), np.array([[1., 0.]]), np.array([[0., 1.]]), np.array([[1., 1.]])],
         [z, z, z, z], [z], []]
    z = np.zeros((0, 1, 0))
    M = [[np.array([[[1.]]]), np.array([[[1.]]]), np.array([[[1.]]]), np.array([[[1.]]])],
         [z, z, z, z], [z], []]

    e = basix.create_custom_element(
        basix.CellType.quadrilateral, [], wcoeffs,
        x, M, basix.MapType.identity, False, 1, 1)
    ufl_element = BasixElement(e)

    mesh = create_unit_square(MPI.COMM_WORLD, 10, 10, CellType.quadrilateral)
    V = FunctionSpace(mesh, ufl_element)

    run_scalar_test(V, 1)
