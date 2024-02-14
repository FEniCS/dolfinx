from mpi4py import MPI
from petsc4py import PETSc

import numpy as np
import pytest

import basix
import basix.ufl
import ufl
from dolfinx.fem import (
    Function,
    assemble_scalar,
    dirichletbc,
    form,
    functionspace,
    locate_dofs_topological,
)
from dolfinx.fem.petsc import apply_lifting, assemble_matrix, assemble_vector, set_bc
from dolfinx.mesh import CellType, create_unit_cube, create_unit_square, exterior_facet_indices
from ufl import SpatialCoordinate, TestFunction, TrialFunction, div, dx, grad, inner


def run_scalar_test(V, degree):
    mesh = V.mesh
    u, v = TrialFunction(V), TestFunction(V)
    a = inner(grad(u), grad(v)) * dx

    # Get quadrature degree for bilinear form integrand (ignores effect of non-affine map)
    a = inner(grad(u), grad(v)) * dx(metadata={"quadrature_degree": -1})
    a.integrals()[0].metadata()[
        "quadrature_degree"
    ] = ufl.algorithms.estimate_total_polynomial_degree(a)
    a = form(a)

    # Source term
    x = SpatialCoordinate(mesh)
    u_exact = x[1] ** degree
    f = -div(grad(u_exact))

    # Set quadrature degree for linear form integrand (ignores effect of non-affine map)
    L = inner(f, v) * dx(metadata={"quadrature_degree": -1})
    L.integrals()[0].metadata()[
        "quadrature_degree"
    ] = ufl.algorithms.estimate_total_polynomial_degree(L)
    L = form(L)

    u_bc = Function(V)
    u_bc.interpolate(lambda x: x[1] ** degree)

    # Create Dirichlet boundary condition
    facetdim = mesh.topology.dim - 1
    mesh.topology.create_connectivity(facetdim, mesh.topology.dim)
    bndry_facets = exterior_facet_indices(mesh.topology)
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

    M = (u_exact - uh) ** 2 * dx
    M = form(M)
    error = mesh.comm.allreduce(assemble_scalar(M), op=MPI.SUM)
    assert np.abs(error) < 1.0e-6

    solver.destroy()
    A.destroy()
    b.destroy()


@pytest.mark.parametrize("degree", range(1, 6))
def test_basix_element_wrapper(degree):
    ufl_element = basix.ufl.element(
        basix.ElementFamily.P, basix.CellType.triangle, degree, basix.LagrangeVariant.gll_isaac
    )
    mesh = create_unit_square(MPI.COMM_WORLD, 10, 10)
    V = functionspace(mesh, ufl_element)
    run_scalar_test(V, degree)


def test_custom_element_triangle_degree1():
    wcoeffs = np.eye(3)
    z = np.zeros((0, 2))
    x = [
        [np.array([[0.0, 0.0]]), np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]])],
        [z, z, z],
        [z],
        [],
    ]
    z = np.zeros((0, 1, 0, 1))
    M = [[np.array([[[[1.0]]]]), np.array([[[[1.0]]]]), np.array([[[[1.0]]]])], [z, z, z], [z], []]
    ufl_element = basix.ufl.custom_element(
        basix.CellType.triangle,
        [],
        wcoeffs,
        x,
        M,
        0,
        basix.MapType.identity,
        basix.SobolevSpace.H1,
        False,
        1,
        1,
    )
    mesh = create_unit_square(MPI.COMM_WORLD, 10, 10)
    V = functionspace(mesh, ufl_element)
    run_scalar_test(V, 1)


def test_custom_element_triangle_degree4():
    wcoeffs = np.eye(15)
    x = [
        [np.array([[0.0, 0.0]]), np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]])],
        [
            np.array([[0.75, 0.25], [0.5, 0.5], [0.25, 0.75]]),
            np.array([[0.0, 0.25], [0.0, 0.5], [0.0, 0.75]]),
            np.array([[0.25, 0.0], [0.5, 0.0], [0.75, 0.0]]),
        ],
        [np.array([[0.25, 0.25], [0.5, 0.25], [0.25, 0.5]])],
        [],
    ]
    id = np.array([[[[1.0], [0.0], [0.0]]], [[[0.0], [1.0], [0.0]]], [[[0.0], [0.0], [1.0]]]])
    M = [
        [np.array([[[[1.0]]]]), np.array([[[[1.0]]]]), np.array([[[[1.0]]]])],
        [id, id, id],
        [id],
        [],
    ]

    ufl_element = basix.ufl.custom_element(
        basix.CellType.triangle,
        [],
        wcoeffs,
        x,
        M,
        0,
        basix.MapType.identity,
        basix.SobolevSpace.H1,
        False,
        4,
        4,
    )
    mesh = create_unit_square(MPI.COMM_WORLD, 10, 10)
    V = functionspace(mesh, ufl_element)
    run_scalar_test(V, 4)


def test_custom_element_triangle_degree4_integral():
    pts, wts = basix.make_quadrature(basix.CellType.interval, 10)
    tab = basix.create_element(basix.ElementFamily.P, basix.CellType.interval, 2).tabulate(0, pts)[
        0, :, :, 0
    ]
    wcoeffs = np.eye(15)
    x = [
        [np.array([[0.0, 0.0]]), np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]])],
        [
            np.array([[1.0 - p[0], p[0]] for p in pts]),
            np.array([[0.0, p[0]] for p in pts]),
            np.array([[p[0], 0.0] for p in pts]),
        ],
        [np.array([[0.25, 0.25], [0.5, 0.25], [0.25, 0.5]])],
        [],
    ]

    assert pts.shape[0] != 3
    quadrature_mat = np.zeros([3, 1, pts.shape[0], 1])
    for dof in range(3):
        for p in range(pts.shape[0]):
            quadrature_mat[dof, 0, p, 0] = wts[p] * tab[p, dof]

    M = [
        [np.array([[[[1.0]]]]), np.array([[[[1.0]]]]), np.array([[[[1.0]]]])],
        [quadrature_mat, quadrature_mat, quadrature_mat],
        [np.array([[[[1.0], [0.0], [0.0]]], [[[0.0], [1.0], [0.0]]], [[[0.0], [0.0], [1.0]]]])],
        [],
    ]

    ufl_element = basix.ufl.custom_element(
        basix.CellType.triangle,
        [],
        wcoeffs,
        x,
        M,
        0,
        basix.MapType.identity,
        basix.SobolevSpace.H1,
        False,
        4,
        4,
    )
    mesh = create_unit_square(MPI.COMM_WORLD, 10, 10)
    V = functionspace(mesh, ufl_element)
    run_scalar_test(V, 4)


def test_custom_element_quadrilateral_degree1():
    wcoeffs = np.eye(4)
    z = np.zeros((0, 2))
    x = [
        [
            np.array([[0.0, 0.0]]),
            np.array([[1.0, 0.0]]),
            np.array([[0.0, 1.0]]),
            np.array([[1.0, 1.0]]),
        ],
        [z, z, z, z],
        [z],
        [],
    ]
    z = np.zeros((0, 1, 0, 1))
    M = [
        [
            np.array([[[[1.0]]]]),
            np.array([[[[1.0]]]]),
            np.array([[[[1.0]]]]),
            np.array([[[[1.0]]]]),
        ],
        [z, z, z, z],
        [z],
        [],
    ]
    ufl_element = basix.ufl.custom_element(
        basix.CellType.quadrilateral,
        [],
        wcoeffs,
        x,
        M,
        0,
        basix.MapType.identity,
        basix.SobolevSpace.H1,
        False,
        1,
        1,
    )
    mesh = create_unit_square(MPI.COMM_WORLD, 10, 10, CellType.quadrilateral)
    V = functionspace(mesh, ufl_element)
    run_scalar_test(V, 1)


@pytest.mark.parametrize(
    "cell_type",
    [CellType.triangle, CellType.quadrilateral, CellType.tetrahedron, CellType.hexahedron],
)
@pytest.mark.parametrize(
    "element_family",
    [
        basix.ElementFamily.N1E,
        basix.ElementFamily.N2E,
        basix.ElementFamily.RT,
        basix.ElementFamily.BDM,
    ],
)
def test_vector_copy_degree1(cell_type, element_family):
    if cell_type in [CellType.triangle, CellType.quadrilateral]:
        tdim = 2
        mesh = create_unit_square(MPI.COMM_WORLD, 10, 10, cell_type)
    else:
        tdim = 3
        mesh = create_unit_cube(MPI.COMM_WORLD, 5, 5, 5, cell_type)

    def func(x):
        return x[:tdim]

    e1 = basix.ufl.element(element_family, getattr(basix.CellType, cell_type.name), 1)

    e2 = basix.ufl.custom_element(
        e1._element.cell_type,
        e1._element.value_shape,
        e1._element.wcoeffs,
        e1._element.x,
        e1._element.M,
        0,
        e1._element.map_type,
        e1._element.sobolev_space,
        e1._element.discontinuous,
        e1._element.embedded_subdegree,
        e1._element.embedded_superdegree,
    )

    space1 = functionspace(mesh, e1)
    space2 = functionspace(mesh, e2)

    f1 = Function(space1)
    f2 = Function(space2)
    f1.interpolate(func)
    f2.interpolate(func)

    diff = f1 - f2
    error = assemble_scalar(form(ufl.inner(diff, diff) * ufl.dx))
    assert np.isclose(error, 0)


@pytest.mark.parametrize(
    "cell_type",
    [CellType.triangle, CellType.quadrilateral, CellType.tetrahedron, CellType.hexahedron],
)
@pytest.mark.parametrize("element_family", [basix.ElementFamily.P, basix.ElementFamily.serendipity])
def test_scalar_copy_degree1(cell_type, element_family):
    if element_family == basix.ElementFamily.serendipity and cell_type in [
        CellType.triangle,
        CellType.tetrahedron,
    ]:
        pytest.xfail("Serendipity elements cannot be created on simplices")

    if cell_type in [CellType.triangle, CellType.quadrilateral]:
        mesh = create_unit_square(MPI.COMM_WORLD, 10, 10, cell_type)
    else:
        mesh = create_unit_cube(MPI.COMM_WORLD, 5, 5, 5, cell_type)

    def func(x):
        return x[0]

    e1 = basix.ufl.element(element_family, getattr(basix.CellType, cell_type.name), 1)
    e2 = basix.ufl.custom_element(
        e1._element.cell_type,
        e1._element.value_shape,
        e1._element.wcoeffs,
        e1._element.x,
        e1._element.M,
        0,
        e1._element.map_type,
        e1._element.sobolev_space,
        e1._element.discontinuous,
        e1._element.embedded_subdegree,
        e1._element.embedded_superdegree,
    )

    space1 = functionspace(mesh, e1)
    space2 = functionspace(mesh, e2)

    f1 = Function(space1)
    f2 = Function(space2)
    f1.interpolate(func)
    f2.interpolate(func)

    diff = f1 - f2
    error = assemble_scalar(form(ufl.inner(diff, diff) * ufl.dx))
    assert np.isclose(error, 0)
