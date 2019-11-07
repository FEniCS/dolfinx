import pytest

from dolfin import (UnitIntervalMesh, UnitSquareMesh, UnitCubeMesh, Function, FunctionSpace,
                    TrialFunction, TestFunction, DirichletBC, MPI, solve)
from dolfin.cpp.mesh import CellType
from petsc4py import PETSc

from ufl import inner, grad, dx, SpatialCoordinate
import numpy as np
import sympy as sp


def ManufacturedSolution(degree, gdim):
    """
    Generate manufactured solution as function of x,y,z and the corresponding Laplacian
    """
    xvec = sp.symbols("x[0] x[1] x[2]")
    x, y, z = xvec
    u = gdim**3 * (x)**(degree - 1) * (1 - x)

    if gdim > 1:
        u *= (y)**(degree - 1) * (1 - y)
    if gdim > 2:
        u *= (z)**(degree - 1) * (1 - z)

    x_ = sp.symbols("x y z")
    u_lambda = u
    for i in range(gdim):
        u_lambda = u_lambda.subs(xvec[i], x_[i])
    u_lambda = sp.lambdify((x_), u_lambda, 'numpy')

    laplace_u = 0
    for i in range(gdim):
        laplace_u += sp.diff(u, xvec[i], xvec[i])
    return u_lambda, laplace_u


def boundary(x):
    """
    gdim-dimensional boundary marker
    """
    condition = np.logical_or(x[:, 0] < 1.0e-6, x[:, 0] > 1.0 - 1.0e-6)
    for dim in range(1, x.shape[1]):
        c_dim = np.logical_or(x[:, dim] < 1.0e-6, x[:, dim] > 1.0 - 1.0e-6)
        condition = np.logical_or(condition, c_dim)
    return condition


# (CellType.tetrahedron, 3),
@pytest.mark.parametrize("degree", [2, 3, 4])
@pytest.mark.parametrize("cell_type, gdim", [(CellType.interval, 1), (CellType.triangle, 2),
                                             (CellType.quadrilateral, 2),
                                             (CellType.hexahedron, 3)])
def test_manufactured_poisson(degree, cell_type, gdim):
    """
    Manufactured Poisson problem, solving u = Pi_{i=0}^gdim (1 - x[i]) * x[i]^(p - 1)
    where p is the degree of the Lagrange function space. Solved on the
    gdim-dimensional UnitMesh, with homogeneous Dirichlet boundary conditions.

    Comparing values at each dof with the analytical solution.

    """
    if gdim == 1:
        mesh = UnitIntervalMesh(MPI.comm_world, 5)
    if gdim == 2:
        mesh = UnitSquareMesh(MPI.comm_world, 10, 10, cell_type)
    elif gdim == 3:
        mesh = UnitCubeMesh(MPI.comm_world, 5, 5, 5, cell_type)
    V = FunctionSpace(mesh, ("CG", degree))
    u, v = TrialFunction(V), TestFunction(V)
    x = SpatialCoordinate(mesh)

    u_exact, laplace_u = ManufacturedSolution(degree, mesh.geometric_dimension())

    f = -eval(str(laplace_u))
    a = inner(grad(u), grad(v)) * dx
    lhs = inner(f, v) * dx(metadata={'quadrature_degree': degree + 3})

    u_bc = Function(V)
    with u_bc.vector.localForm() as u_local:
        u_local.set(0.0)
    bc = DirichletBC(V, u_bc, boundary)

    uh = Function(V)
    solve(a == lhs, uh, bc)
    uh.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    x = V.tabulate_dof_coordinates()
    x_ = np.zeros((x.shape[0], 3))

    # Fill with coordinates to work with eval
    if gdim == 1:
        x_[:, :1] = x
    elif gdim == 2:
        x_[:, :2] = x
    else:
        x_ = x

    cell = 4

    # Find points of dofs in i-th cell
    dofs_i = V.dofmap.cell_dofs(cell)
    x_i = np.array([x_[dof, :] for dof in dofs_i])

    # Function values for dofs at i-th cell
    result_i = uh.eval(x_i, cell * np.ones(len(x_i)))
    # Compute exact solution
    if mesh.geometric_dimension() == 1:
        exact_i = u_exact(x_i[:, 0], 0, 0)
    elif mesh.geometric_dimension() == 2:
        exact_i = u_exact(x_i[:, 0], x_i[:, 1], 0)
    elif mesh.geometric_dimension() == 3:
        exact_i = u_exact(x_i[:, 0], x_i[:, 1], x_i[:, 2])
    # Reshape
    result_i = result_i.reshape(exact_i.shape)
    result_i[np.abs(result_i) < 1e-5] = 0
    exact_i[np.abs(exact_i) < 1e-5] = 0

    assert result_i == pytest.approx(exact_i, rel=1e-2)
