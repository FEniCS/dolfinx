import pytest

from dolfin import (UnitIntervalMesh, UnitSquareMesh, UnitCubeMesh, Function, FunctionSpace,
                    TrialFunction, TestFunction, DirichletBC, MPI, solve)
from dolfin.cpp.mesh import CellType

from ufl import inner, grad, dx, SpatialCoordinate
import numpy as np
import sympy as sp


class ManufacturedSolution():
    def __init__(self, degree, gdim):
        """
        Initialize manufactured solution for given polynomial degree and
        geometric dimension gdim.
        """
        self.x = sp.symbols("x[0] x[1] x[2]")
        x, y, z = self.x
        self.u = (x)**(degree - 1) * (1 - x)

        if gdim > 1:
            self.u *= (y)**(degree - 1) * (1 - y)
        if gdim > 2:
            self.u *= (z)**(degree - 1) * (1 - z)
        self.gdim = gdim

        x_ = sp.symbols("x y z")

        self.u_lambda = self.u
        for i in range(gdim):
            self.u_lambda = self.u_lambda.subs(self.x[i], x_[i])

        self.u_lambda = sp.lambdify((x_), self.u_lambda, 'numpy')

    def eval(self, *args):
        """
        Evaluate function at (x,y,z), where x,y,z each are individual arrays of the x, y and z
        coordinates
        """
        return self.u_lambda(*args)

    def __call__(self, x):
        """
        Evaluate function at a single point (x,y,z) in space
        """
        out_u = self.u
        for i in range(self.gdim):
            out_u = out_u.subs(self.x[i], x[i])
        return out_u

    def laplace(self):
        """
        Compute Laplace of analytical solution to obtain source function
        """
        laplace_u = 0
        for i in range(self.gdim):
            laplace_u += sp.diff(self.u, self.x[i], self.x[i])
        return laplace_u


def boundary(x):
    """
    gdim-dimensional boundary marker
    """
    condition = np.logical_or(x[:, 0] < 1.0e-6, x[:, 0] > 1.0 - 1.0e-6)
    for dim in range(1, x.shape[1]):
        c_dim = np.logical_or(x[:, dim] < 1.0e-6, x[:, dim] > 1.0 - 1.0e-6)
        condition = np.logical_or(condition, c_dim)
    return condition


@pytest.mark.parametrize("degree", [2, 3, 4])
@pytest.mark.parametrize("cell_type, gdim", [(CellType.interval, 1), (CellType.triangle, 2),
                                             (CellType.quadrilateral, 2), (CellType.tetrahedron, 3),
                                             (CellType.hexahedron, 3)])
def test_manufactured_poisson(degree, cell_type, gdim):
    """
    Manufactured Poisson problem, solving u = Pi_{i=0}^gdim (1 - x[i]) * x[i]^(p - 1)
    where p is the degree of the Lagrange function space. Solved on the
    gdim-dimensional UnitMesh, with homogeneous Dirichlet boundary conditions.

    Comparing values at each dof with the analytical solution.

    """
    if gdim == 1:
        mesh = UnitIntervalMesh(MPI.comm_world, 10)
    if gdim == 2:
        mesh = UnitSquareMesh(MPI.comm_world, 15, 15, cell_type)
    elif gdim == 3:
        mesh = UnitCubeMesh(MPI.comm_world, 10, 10, 10, cell_type)
    V = FunctionSpace(mesh, ("CG", degree))
    u, v = TrialFunction(V), TestFunction(V)
    x = SpatialCoordinate(mesh)

    man_sol = ManufacturedSolution(degree, mesh.geometric_dimension())

    f = -eval(str(man_sol.laplace()))
    a = inner(grad(u), grad(v)) * dx
    lhs = inner(f, v) * dx

    u_bc = Function(V)
    with u_bc.vector.localForm() as u_local:
        u_local.set(0.0)
    bc = DirichletBC(V, u_bc, boundary)

    uh = Function(V)
    solve(a == lhs, uh, bc)
    x = V.tabulate_dof_coordinates()
    x_ = np.zeros((x.shape[0], 3))
    # Fill with coordinates to work with eval
    if gdim == 1:
        x_[:, :1] = x
    elif gdim == 2:
        x_[:, :2] = x
    else:
        x_ = x

    for i in range(V.mesh.num_cells()):
        # Find points of dofs in i-th cell
        dofs_i = V.dofmap.cell_dofs(i)
        x_i = np.array([x_[dof, :] for dof in dofs_i])

        # Function values for dofs at i-th cell
        result_i = uh.eval(x_i, i * np.ones(len(x_i)))

        # Compute exact solution
        if mesh.geometric_dimension() == 1:
            exact_i = man_sol.eval(x_i[:, 0], 0, 0)
        elif mesh.geometric_dimension() == 2:
            exact_i = man_sol.eval(x_i[:, 0], x_i[:, 1], 0)
        elif mesh.geometric_dimension() == 3:
            exact_i = man_sol.eval(x_i[:, 0], x_i[:, 1], x_i[:, 2])
        # Reshape
        result_i = result_i.reshape(exact_i.shape)
        # Measure absolute error as values close to zero breaks the rtol.
        assert max(exact_i - result_i) < 1e-3
