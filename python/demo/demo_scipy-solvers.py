# # $L^2$ projection through minimization using SciPy solvers
# In this demo, we will go through how to do a `projection` of a spatially varying function `g`,
# into a function space `V`.
#
# An $L^2$ projection can be viewed as the following minimization problem:
#
# $$ \min_{u\in V} G(u) = \frac{1}{2}\int_\Omega (u-g)\cdot (u-g)~\mathrm{d}x.$$

# We start by importing the necessary modules

from mpi4py import MPI

import numpy as np
import numpy.typing as npt
import scipy.optimize
import scipy.sparse.linalg

import dolfinx
import ufl

# Next we define the `domain` ($\Omega$) and the function space `V` where we want to
# project the function `g` into.

Nx = 15
Ny = 20
domain = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, Nx, Ny)
V = dolfinx.fem.functionspace(domain, ("Lagrange", 1))
x, y = ufl.SpatialCoordinate(domain)
g = ufl.sin(ufl.pi * x) * ufl.cos(ufl.pi * y)

# The functional $G$ can be defined as follows

uh = dolfinx.fem.Function(V)
G = ufl.inner(uh - g, uh - g) * ufl.dx

G_compiled = dolfinx.fem.form(G)

# We can find the minimum of this function by differentiating
# the function with respect to `u`:
#
# $$ F(u, \delta u) = \frac{\mathrm{d}G}{\mathrm{d}u}[\delta u] = 0, \quad \forall \delta u \in V.$$

du = ufl.TestFunction(V)
residual = ufl.derivative(G, uh, du)

# We generate the integration kernels for the residua, and assemble an initial
# residual vector `b`.

F = dolfinx.fem.form(residual)
b = dolfinx.fem.assemble_vector(F)

# A convenience function for computing the value of the functional


def compute_functional_value() -> dolfinx.default_scalar_type:
    local_contribution = dolfinx.fem.assemble_scalar(G_compiled)
    return domain.comm.allreduce(local_contribution, op=MPI.SUM)


# We can use the scipy Newton-Krylov solver to solve this problem.
# This solver takes in a function that computes the residual of the functional
# We re-use our structures for `uh` and `b` to avoid unnecessary allocations.


def compute_residual(x) -> npt.NDArray[dolfinx.default_scalar_type]:
    """
    Evaluate the residual F(x) = 0

    :param x: Input vector with current solution
    :return: Returns residual array
    """
    uh.x.array[:] = x
    b.array[:] = 0
    dolfinx.fem.assemble_vector(b.array, F)
    b.scatter_reverse(dolfinx.la.InsertMode.add)
    return b.array


uh.x.array[:] = 0
print("Jacobian free Newton Krylov")
print(f"Initial functional {compute_functional_value():.2e}")
solution = scipy.optimize.newton_krylov(compute_residual, uh.x.array, method="lgmres", verbose=True)
uh.x.array[:] = solution
print(f"Minimal functional {compute_functional_value():.2e}")

# We could also exploit the Jacobian information and use a Newton method with an exact gradient
#
#  $$ J(u, \delta u, \delta v) = \frac{\mathrm{d} F}{\mathrm{d}u}[\delta u, \delta v] =
# \frac{\mathrm{d}^2G}{\mathrm{d}u^2}[\delta u, \delta v].$$

dv = ufl.TrialFunction(V)
jacobian = ufl.derivative(residual, uh, dv)

# We assemble the Jacobian matrix and its inverse

J_compiled = dolfinx.fem.form(jacobian)
A = dolfinx.fem.assemble_matrix(J_compiled)
A_scipy = A.to_scipy()
Ainv = scipy.sparse.linalg.splu(A_scipy)

# If the Jacobian is independent of uh, we can avoid re-assembling the matrix

is_nonlinear = uh._cpp_object in J_compiled._cpp_object.coefficients

# We next create a function for solving the Krylov subspace problem
# where we use the Jacobian and scipy's sparse LU solver to solve the linear system.


def solve_system(_A, x, **kwargs) -> tuple[npt.NDArray[np.float64], int]:
    """Compute the Jacobian and solve the linear system Ax"""
    if is_nonlinear:
        A.data[:] = 0
        uh.x.array[:] = x
        dolfinx.fem.assemble_matrix(A, J_compiled)
        return scipy.sparse.linalg.splu(A_scipy).solve(x), 0
    else:
        return Ainv.solve(x), 0


print("Newton Krylov with Jacobian")
uh.x.array[:] = 0
print(f"Initial function {compute_functional_value():.2e}")
solution = scipy.optimize.newton_krylov(
    compute_residual, uh.x.array, method=solve_system, verbose=True
)
uh.x.array[:] = solution
print(f"Minimal functional {compute_functional_value():.2e}")
