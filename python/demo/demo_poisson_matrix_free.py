# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Matrix-free Conjugate Gradient solver for the Poisson equation
#
# This demo illustrates how to solve the Poisson equation using a
# matrix-free Conjugate Gradient (CG) solver. In particular, it
# illustrates how to
#
# - Solve a linear partial differential equation using a matrix-free
# Conjugate Gradient (CG) solver
# - Create and apply Dirichlet boundary conditions
# - Compute errors against the exact solution and against a
# direct solver for the assembled matrix
#
# {download}`Python script <./demo_poisson_matrix_free.py>`\
# {download}`Jupyter notebook <./demo_poisson_matrix_free.ipynb>`
#
# ```{note}
# This demo illustrates the use of a matrix-free Conjugate Gradient
# solver. Many practical problems will also require a preconditioner
# to create an efficient solver. This is not covered here.
# ```
#
# ## Equation and problem definition
#
# For a domain $\Omega \subset \mathbb{R}^n$ with boundary $\partial
# \Omega$, the Poisson equation with
# Dirichlet boundary conditions reads:
#
# $$
# \begin{align}
# - \nabla^{2} u &= f \quad {\rm in} \ \Omega, \\
#       u &= u_{\rm D} \; {\rm on} \ \partial\Omega.
# \end{align}
# $$
#
# The variational problem reads: find $u \in V$ such
# that
#
# $$
# a(u, v) = L(v) \quad \forall \ v \in V,
# $$
#
# where $V$ is a suitable function space and
#
# $$
# \begin{align}
# a(u, v) &:= \int_{\Omega} \nabla u \cdot \nabla v \, {\rm d} x, \\
# L(v)    &:= \int_{\Omega} f v \, {\rm d} x.
# \end{align}
# $$
#
# The expression $a(u, v)$ is the bilinear form and $L(v)$
# is the linear form. It is assumed that all functions in $V$
# satisfy the Dirichlet boundary conditions ($u = u_{\rm D} \ {\rm on} \
# \partial\Omega$).
#
# In this demo we consider:
#
# - $\Omega = [0,1] \times [0,1]$ (a square)
# - $u_{\rm D} = 1 + x^2 + 2y^2$
# - $f = -6$
#
# The function $u_{\rm D}$ for the Dirichlet boundary condition is
# in this case also the exact solution of the posed problem.
#
# ## Implementation
#
# The modules that will be used are imported:

from mpi4py import MPI

import numpy as np

import basix
import dolfinx
import dolfinx.fem.petsc
import ufl
from dolfinx import fem, la
from ufl import action, dx, grad, inner

ffcx_options = {"sum_factorization": True}

# We begin by using {py:func}`create_rectangle
# <dolfinx.mesh.create_rectangle>` to create a rectangular
# {py:class}`Mesh <dolfinx.mesh.Mesh>` of the domain, and creating a
# finite element {py:class}`FunctionSpace <dolfinx.fem.FunctionSpace>`
# $V$ on the mesh.


# Create mesh with tensor product ordering
comm = MPI.COMM_WORLD
mesh = dolfinx.mesh.create_tp_rectangle(comm, [[0.0, 0.0], [1.0, 1.0]], [10, 10])
dtype = mesh.geometry.x.dtype

# Create function space
degree = 3
element = basix.create_tp_element(
    basix.ElementFamily.P,
    basix.CellType.quadrilateral,
    degree,
    basix.LagrangeVariant.gll_warped,
    dtype=dtype,
)
e_ufl = basix.ufl._BasixElement(element)
V = fem.functionspace(mesh, e_ufl)

# +

# The second argument to {py:class}`FunctionSpace
# <dolfinx.fem.FunctionSpace>` is a tuple consisting of `(family,
# degree)`, where `family` is the finite element family, and `degree`
# specifies the polynomial degree. In this case `V` consists of
# second-order, continuous Lagrange finite element functions.
#
# Next, we locate the mesh facets that lie on the
# domain boundary $\partial\Omega$.
# We can do this by first calling {py:func}`create_connectivity
# <dolfinx.mesh.topology.create_connectivity>` and then retrieving all
# facets on the boundary using
# {py:func}`exterior_facet_indices <dolfinx.mesh.exterior_facet_indices>`.

tdim = mesh.topology.dim
mesh.topology.create_connectivity(tdim - 1, tdim)
facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)

# We now find the degrees-of-freedom that are associated with the
# boundary facets using {py:func}`locate_dofs_topological
# <dolfinx.fem.locate_dofs_topological>`

dofs = fem.locate_dofs_topological(V=V, entity_dim=tdim - 1, entities=facets)

# and use {py:func}`dirichletbc <dolfinx.fem.dirichletbc>` to create a
# {py:class}`DirichletBCMetaClass <dolfinx.fem.DirichletBCMetaClass>`
# class that represents the boundary condition. On the boundary we prescribe
# the {py:class}`Function <dolfinx.fem.Function>` `uD`, which is obtained by
# interpolating the expression $u_{\rm D}$ onto the finite element space $V$.

uD = fem.Function(V, dtype=dtype)
uD.interpolate(lambda x: 1 + x[0] ** 2 + 2 * x[1] ** 2)

bc = fem.dirichletbc(value=uD, dofs=dofs)

# Next, we express the variational problem using UFL.

x = ufl.SpatialCoordinate(mesh)
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
f = fem.Constant(mesh, -6.0)
a = inner(grad(u), grad(v)) * dx
L = inner(f, v) * dx
L_fem = fem.form(L, dtype=dtype, form_compiler_options=ffcx_options)

# For the matrix-free solvers we also define a second linear form `M` as
# the {py:class}`action <ufl.action>` of the bilinear form $a$ onto an
# arbitrary {py:class}`Function <dolfinx.fem.Function>` `ui`. This linear
# form is defined as
#
# $$
# M(v) = a(u_i, v) \quad \text{for} \; \ u_i \in V.
# $$

ui = fem.Function(V)
M = action(a, ui)
M_fem = fem.form(M, dtype=dtype, form_compiler_options=ffcx_options)

# ### Direct solver using the assembled matrix
#
# To validate the results of the matrix-free solvers, we first compute the
# solution with a direct solver using the assembled matrix.
problem = fem.petsc.LinearProblem(
    a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
)
uh_lu = problem.solve()

# The exact solution $u_{\rm D}$ is interpolated onto the finite element


# The error of the finite element solution `uh_lu` compared to the exact
# solution $u_{\rm D}$ is calculated below in the $L_2$-norm.


# +
def L2Norm(u):
    val = fem.assemble_scalar(fem.form(inner(u, u) * dx))
    return np.sqrt(comm.allreduce(val, op=MPI.SUM))


error_L2_lu = L2Norm(uh_lu - uD)
if comm.rank == 0:
    print("Direct solver using the assembled matrix:")
    print(f"L2-error against exact solution:  {error_L2_lu:.4e}")
# -

# ### Matrix-free Conjugate Gradient solvers
#
# For the matrix-free solvers, the RHS vector $b$ is first assembled based
# on the linear form $L$.  To account for the Dirichlet boundary conditions
# in $b$, we apply lifting, i.e. set $b - A x_{\rm bc}$ as new RHS vector $b$.
# Since we want to avoid assembling the matrix `A`, we compute the necessary
# matrix-vector product using the linear form `M` implicitly.

b = fem.assemble_vector(L_fem)
# Apply lifting: b <- b - A * x_bc
ui.x.array[:] = 0.0
fem.set_bc(ui.x.array, [bc], scale=-1.0)
b.scatter_reverse(la.InsertMode.add)

# Set BC dofs to zero on RHS (effectively zeros column in A)
fem.set_bc(b.array, [bc], scale=0.0)
fem.set_bc(ui.x.array, [bc], scale=0.0)
b.scatter_forward()

# In the following, different variants are presented in which the posed
# Poisson problem is solved using matrix-free CG solvers. In each case
# we want to achieve convergence up to a relative tolerence `rtol = 1e-6`
# within `max_iter = 200` iterations.

# #### 1. Implementation using DOLFINx vectors

# To implement the matrix-free CG solver using *DOLFINx* vectors, we define the
# function `action_A` with which the matrix-vector product $y = A x$
# is computed.


def action_A(x, y):
    ui.x.array[:] = x.array

    # Update coefficient ui of the linear form M
    ui.x.scatter_forward()

    # Compute action of A on ui using the linear form M
    y.array[:] = 0.0
    fem.assemble_vector(y.array, M_fem)
    y.scatter_reverse(la.InsertMode.add)

    # Set BC dofs to zero (effectively zeroes rows of A)
    fem.set_bc(y.array, [bc], scale=0.0)


# Basic Conjugate Gradient solver
def cg(comm, action_A, x0, b, max_iter=200, rtol=1e-6):
    rtol2 = rtol**2

    nr = b.index_map.size_local

    def _global_dot(comm, v0, v1):
        # Only use the owned dofs in vector (up to nr)
        return comm.allreduce(np.vdot(v0[:nr], v1[:nr]), MPI.SUM)

    # Get initial y = A.x (using u_i.x)
    y = la.vector(b.index_map, 1, dtype)
    action_A(x0, y)

    # Copy ui.x to x
    x = x0.array.copy()

    # Copy residual to p
    r = b.array - y.array
    p = la.vector(b.index_map, 1, dtype)
    p.array[:] = r

    # Iterations of CG
    rnorm0 = _global_dot(comm, r, r)
    rnorm = rnorm0
    for k in range(max_iter):
        action_A(p, y)
        alpha = rnorm / _global_dot(comm, p.array, y.array)

        x += alpha * p.array
        r -= alpha * y.array
        rnorm_new = _global_dot(comm, r, r)
        beta = rnorm_new / rnorm
        rnorm = rnorm_new
        if comm.rank == 0:
            print(k, np.sqrt(rnorm / rnorm0))
        if rnorm / rnorm0 < rtol2:
            ui.x.array[:] = x[:]
            ui.x.scatter_forward()
            return k
        p.array[:] = beta * p.array + r

    raise RuntimeError(f"Solver exceeded max iterations ({max_iter}).")


# This matrix-free solver is now used to compute the finite element solution.
# After that, the error against the exact solution in the $L_2$-norm and the
# error of the coefficients against the solution obtained by the direct
# solver is computed.
# +
rtol = 1e-6
iter_cg1 = cg(mesh.comm, action_A, ui.x, b, max_iter=200, rtol=rtol)

# Set BC values in the solution vectors
fem.set_bc(ui.x.array, [bc], scale=1.0)

# Print CG iteration number and errors
error_L2_cg1 = L2Norm(ui - uD)
error_lu_cg1 = np.linalg.norm(ui.x.array - uh_lu.x.array)
if mesh.comm.rank == 0:
    print("Matrix-free CG solver using DOLFINx vectors:")
    print(f"CG iterations until convergence:  {iter_cg1}")
    print(f"L2-error against exact solution:  {error_L2_cg1:.4e}")
    print(f"Coeff. error against LU solution: {error_lu_cg1:.4e}")
