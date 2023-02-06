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
# This demo is implemented in a single Python file,
# {download}`demo_poisson_matrix_free.py`, which contains both the
# variational forms and the solver. It illustrates how to:
#
# - Solve a linear partial differential equation using a matrix-free
# Conjugate Gradient (CG) solver
# - Create and apply Dirichlet boundary conditions
# - Compute errors against the exact solution and against a
# direct solver for the assembled matrix
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

import numpy as np

import ufl
from dolfinx import fem, mesh
from ufl import action, dx, grad, inner

from mpi4py import MPI
from petsc4py import PETSc
from petsc4py.PETSc import ScalarType

# We begin by using {py:func}`create_rectangle
# <dolfinx.mesh.create_rectangle>` to create a rectangular
# {py:class}`Mesh <dolfinx.mesh.Mesh>` of the domain, and creating a
# finite element {py:class}`FunctionSpace <dolfinx.fem.FunctionSpace>`
# $V$ on the mesh.

msh = mesh.create_rectangle(comm=MPI.COMM_WORLD,
                            points=((0.0, 0.0), (1.0, 1.0)), n=(10, 10),
                            cell_type=mesh.CellType.triangle,
                            ghost_mode=mesh.GhostMode.none)
V = fem.FunctionSpace(msh, ("Lagrange", 2))

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

msh.topology.create_connectivity(1, msh.topology.dim)
facets = mesh.exterior_facet_indices(msh.topology)

# We now find the degrees-of-freedom that are associated with the
# boundary facets using {py:func}`locate_dofs_topological
# <dolfinx.fem.locate_dofs_topological>`

dofs = fem.locate_dofs_topological(V=V, entity_dim=1, entities=facets)

# and use {py:func}`dirichletbc <dolfinx.fem.dirichletbc>` to create a
# {py:class}`DirichletBCMetaClass <dolfinx.fem.DirichletBCMetaClass>`
# class that represents the boundary condition. On the boundary we prescribe
# the {py:class}`Function <dolfinx.fem.Function>` `uD`, which is obtained by
# interpolating the expression $u_{\rm D}$ onto the finite element space $V$.

uD = fem.Function(V, dtype=ScalarType)
uD.interpolate(lambda x: 1 + x[0]**2 + 2 * x[1]**2)
bc = fem.dirichletbc(value=uD, dofs=dofs)

# Next, we express the variational problem using UFL.

x = ufl.SpatialCoordinate(msh)
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
f = fem.Constant(msh, ScalarType(-6))
a = inner(grad(u), grad(v)) * dx
L = inner(f, v) * dx

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

# ### Direct solver using the assembled matrix
#
# To validate the results of the matrix-free solvers, we first compute the
# solution with a direct solver using the assembled matrix.

problem = fem.petsc.LinearProblem(a, L, bcs=[bc],
                                  petsc_options={"ksp_type": "preonly",
                                                 "pc_type": "lu"})
uh_lu = problem.solve()


# The error of the finite element solution `uh_lu` compared to the exact
# solution $u_{\rm D}$ is calculated below in the $L_2$-norm.

# +
def L2Norm(u):
    return np.sqrt(msh.comm.allreduce(
        fem.assemble_scalar(fem.form(inner(u, u) * dx)),
        op=MPI.SUM))


error_L2_lu = L2Norm(uh_lu - uD)
if msh.comm.rank == 0:
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

b = fem.petsc.assemble_vector(fem.form(L))
# Apply lifting: b <- b - A * x_bc
fem.set_bc(ui.x.array, [bc], scale=-1)
fem.petsc.assemble_vector(b, fem.form(M))
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
fem.petsc.set_bc(b, [bc], scale=0.0)
b.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

# In the following, different variants are presented in which the posed
# Poisson problem is solved using matrix-free CG solvers. In each case
# we want to achieve convergence up to a relative tolerence `rtol = 1e-6`
# within `max_iter = 200` iterations.

rtol = 1e-6
max_iter = 200


# #### 1. Implementation using PETSc vectors

# To implement the matrix-free CG solver using *PETSc* vectors, we define the
# function `action_A` with which the matrix-vector product $y = A x$
# is computed.

def action_A(x):
    # Update coefficient ui of the linear form M
    x.copy(ui.vector)
    ui.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                          mode=PETSc.ScatterMode.FORWARD)

    # Compute action of A on x using the linear form M
    y = fem.petsc.assemble_vector(fem.form(M))

    # Set BC dofs to zero (effectively zeroes rows of A)
    with y.localForm() as y_local:
        fem.set_bc(y_local, [bc], scale=0.0)
    y.ghostUpdate(addv=PETSc.InsertMode.ADD,
                  mode=PETSc.ScatterMode.REVERSE)
    return y


# This function can be used to replace the matrix-vector product in the
# plain Conjugate Gradient method by Hestenes and Stiefel.

def cg(action_A, b, x, max_iter=200, rtol=1e-6):
    # Create working vectors
    y = b.duplicate()
    b.copy(y)

    # Compute initial residual r0 = b - A x0
    y = action_A(x)
    r = b - y

    # Create work vector for the search direction p
    p = r.duplicate()
    r.copy(p)
    p.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                  mode=PETSc.ScatterMode.FORWARD)
    r_norm2 = r.dot(r)
    r0_norm2 = r_norm2
    eps = rtol**2
    k = 0
    while k < max_iter:
        k += 1

        # Compute y = A p
        y = action_A(p)

        # Compute alpha = r.r / p.y
        alpha = r_norm2 / p.dot(y)

        # Update x (x <- x + alpha * p)
        x.axpy(alpha, p)

        # Update r (r <- r - alpha * y)
        r.axpy(-alpha, y)

        # Update residual norm
        r_norm2_new = r.dot(r)
        beta = r_norm2_new / r_norm2
        r_norm2 = r_norm2_new

        # Convergence test
        if abs(r_norm2 / r0_norm2) < eps:
            break

        # Update p (p <- beta * p + r)
        p.aypx(beta, r)
    return k


# This matrix-free solver is now used to compute the finite element solution.
# After that, the error against the exact solution in the $L_2$-norm and the
# error of the coefficients against the solution obtained by the direct
# solver is computed.

# +
uh_cg1 = fem.Function(V, dtype=ScalarType)
iter_cg1 = cg(action_A, b, uh_cg1.vector, max_iter=max_iter, rtol=rtol)

# Set BC values in the solution vectors
uh_cg1.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                          mode=PETSc.ScatterMode.FORWARD)
with uh_cg1.vector.localForm() as y_local:
    fem.set_bc(y_local, [bc], scale=1.0)

# Print CG iteration number and errors
error_L2_cg1 = L2Norm(uh_cg1 - uD)
error_lu_cg1 = np.linalg.norm(uh_cg1.x.array - uh_lu.x.array)
if msh.comm.rank == 0:
    print("Matrix-free CG solver using PETSc vectors:")
    print(f"CG iterations until convergence:  {iter_cg1}")
    print(f"L2-error against exact solution:  {error_L2_cg1:.4e}")
    print(f"Coeff. error against LU solution: {error_lu_cg1:.4e}")
assert error_L2_cg1 < rtol


# -
# ### 2. Implementation using the built-in PETSc CG solver
#
# Another approach is to use the existing CG solver of *PETSc* with a
# virtual *PETSc* matrix in order to obtain a matrix-free Conjugate
# Gradient solver. For this purpose, we create a class `Poisson` to
# emulate the assembled matrix `A` of the Poisson problem
# considered here.

class Poisson:
    def create(self, A):
        M, N = A.getSize()
        assert M == N

    def mult(self, A, x, y):
        action_A(x).copy(y)


# With this we can define a virtual *PETSc* matrix, where every
# matrix-vector product is implicitly performed matrix-free.

A = PETSc.Mat()
A.create(comm=msh.comm)
A.setSizes(((b.local_size, PETSc.DETERMINE),
            (b.local_size, PETSc.DETERMINE)), bsize=1)
A.setType(PETSc.Mat.Type.PYTHON)
A.setPythonContext(Poisson())
A.setUp()

# This matrix can then be passed as an operator to a predefined
# Conjugate Gradient solver in the KSP framework, automatically making
# that solver matrix-free.

# +
solver = PETSc.KSP().create(msh.comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.CG)
solver.getPC().setType(PETSc.PC.Type.NONE)
solver.setTolerances(rtol=rtol, max_it=max_iter)
solver.setConvergenceHistory()


# Set custom convergence test to resemble our CG solver exactly
def converged(ksp, iter, r_norm):
    rtol, _, _, max_iter = ksp.getTolerances()
    if iter > max_iter:
        return PETSc.KSP.ConvergedReason.DIVERGED_MAX_IT
    r0_norm = ksp.getConvergenceHistory()[0]
    if r_norm / r0_norm < rtol:
        return PETSc.KSP.ConvergedReason.CONVERGED_RTOL
    return PETSc.KSP.ConvergedReason.ITERATING


solver.setConvergenceTest(converged)
# -

# Again, the solver is applied and the errors are computed.

# +
uh_cg2 = fem.Function(V)
solver.solve(b, uh_cg2.vector)

# Set BC values in the solution vectors
uh_cg2.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                          mode=PETSc.ScatterMode.FORWARD)
with uh_cg2.vector.localForm() as y_local:
    fem.set_bc(y_local, [bc], scale=1.0)

# Print CG iteration number and errors
iter_cg2 = solver.getIterationNumber()
error_L2_cg2 = L2Norm(uh_cg2 - uD)
error_lu_cg2 = np.linalg.norm(uh_cg2.x.array - uh_lu.x.array)
if msh.comm.rank == 0:
    print("Matrix-free CG solver using the built-in PETSc KSP solver:")
    print(f"CG iterations until convergence:  {iter_cg2}")
    print(f"L2-error against exact solution:  {error_L2_cg2:.4e}")
    print(f"Coeff. error against LU solution: {error_lu_cg2:.4e}")
assert error_L2_cg2 < rtol


# -

# ### 3. Implementation using a custom PETSc KSP solver
#
# Furthermore, it is also possible to write a custom Conjugate Gradient
# solver in the KSP framework, which is matrix-free as before. For this
# purpose, a base class for a custom KSP solver is created.

class CustomKSP:
    def create(self, ksp):
        # Work vectors
        self.vectors = []

    def destroy(self, ksp):
        for v in self.vectors:
            v.destroy()

    def setUp(self, ksp):
        self.vectors = ksp.getWorkVecs(right=2, left=None)

    def reset(self, ksp):
        for v in self.vectors:
            v.destroy()
        del self.vectors

    def converged(self, ksp, r):
        k = ksp.getIterationNumber()
        r_norm = r.norm()
        ksp.setResidualNorm(r_norm)
        ksp.logConvergenceHistory(r_norm)
        ksp.monitor(k, r_norm)
        reason = ksp.callConvergenceTest(k, r_norm)
        if not reason:
            ksp.setIterationNumber(k + 1)
        else:
            ksp.setConvergedReason(reason)
        return reason


# A user-defined Conjugate Gradient solver can then be defined based
# on this prototype.

class CG(CustomKSP):
    def setUp(self, ksp):
        super(CG, self).setUp(ksp)
        p = self.vectors[0].duplicate()
        y = p.duplicate()
        self.vectors += [p, y]

    def solve(self, ksp, b, x):
        A, _ = ksp.getOperators()

        # Create work vectors
        r, _, p, y = self.vectors
        b.copy(y)

        # Compute initial residual r0 = b - A x0
        A.mult(x, y)
        r = b - y

        # Create work vector for the search direction p
        r.copy(p)
        p.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                      mode=PETSc.ScatterMode.FORWARD)
        r_norm2 = r.dot(r)
        self.r0_norm2 = r_norm2
        while not self.converged(ksp, r):

            # Compute y = A p
            A.mult(p, y)

            # Compute alpha = r.r / p.y
            alpha = r_norm2 / p.dot(y)

            # Update x (x <- x + alpha * p)
            x.axpy(alpha, p)

            # Update x (r <- r - alpha * y)
            r.axpy(-alpha, y)

            # Update residual norm
            r_norm2_new = r.dot(r)
            beta = r_norm2_new / r_norm2
            r_norm2 = r_norm2_new

            # Update p (p <- beta * p + r)
            p.aypx(beta, r)


# As before, a matrix-free solver can be achieved by passing the
# emulated matrix `A` as the operator.

solver = PETSc.KSP().create(msh.comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.PYTHON)
solver.setPythonContext(CG())
solver.setTolerances(rtol=rtol, max_it=max_iter)
solver.setConvergenceHistory()
solver.setConvergenceTest(converged)

# The computed solution is again compared with the exact solution and
# the direct solver using the assembled matrix.

# +
uh_cg3 = fem.Function(V)
solver.solve(b, uh_cg3.vector)

# Set BC values in the solution vectors
uh_cg3.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                          mode=PETSc.ScatterMode.FORWARD)
with uh_cg3.vector.localForm() as y_local:
    fem.set_bc(y_local, [bc], scale=1.0)

# Print CG iteration number and errors
iter_cg3 = solver.getIterationNumber()
error_L2_cg3 = L2Norm(uh_cg3 - uD)
error_lu_cg3 = np.linalg.norm(uh_cg3.x.array - uh_lu.x.array)
if msh.comm.rank == 0:
    print("Matrix-free CG solver using a custom PETSc KSP solver:")
    print(f"CG iterations until convergence:  {iter_cg3}")
    print(f"L2-error against exact solution:  {error_L2_cg3:.4e}")
    print(f"Coeff. error against LU solution: {error_lu_cg3:.4e}")
assert error_L2_cg3 < rtol
