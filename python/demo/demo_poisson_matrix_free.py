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
# - Validate the results against the exact solution and against a
# direct solver for the assembled matrix
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
# The function $u_{\rm D}$ for the Dirichlet boundary conditions is
# in this case also the exact solution of the posed problem.
#
# ## Implementation
#
# The modules that will be used are imported:

# +
import numpy as np

import ufl
from dolfinx import fem, mesh
from ufl import action, dx, grad, inner

from mpi4py import MPI
from petsc4py import PETSc
from petsc4py.PETSc import ScalarType

# -

# We begin by using {py:func}`create_rectangle
# <dolfinx.mesh.create_rectangle>` to create a rectangular
# {py:class}`Mesh <dolfinx.mesh.Mesh>` of the domain, and creating a
# finite element {py:class}`FunctionSpace <dolfinx.fem.FunctionSpace>`
# $V$ on the mesh.

msh = mesh.create_rectangle(comm=MPI.COMM_SELF,
                            points=((0.0, 0.0), (1.0, 1.0)), n=(10, 10),
                            cell_type=mesh.CellType.triangle,
                            ghost_mode=mesh.GhostMode.none)
V = fem.FunctionSpace(msh, ("Lagrange", 2))

# The second argument to {py:class}`FunctionSpace
# <dolfinx.fem.FunctionSpace>` is a tuple consisting of `(family,
# degree)`, where `family` is the finite element family, and `degree`
# specifies the polynomial degree. in this case `V` consists of
# second-order, continuous Lagrange finite element functions.
#
# Next, we locate the mesh facets that lie on the
# domain boundary $\partial\Omega$.
# We can do this by first calling {py:func}`create_connectivity
# <dolfinx.mesh.locate_entities_boundary>` and then retrieving all
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
# {py:class}`Function <dolfinx.fem.Function>` `u_D`, which is obtained by
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
# the {py:class}`action <ufl.action>` of the bilinear form `a` onto an
# arbitrary {py:class}`Function <dolfinx.fem.Function>` `ui`.

ui = fem.Function(V)
M = action(a, ui)

# ### Direct solver using the assembled matrix
#
# For validation

problem = fem.petsc.LinearProblem(a, L, bcs=[bc],
                                  petsc_options={"ksp_type": "preonly",
                                                 "pc_type": "lu"})
uh_lu = problem.solve()


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

fem.set_bc(ui.x.array, [bc], scale=-1)
b = fem.petsc.assemble_vector(fem.form(L))
b += fem.petsc.assemble_vector(fem.form(M))
b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
fem.set_bc(b.array, [bc], scale=0.0)
b.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)


# #### 1. Implementation using PETSc vectors

def action_A(x_vec, y_vec):
    x_vec.copy(ui.vector)
    y = fem.petsc.assemble_vector(fem.form(M))
    fem.set_bc(y, [bc], scale=0.0)
    y.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    y.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    y.copy(y_vec)


def cg(action_A, b, x, kmax=50, rtol=1e-8):
    """CG method
    """
    y = b.duplicate()
    action_A(x, y)
    r = b - y
    p = r.duplicate()
    r.copy(p)
    r_norm2 = r.dot(r)
    r0_norm2 = r_norm2
    eps = rtol**2
    k = 0
    while (k < kmax):
        k += 1
        action_A(p, y)
        alpha = r_norm2 / p.dot(y)
        x.axpy(alpha, p)
        r.axpy(-alpha, y)
        r_norm2_new = r.dot(r)
        beta = r_norm2_new / r_norm2
        r_norm2 = r_norm2_new
        if abs(r_norm2 / r0_norm2) < eps:
            break
        p.aypx(beta, r)
    return k


# +
uh_cg1 = fem.Function(V, dtype=ScalarType)
k = cg(action_A, b, uh_cg1.vector, kmax=200, rtol=1e-6)
fem.set_bc(uh_cg1.vector, [bc], scale=1.0)

error_L2_cg1 = L2Norm(uh_cg1 - uD)
error_lu_cg1 = np.linalg.norm(uh_cg1.x.array - uh_lu.x.array)
if msh.comm.rank == 0:
    print("Matrix-free CG solver using PETSc vectors:")
    print(f"CG iterations until convergence:  {k}")
    print(f"L2-error against exact solution:  {error_L2_cg1:.4e}")
    print(f"Coeff. error against LU solution: {error_lu_cg1:.4e}")


# -

# #### 2. Implementation using NumPy arrays

def action_A_np(x_array):
    ui.vector.setArray(x_array)
    y = fem.petsc.assemble_vector(fem.form(M))
    fem.set_bc(y.array, [bc], scale=0)
    return y.array


def cg_np(action_A, b, x0=None, kmax=50, rtol=1e-8):
    """CG method
    """
    if x0 is None:
        x0 = np.zeros_like(b)
    r = b - action_A(x0)
    p = r
    r_norm2 = np.dot(r, r)
    r0_norm2 = r_norm2
    x = np.copy(x0)
    k = 0
    eps = rtol**2
    while (k < kmax):
        k += 1
        y = action_A(p)
        alpha = r_norm2 / np.dot(p, y)
        x = x + alpha * p
        r = r - alpha * y
        r_norm2_new = np.dot(r, r)
        beta = r_norm2_new / r_norm2
        p = r + beta * p
        r_norm2 = r_norm2_new
        if r_norm2 / r0_norm2 < eps:
            break
    return x, k


# +
cg2_x_array, k = cg_np(action_A_np, b.array, kmax=200, rtol=1e-6)
fem.set_bc(cg2_x_array, [bc], scale=1.0)
uh_cg2 = fem.Function(V, dtype=ScalarType)
uh_cg2.vector.setArray(cg2_x_array)

error_L2_cg2 = L2Norm(uh_cg2 - uD)
error_lu_cg2 = np.linalg.norm(uh_cg2.x.array - uh_lu.x.array)
if msh.comm.rank == 0:
    print("Matrix-free CG solver using NumPy arrays:")
    print(f"CG iterations until convergence:  {k}")
    print(f"L2-error against exact solution:  {error_L2_cg2:.4e}")
    print(f"Coeff. error against LU solution: {error_lu_cg2:.4e}")


# -
# ### 3. Implementation using the built-in PETSc KSP solver

class Poisson:
    def create(self, A):
        M, N = A.getSize()
        assert M == N

    def mult(self, A, x, y):
        action_A(x, y)


A = PETSc.Mat()
A.create(comm=msh.comm)
print(b.getSize(), b.local_size, b.getLocalSize(), b.getSizes(), b.getBlockSize())
A.setSizes([b.getSize(), b.getSize()])
A.setType(PETSc.Mat.Type.PYTHON)
A.setPythonContext(Poisson())
A.setUp()

solver = PETSc.KSP().create(msh.comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.CG)
solver.getPC().setType(PETSc.PC.Type.NONE)
solver.setTolerances(rtol=1e-6)
solver.setConvergenceHistory()

# +
uh_cg3 = fem.Function(V)
solver.solve(b, uh_cg3.vector)
fem.set_bc(uh_cg3.vector, [bc], scale=1.0)

error_L2_cg3 = L2Norm(uh_cg3 - uD)
error_lu_cg3 = np.linalg.norm(uh_cg3.x.array - uh_lu.x.array)
if msh.comm.rank == 0:
    print("Matrix-free CG solver using the built-in PETSc KSP solver:")
    print(f"CG iterations until convergence:  {k}")
    print(f"L2-error against exact solution:  {error_L2_cg3:.4e}")
    print(f"Coeff. error against LU solution: {error_lu_cg3:.4e}")


# -

# ### 4. Implementation using a custom PETSc KSP solver

class CustomKSP:
    def create(self, ksp):
        self.vectors = []

    def destroy(self, ksp):
        for v in self.vectors:
            v.destroy()

    def setUp(self, ksp):
        self.vectors[:] = ksp.getWorkVecs(right=2, left=None)

    def reset(self, ksp):
        for v in self.vectors:
            v.destroy()
        del self.vectors[:]

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


class CG(CustomKSP):
    def setUp(self, ksp):
        super(CG, self).setUp(ksp)
        p = self.vectors[0].duplicate()
        y = p.duplicate()
        self.vectors += [p, y]

    def solve(self, ksp, b, x):
        A, _ = ksp.getOperators()
        r, _, p, y = self.vectors
        A.mult(x, r)
        r.aypx(-1, b)
        r.copy(p)
        r0_norm = r.dot(r)
        r_norm = r0_norm
        while not self.converged(ksp, r):
            A.mult(p, y)
            alpha = r_norm / p.dot(y)
            x.axpy(alpha, p)
            r.axpy(-alpha, y)
            r_norm_new = r.dot(r)
            beta = r_norm_new / r_norm
            r_norm = r_norm_new
            p.aypx(beta, r)


solver = PETSc.KSP().create(msh.comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.PYTHON)
solver.setPythonContext(CG())
solver.setTolerances(rtol=1e-6)
solver.setConvergenceHistory()

# +
uh_cg4 = fem.Function(V)
solver.solve(b, uh_cg4.vector)
fem.set_bc(uh_cg4.vector, [bc], scale=1.0)

error_L2_cg4 = L2Norm(uh_cg4 - uD)
error_lu_cg4 = np.linalg.norm(uh_cg4.x.array - uh_lu.x.array)
if msh.comm.rank == 0:
    print("Matrix-free CG solver using a custom PETSc KSP solver:")
    print(f"CG iterations until convergence:  {k}")
    print(f"L2-error against exact solution:  {error_L2_cg4:.4e}")
    print(f"Coeff. error against LU solution: {error_lu_cg4:.4e}")
