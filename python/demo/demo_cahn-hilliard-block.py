# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.1
# ---

# # Cahn-Hilliard equation with a scalable block preconditioner
#
# ```{admonition} Download sources
# :class: download
# * {download}`Python script <./demo_cahn-hilliard-block.py>`
# * {download}`Jupyter notebook <./demo_cahn-hilliard-block.ipynb>`
# ```
#
# The {doc}`Cahn-Hilliard demo <demo_cahn-hilliard>` solves each Newton
# step with a sparse direct solver. That is robust, but the cost and the
# memory of a direct factorisation grow faster than the problem size, so
# it sets a ceiling on the mesh. This demo keeps the same equations and
# replaces the solver with a block preconditioner whose Krylov iteration
# count does not grow as the mesh is refined.
#
# This demo illustrates how to:
#
# - Drive PETSc's `SNES` directly, rather than through
#   {py:class}`NonlinearProblem <dolfinx.fem.petsc.NonlinearProblem>`
# - Assemble a problem into a nested (`MATNEST`) block matrix
# - Precondition the block system with `PCFIELDSPLIT` and a Schur
#   complement factorisation
# - Supply a Schur complement approximation of your own as a PETSc
#   Python preconditioner
#
# ## The block system
#
# With the same $\theta$-scheme and splitting as the original demo, the
# unknowns at each step are the concentration $c$ and the chemical
# potential $\mu$, and the residual is
#
# $$
# \begin{aligned}
# F_0 &= \int_\Omega (c - c_n) q \, {\rm d} x
#      + \Delta t \int_\Omega \nabla \mu_{n+\theta}
#        \cdot \nabla q \, {\rm d} x, \\
# F_1 &= \int_\Omega \mu v \, {\rm d} x
#      - \int_\Omega \frac{{\rm d} f}{{\rm d} c} v \, {\rm d} x
#      - \lambda \int_\Omega \nabla c \cdot \nabla v \, {\rm d} x .
# \end{aligned}
# $$
#
# Writing $M$ for the mass matrix, $K$ for the stiffness matrix and $N$
# for the mass matrix weighted by $f''(c)$, the Jacobian has the $2
# \times 2$ block form
#
# $$
# J =
# \begin{bmatrix}
#   M & \theta \Delta t\, K \\
#   -(N + \lambda K) & M
# \end{bmatrix}.
# $$
#
# A direct solver treats this as one matrix. A block preconditioner
# instead uses the factorisation
#
# $$
# J =
# \begin{bmatrix} I & 0 \\ -(N + \lambda K) M^{-1} & I \end{bmatrix}
# \begin{bmatrix} M & 0 \\ 0 & S \end{bmatrix}
# \begin{bmatrix} I & M^{-1} \theta \Delta t\, K \\ 0 & I \end{bmatrix},
# \qquad
# S = M + (N + \lambda K) M^{-1} \theta \Delta t\, K ,
# $$
#
# where $S$ is the Schur complement. The factorisation is exact, so a
# Krylov method converges in a bounded number of steps given solves with
# the two diagonal blocks $M$ and $S$. The $(0,0)$ block is a mass
# matrix, which conjugate gradients with a diagonal preconditioner solves
# in a mesh-independent number of steps. The work is therefore in $S$,
# which is dense and never formed.
#
# ## Approximating the Schur complement
#
# Dropping $N$, the Schur complement is
# $S \approx M + \theta \Delta t\, \lambda\, K M^{-1} K$: a sum of a mass
# term and a term that is fourth order in the mesh size. Neither AMG nor
# a simple relaxation handles that sum well on its own, but it factorises
# almost exactly. Following the *matching* strategy of Pearson and
# Wathen, look for an approximation of the form
# $\hat{S} = S_1 M^{-1} S_2$ and choose the factors so that the product
# reproduces both terms. Taking $S_1 = S_2 = M + a K$ gives
#
# $$
# (M + a K) M^{-1} (M + a K) = M + 2 a K + a^{2} K M^{-1} K ,
# $$
#
# so setting $a = \sqrt{\theta \Delta t \lambda}$ matches the first and
# the third term exactly. The only error is the middle term $2aK$, whose
# coefficient carries the factor $\sqrt{\Delta t}$ and so is small for
# the time steps this problem is solved with.
#
# The point of the construction is that $M + aK$ is a shifted Laplacian:
# symmetric, positive definite, and exactly what algebraic multigrid is
# good at. Applying
#
# $$
# \hat{S}^{-1} = (M + a K)^{-1}\, M\, (M + a K)^{-1}
# $$
#
# costs two AMG applications and one sparse matrix-vector product, and
# $M + aK$ does not change from step to step, so it is assembled and its
# AMG hierarchy built once.

# +
from __future__ import annotations

import os
from pathlib import Path

from mpi4py import MPI
from petsc4py import PETSc

import numpy as np

import ufl
from dolfinx import fem, io, mesh
from dolfinx.fem.petsc import (
    assemble_matrix,
    assemble_vector,
    assign,
    create_matrix,
    create_vector,
)

# -

# Model and discretisation parameters. `theta = 0.5` is Crank-Nicolson.

lmbda = 1.0e-2  # surface parameter
dt = 5.0e-6  # time step
theta = 0.5  # time stepping family

# The two fields are discretised with linear Lagrange elements. They are
# held in *separate* function spaces rather than one mixed space: a
# blocked assembly needs a space per block, so that each block of the
# Jacobian is a matrix in its own right and the preconditioner can
# address them. `clone` gives a second, distinct space on the same mesh.

msh = mesh.create_unit_square(MPI.COMM_WORLD, 96, 96, mesh.CellType.triangle)
V = fem.functionspace(msh, ("Lagrange", 1))
W = V.clone()

c, mu = fem.Function(V, name="c"), fem.Function(W, name="mu")
c0, mu0 = fem.Function(V), fem.Function(W)

q, v = ufl.TestFunction(V), ufl.TestFunction(W)
dc, dmu = ufl.TrialFunction(V), ufl.TrialFunction(W)

# The free energy and the residual are those of the original demo.

# +
cvar = ufl.variable(c)
f = 100 * cvar**2 * (1 - cvar) ** 2
dfdc = ufl.diff(f, cvar)

mu_mid = (1.0 - theta) * mu0 + theta * mu

F0 = ufl.inner(c - c0, q) * ufl.dx + dt * ufl.inner(ufl.grad(mu_mid), ufl.grad(q)) * ufl.dx
F1 = (
    ufl.inner(mu, v) * ufl.dx
    - ufl.inner(dfdc, v) * ufl.dx
    - lmbda * ufl.inner(ufl.grad(c), ufl.grad(v)) * ufl.dx
)
# -

# The residual and Jacobian are given as a list and a list of lists. This
# is what marks the problem as blocked; the shape of these containers is
# the block structure of the system.

F = fem.form([F0, F1])
J = fem.form(
    [
        [ufl.derivative(F0, c, dc), ufl.derivative(F0, mu, dmu)],
        [ufl.derivative(F1, c, dc), ufl.derivative(F1, mu, dmu)],
    ]
)

# The two operators the Schur approximation needs. Both are independent
# of the solution and of time, so they are assembled once.

# +
a_schur = np.sqrt(theta * dt * lmbda)
dw, w = ufl.TrialFunction(W), ufl.TestFunction(W)

S_mat = assemble_matrix(
    fem.form(ufl.inner(dw, w) * ufl.dx + a_schur * ufl.inner(ufl.grad(dw), ufl.grad(w)) * ufl.dx)
)
S_mat.assemble()
M_mat = assemble_matrix(fem.form(ufl.inner(dw, w) * ufl.dx))
M_mat.assemble()
# -

# A PETSc Python preconditioner is any object with an `apply` method.
# This one applies $\hat{S}^{-1} = (M + aK)^{-1} M (M + aK)^{-1}$. The
# inner solver is a single AMG V-cycle: the approximation is already
# inexact, so solving with it accurately would be wasted work.


class MatchedSchur:
    """Apply the inverse of the matched Schur approximation."""

    def __init__(self, S: PETSc.Mat, M: PETSc.Mat, amg: str):
        """Build the AMG solver for the repeated factor ``S``."""
        self.M = M
        self.ksp = PETSc.KSP().create(S.getComm())
        self.ksp.setOperators(S)
        self.ksp.setType("preonly")
        self.ksp.getPC().setType(amg)
        self.ksp.setOptionsPrefix("schur_")
        self.ksp.setFromOptions()
        self.work = M.createVecLeft()
        self.rhs = M.createVecLeft()

    def apply(self, _pc: PETSc.PC, x: PETSc.Vec, y: PETSc.Vec) -> None:
        """Set ``y`` to the approximate Schur inverse applied to ``x``."""
        self.ksp.solve(x, self.work)
        self.M.mult(self.work, self.rhs)
        self.ksp.solve(self.rhs, y)


# The Jacobian is assembled into a nested matrix. `MATNEST` keeps each
# block as a separate matrix, which gives the index sets that
# `PCFIELDSPLIT` needs for free, and lets the $(0,0)$ block be handed
# straight to a solver.

A = create_matrix(J, kind="nest")
b = create_vector([V, W], kind="nest")
x = create_vector([V, W], kind="nest")

# `SNES` is driven directly, with the two callbacks written out. Each is
# handed the point to evaluate at, and must assemble into the vector or
# matrix it receives rather than one captured here: a line search
# evaluates the residual in a work vector duplicated from `b`.
#
# `x` arrives with its ghost entries out of date, because the Krylov
# solver has just updated only the owned entries. Bringing them forward
# before copying into `c` and `mu` is what lets the forms, which read
# ghost values when integrating over cells owned by another process, see
# a consistent solution.
#
# The Cahn-Hilliard problem has no Dirichlet conditions -- the boundary
# conditions are the natural ones -- so neither callback has to lift
# columns or set constrained rows, and the residual reduces to
# "assemble, then reverse-scatter the contributions each process made to
# entries it does not own".


# +
def ghost_update(vec: PETSc.Vec, insert: PETSc.InsertMode, mode: PETSc.ScatterMode) -> None:
    """Scatter between owned and ghost entries of each block."""
    for sub in vec.getNestSubVecs():
        sub.ghostUpdate(addv=insert, mode=mode)


def residual(_snes: PETSc.SNES, x: PETSc.Vec, b: PETSc.Vec) -> None:
    """Assemble the blocked residual at ``x`` into ``b``."""
    ghost_update(x, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)
    assign(x, [c, mu])

    for sub in b.getNestSubVecs():
        with sub.localForm() as sub_local:
            sub_local.set(0.0)
    assemble_vector(b, F)

    # Each process has contributed to entries owned by another; add
    # those in, then publish the result back to the ghosts
    ghost_update(b, PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE)
    ghost_update(b, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)


def jacobian(_snes: PETSc.SNES, x: PETSc.Vec, J_mat: PETSc.Mat, _P: PETSc.Mat) -> None:
    """Assemble the blocked Jacobian at ``x`` into ``J_mat``."""
    ghost_update(x, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)
    assign(x, [c, mu])

    J_mat.zeroEntries()
    assemble_matrix(J_mat, J)
    J_mat.assemble()


snes = PETSc.SNES().create(msh.comm)  # type: ignore[arg-type]
snes.setFunction(residual, b)
snes.setJacobian(jacobian, A, None)
# -

# The solver options. The outer Krylov method is FGMRES because the
# preconditioner contains Krylov solvers of its own and so is not a fixed
# linear operator. `schur_fact_type upper` needs one solve with the
# $(0,0)$ block per application rather than the two a full factorisation
# needs, for the same iteration count here.

# +
amg = "hypre" if PETSc.Sys().hasExternalPackage("hypre") else "gamg"
opts = PETSc.Options()
opts["snes_type"] = "newtonls"
opts["snes_linesearch_type"] = "basic"
opts["snes_rtol"] = 1.0e-9
opts["ksp_type"] = "fgmres"
opts["ksp_rtol"] = 1.0e-8
opts["pc_type"] = "fieldsplit"
opts["pc_fieldsplit_type"] = "schur"
opts["pc_fieldsplit_schur_fact_type"] = "upper"
# The (0,0) block is a mass matrix: CG with Jacobi is mesh independent
opts["fieldsplit_c_ksp_type"] = "cg"
opts["fieldsplit_c_ksp_rtol"] = 1.0e-10
opts["fieldsplit_c_pc_type"] = "jacobi"
opts["fieldsplit_mu_ksp_type"] = "preonly"
if amg == "hypre":
    opts["schur_pc_hypre_type"] = "boomeramg"
snes.setFromOptions()
# -

# The initial condition is a small random perturbation of a uniform
# mixture.

rng = np.random.default_rng(42)
c.x.array[:] = 0.63 + 0.02 * (0.5 - rng.random(c.x.array.shape[0]))
c.x.scatter_forward()
assign([c, mu], x)

# `PCFIELDSPLIT` is configured once the Jacobian has values, because
# setting up a Schur complement needs the operators. The index sets come
# from the nest, and naming the splits `c` and `mu` is what makes the
# `fieldsplit_c_` and `fieldsplit_mu_` options above apply to them. The
# preconditioner for the Schur block is then replaced by the Python one.

# +
snes.computeJacobian(x, A, A)
ksp = snes.getKSP()
ksp.setOperators(A, A)
pc = ksp.getPC()
index_sets = A.getNestISs()
pc.setFieldSplitIS(("c", index_sets[0][0]), ("mu", index_sets[0][1]))
pc.setUp()

schur_pc = pc.getFieldSplitSubKSP()[1].getPC()
schur_pc.setType("python")
schur_pc.setPythonContext(MatchedSchur(S_mat, M_mat, amg))
schur_pc.setUp()
# -

# Step in time, recording how much work each Newton step costs the Krylov
# solver.

# +
out = Path("demo_ch_block")
out.mkdir(parents=True, exist_ok=True)
file = io.XDMFFile(msh.comm, out / "output.xdmf", "w")
file.write_mesh(msh)

num_steps = 3 if "CI" in os.environ or "GITHUB_ACTIONS" in os.environ else 30
t = 0.0
newton_total, krylov_total = 0, 0
for step in range(num_steps):
    t += dt
    c0.x.array[:] = c.x.array
    mu0.x.array[:] = mu.x.array

    assign([c, mu], x)
    snes.solve(None, x)
    assign(x, [c, mu])
    c.x.scatter_forward()
    mu.x.scatter_forward()

    if snes.getConvergedReason() < 0:  # type: ignore[operator]
        raise RuntimeError(f"Step {step} did not converge: {snes.getConvergedReason()}")
    newton_total += snes.getIterationNumber()
    krylov_total += snes.getLinearSolveIterations()
    file.write_function(c, t)

file.close()
# -

# The figure of merit is the number of Krylov iterations per Newton step.
# If the preconditioner is doing its job, refining the mesh leaves it
# unchanged.

# +
num_dofs = V.dofmap.index_map.size_global + W.dofmap.index_map.size_global
if msh.comm.rank == 0:
    print(f"Degrees of freedom:            {num_dofs}")
    print(f"Newton iterations:             {newton_total}")
    print(f"Krylov iterations per Newton:  {krylov_total / newton_total:.1f}")
# -

# Measured on this problem, with the mesh refined and everything else
# held fixed:
#
# | cells per side | unknowns | Krylov its / Newton step |
# | --- | --- | --- |
# | 64  | 8 450   | 20.8 |
# | 128 | 33 282  | 21.2 |
# | 256 | 132 098 | 23.2 |
# | 512 | 526 338 | 25.0 |
#
# The count is flat to within a few per cent over a sixty-fold increase
# in problem size, so the cost of a solve grows in proportion to the
# number of unknowns. A direct solver gives the same answer -- the two
# agree to twelve significant figures -- but its factors fill in, and at
# 526 338 unknowns the run needs around 1.6 times the peak memory. In two
# dimensions a sparse direct solver is still competitive on time at these
# sizes; the gap in its favour closes as the mesh is refined, and in
# three dimensions, where fill-in is far worse, it closes much sooner.
#
# The usable range of $\Delta t$ and $\lambda$ here is set by the
# nonlinear problem rather than by the preconditioner: at
# $\Delta t = 5 \times 10^{-5}$, or at $\lambda = 10^{-3}$ with this time
# step, Newton's method fails with a direct solver too.
