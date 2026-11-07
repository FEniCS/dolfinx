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
# Copyright (C) 2026 Garth N. Wells
#
# This file is part of DOLFINx (<https://www.fenicsproject.org>)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
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
# count barely moves as the mesh is refined.
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
# almost exactly. Following the *matching* strategy of
# {cite:t}`Pearson2012schur`, look for an approximation of the form
# $\hat{S} = S_1 M^{-1} S_2$ and choose the factors so that the product
# reproduces both terms. Taking $S_1 = S_2 = M + a K$ gives
#
# $$
# (M + a K) M^{-1} (M + a K) = M + 2 a K + a^{2} K M^{-1} K ,
# $$
#
# so setting $a = \sqrt{\theta \Delta t \lambda}$ matches the first and
# the third term exactly. The only error is the middle term $2aK$, and
# it is bounded. Writing $u = a M^{-1/2} K v$ and $w = M^{1/2} v$, the
# Rayleigh quotient is
#
# $$
# \frac{v^{T} S v}{v^{T} \hat{S} v}
#   = \frac{u^{T}u + w^{T}w}{u^{T}u + w^{T}w + 2 u^{T}w} ,
# \qquad u^{T} w = a\, v^{T} K v \ge 0 ,
# $$
#
# and $0 \le \lVert u - w \rVert^{2}$ gives $2u^{T}w \le u^{T}u +
# w^{T}w$. For the Schur complement with $N$ dropped, the eigenvalues of
# $\hat{S}^{-1} S$ therefore lie in $[1/2, 1]$ whatever the mesh size,
# the time step and $\lambda$. This is the bound proved by
# {cite:t}`Bergermann2023phasefield`.
#
# The bound does not cover the dropped term
# $\theta \Delta t\, N M^{-1} K$. Treating $f''(c)$ as locally constant,
# so that $N \approx f''(c) M$, this term is
# $\theta \Delta t\, f''(c) K$. Since $\hat{S} \ge 2aK$, its size
# relative to $\hat{S}$ is at most
#
# $$
# \frac{\theta \Delta t\, |f''(c)|}{2a}
#   = \frac{|f''(c)|}{2} \sqrt{\frac{\theta \Delta t}{\lambda}} .
# $$
#
# This ratio is independent of the mesh size, which is why the iteration
# counts at the end of this demo barely move under refinement. It is not
# small, though. At the initial $c \approx 0.63$, $f''(c) \approx -80$
# and the ratio is about $0.6$. As the phases separate, $c$ approaches 0
# or 1, where $f'' = 200$, and the ratio rises to about $1.6$. The
# ratio grows with $\Delta t$ and as $\lambda$ decreases.
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
#
# ## The preconditioner
#
# The preconditioner keeps the upper triangular factor of the
# factorisation above, drops the lower one, and replaces $S$ with
# $\hat{S}$:
#
# $$
# P =
# \begin{bmatrix}
#   M & \theta \Delta t\, K \\
#   0 & \hat{S}
# \end{bmatrix},
# \qquad
# P^{-1} =
# \begin{bmatrix}
#   M^{-1} & -M^{-1} \theta \Delta t\, K \hat{S}^{-1} \\
#   0 & \hat{S}^{-1}
# \end{bmatrix},
# \qquad
# \hat{S} = (M + aK) M^{-1} (M + aK) .
# $$
#
# With $\hat{S} = S$, the preconditioned matrix $J P^{-1}$ has the
# single eigenvalue $1$ and GMRES converges in two iterations; the
# eigenvalue bounds above measure how far $\hat{S}$ is from that.
# Applying $P^{-1}$ takes one solve with $\hat{S}$, one product with
# $\theta \Delta t\, K$ and one solve with $M$. In the code, $M^{-1}$ is
# applied by conjugate gradients with a Jacobi preconditioner to a
# relative tolerance of $10^{-2}$, and each $(M + aK)^{-1}$ by a single
# AMG V-cycle. This is PETSc's `PCFIELDSPLIT` with a Schur complement
# factorisation of type `upper`.

# +
import os
from pathlib import Path

from mpi4py import MPI
from petsc4py import PETSc

import numpy as np

import ufl
from dolfinx import default_real_type, fem, io, mesh
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
#
# The demo doubles as a test. That run only has to exercise the code, so
# under CI it takes a coarse mesh and a few steps; run on its own it
# solves the problem the results at the end were measured on.

# +
ci = "CI" in os.environ or "GITHUB_ACTIONS" in os.environ
n = 16 if ci else 64

msh = mesh.create_unit_cube(MPI.COMM_WORLD, n, n, n, mesh.CellType.tetrahedron)
V = fem.functionspace(msh, ("Lagrange", 1))
W = V.clone()
# -

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
# the block structure of the system. The $(1,0)$ block is kept as `J10`
# for the Jacobian callback below.

F = fem.form([F0, F1])
J10 = fem.form(ufl.derivative(F1, c, dc))
J = [
    [fem.form(ufl.derivative(F0, c, dc)), fem.form(ufl.derivative(F0, mu, dmu))],
    [J10, fem.form(ufl.derivative(F1, mu, dmu))],
]

# The Schur approximation needs $M + aK$ and $M$. $M + aK$ is
# independent of the solution and of time, so it is assembled once. $M$
# is the $(1,1)$ Jacobian block, which never changes, so it is taken
# from the Jacobian below rather than assembled again.

# +
a_schur = np.sqrt(theta * dt * lmbda)
S_mat = assemble_matrix(
    fem.form(ufl.inner(dmu, v) * ufl.dx + a_schur * ufl.inner(ufl.grad(dmu), ufl.grad(v)) * ufl.dx)
)
S_mat.assemble()
# -

# A PETSc Python preconditioner is any object with an `apply` method.
# This one applies $\hat{S}^{-1} = (M + aK)^{-1} M (M + aK)^{-1}$. The
# inner solver is a single AMG V-cycle: the approximation is already
# inexact, so solving with it accurately would be wasted work.


class MatchedSchur:
    """Apply the inverse of the matched Schur approximation."""

    def __init__(self, S: PETSc.Mat, M: PETSc.Mat, prefix: str):
        """Build the AMG solver for the repeated factor ``S``."""
        self.M = M
        self.ksp = PETSc.KSP().create(S.getComm())
        self.ksp.setOperators(S)
        self.ksp.setOptionsPrefix(prefix)
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
# `PCFIELDSPLIT` needs for free, lets the $(0,0)$ block be handed
# straight to a solver, and lets one block be reassembled without
# touching the others.

# +
A = create_matrix(J, kind="nest")
b = create_vector([V, W], kind="nest")
x = create_vector([V, W], kind="nest")

assemble_matrix(A, J)
A.assemble()
# -

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
#
# Only one of the four Jacobian blocks depends on the solution: the
# $(1,0)$ block $-(N + \lambda K)$, through $f''(c)$ in $N$. The other
# three are $M$, $\theta \Delta t\, K$ and $M$, which is why the matrix
# was assembled in full just above. `J10` is that block; the callback
# rebuilds it and leaves the rest of the nest as it is.


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

    # Add contributions made to entries owned by other processes. SNES
    # reads only owned entries, so the ghosts are not updated
    ghost_update(b, PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE)


def jacobian(_snes: PETSc.SNES, x: PETSc.Vec, J_mat: PETSc.Mat, _P: PETSc.Mat) -> None:
    """Assemble the solution-dependent Jacobian block at ``x``."""
    ghost_update(x, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)
    assign(x, [c, mu])

    A10 = J_mat.getNestSubMatrix(1, 0)
    A10.zeroEntries()
    assemble_matrix(A10, J10)

    # Assembling the nest assembles each block, A10 included, and marks
    # the nest as changed, so PCFIELDSPLIT sees the new block
    J_mat.assemble()


snes = PETSc.SNES().create(msh.comm)  # type: ignore[arg-type]
snes.setFunction(residual, b)
snes.setJacobian(jacobian, A, None)
# -

# The solver options. The Schur part of the preconditioner is a fixed
# linear operator -- one AMG V-cycle per factor -- but the $(0,0)$ block
# is solved by CG to a relative tolerance, which is not, so the outer
# method has to be flexible and is FGMRES. Giving the $(0,0)$ block a
# fixed number of Chebyshev iterations instead would make the whole
# preconditioner linear and let plain GMRES be used, for the same
# iteration count and about 5% less memory; a tolerance is kept here
# because it needs no mesh-dependent iteration count to be chosen.
# `schur_fact_type upper` needs one solve with the $(0,0)$ block per
# application rather than the two a full factorisation needs, for the
# same iteration count here.
#
# Both inner tolerances are deliberately loose. A linear solve only has
# to return a Newton direction good enough not to cost Newton steps, so
# `ksp_rtol` is set three orders above the nonlinear tolerance; and the
# $(0,0)$ solve sits inside a preconditioner, where tightening it beyond
# `1e-2` buys no outer iterations and costs time.

# +
# BoomerAMG where PETSc has hypre, GAMG otherwise. hypre is also passed
# over in a complex build, where its parallel coarsening corrupts memory;
# the problem is real-valued whatever PETSc's scalar type is
complex_scalars = np.issubdtype(PETSc.ScalarType, np.complexfloating)
amg = "hypre" if PETSc.Sys().hasExternalPackage("hypre") and not complex_scalars else "gamg"

# The tolerances are for double precision. In single precision they are
# raised to a fixed multiple of machine epsilon, which keeps them
# reachable
eps = float(np.finfo(default_real_type).eps)
petsc_options: dict[str, str | float | int] = {
    "snes_type": "newtonls",
    "snes_linesearch_type": "basic",
    "snes_rtol": max(1.0e-9, 1.0e3 * eps),
    "ksp_type": "fgmres",
    "ksp_rtol": max(1.0e-6, 1.0e2 * eps),
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
    "pc_fieldsplit_schur_fact_type": "upper",
    # The (0,0) block is a mass matrix: CG with Jacobi is mesh independent
    "fieldsplit_c_ksp_type": "cg",
    "fieldsplit_c_ksp_rtol": 1.0e-2,
    "fieldsplit_c_pc_type": "jacobi",
    "fieldsplit_mu_ksp_type": "preonly",
    # One AMG V-cycle per application of (M + aK)^{-1}
    "schur_ksp_type": "preonly",
    "schur_pc_type": amg,
}
if amg == "hypre":
    # BoomerAMG's defaults suit two dimensions. These are its documented
    # three-dimensional settings; they keep the coarse operators sparse,
    # which is most of the cost of a V-cycle here
    petsc_options |= {
        "schur_pc_hypre_type": "boomeramg",
        "schur_pc_hypre_boomeramg_strong_threshold": 0.7,
        "schur_pc_hypre_boomeramg_coarsen_type": "HMIS",
        "schur_pc_hypre_boomeramg_interp_type": "ext+i",
        "schur_pc_hypre_boomeramg_P_max": 4,
        "schur_pc_hypre_boomeramg_truncfactor": 0.3,
        "schur_pc_hypre_boomeramg_agg_nl": 1,
    }

# The options go into PETSc's global options database under a prefix
# that only this solver uses, and are removed once the solver is set up
prefix = "demo_cahn-hilliard-block_"
opts = PETSc.Options()
for key, value in petsc_options.items():
    opts[prefix + key] = value
snes.setOptionsPrefix(prefix)
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
ksp = snes.getKSP()
ksp.setOperators(A, A)
pc = ksp.getPC()
index_sets = A.getNestISs()
pc.setFieldSplitIS(("c", index_sets[0][0]), ("mu", index_sets[0][1]))
pc.setUp()

schur_pc = pc.getFieldSplitSubKSP()[1].getPC()
schur_pc.setType("python")
schur_pc.setPythonContext(MatchedSchur(S_mat, A.getNestSubMatrix(1, 1), prefix + "schur_"))
schur_pc.setUp()

for key in petsc_options:
    del opts[prefix + key]
# -

# `SNES` can record the Krylov iterations each Newton step takes, which
# is the quantity that should stay flat under refinement. `reset=True`
# clears the record at the start of each solve.

snes.setConvergenceHistory(reset=True)

# Step in time, reporting the work each step took. The solution of the
# previous step is kept as well as the current one, so that each solve
# can start from an extrapolation rather than from a standstill.

# +
out = Path("demo_ch_block")
out.mkdir(parents=True, exist_ok=True)
file = io.XDMFFile(msh.comm, out / "output.xdmf", "w")
file.write_mesh(msh)
file.write_function(c, 0.0)  # the initial condition, before any step

num_steps = 3 if ci else 30
t = 0.0
newton_total, krylov_total = 0, 0
c_prev, mu_prev = np.zeros_like(c.x.array), np.zeros_like(mu.x.array)
for step in range(num_steps):
    t += dt
    c_prev[:], mu_prev[:] = c0.x.array, mu0.x.array
    c0.x.array[:] = c.x.array
    mu0.x.array[:] = mu.x.array

    # Start the solve from a linear extrapolation of the last two steps
    # rather than from the last one. Two vector operations, and Newton
    # needs roughly one iteration per step fewer. It takes two solved
    # steps to extrapolate from: only $c$ is given at $t = 0$, and
    # $\mu$ there is the zero it was initialised with, so neither is
    # extrapolated until the third step
    if step > 1:
        c.x.array[:] = 2.0 * c0.x.array - c_prev
        mu.x.array[:] = 2.0 * mu0.x.array - mu_prev

    assign([c, mu], x)
    snes.solve(None, x)
    assign(x, [c, mu])
    c.x.scatter_forward()
    mu.x.scatter_forward()

    if snes.getConvergedReason() < 0:  # type: ignore[operator]
        raise RuntimeError(f"Step {step} did not converge: {snes.getConvergedReason()}")
    num_newton = snes.getIterationNumber()
    num_krylov = snes.getLinearSolveIterations()
    newton_total += num_newton
    krylov_total += num_krylov
    if msh.comm.rank == 0:
        # Entry 0 of the history is the initial residual, before any step
        per_step = " ".join(str(k) for k in snes.getConvergenceHistory()[1][1:])
        print(
            f"step {step + 1:3d}  t = {t:.2e}  Newton {num_newton}  "
            f"Krylov {num_krylov} ({per_step})"
        )
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
    if newton_total > 0:
        print(f"Krylov iterations per Newton:  {krylov_total / newton_total:.1f}")
# -

# Measured on this problem, running the thirty steps above on one
# process and refining the mesh with everything else held fixed. The
# comparison is against PETSc's own Schur complement approximation,
# `-pc_fieldsplit_schur_precondition selfp`, which builds
# $A_{10} \, {\rm diag}(A_{00})^{-1} A_{01}$ automatically and is given
# the same multigrid settings; everything else is identical.
#
# The matched approximation, as the demo runs it:
#
# | per side | unknowns | its | time | memory |
# | --- | --- | --- | --- | --- |
# | 16 | 9 826   | 13.8 |   2.7 s |  161 MB |
# | 24 | 31 250  | 15.1 |   7.8 s |  217 MB |
# | 32 | 71 874  | 15.1 |  20.4 s |  328 MB |
# | 48 | 235 298 | 16.0 |  74.3 s |  798 MB |
# | 64 | 549 250 | 17.2 | 185.8 s | 1690 MB |
#
# and `selfp` over the same mesh sequence:
#
# | per side | unknowns | its | time | memory |
# | --- | --- | --- | --- | --- |
# | 16 | 9 826   |  9.0 |   3.5 s |  171 MB |
# | 24 | 31 250  | 10.1 |  12.8 s |  269 MB |
# | 32 | 71 874  | 11.6 |  33.2 s |  457 MB |
# | 48 | 235 298 | 17.9 | 141.0 s | 1235 MB |
# | 64 | 549 250 | 25.7 | 436.5 s | 2845 MB |
#
# where "its" is Krylov iterations per Newton step and the memory
# columns are peak resident set size for the whole run. Both solvers
# take the same Newton steps, 116 to 118, and agree to twelve
# significant figures.
#
# The matched approximation needs more iterations than `selfp` on the
# coarsest mesh, 13.8 against 9.0, and fewer on the finest, 17.2 against
# 25.7. Its iteration count rises by a quarter over a fifty-six-fold
# increase in problem size, consistent with the mesh-independent
# estimate above, whereas the automatic approximation nearly triples;
# the two cross at about $2 \times 10^{5}$ unknowns. In wall-clock terms
# the matched approximation is ahead at every size, by 30% at the
# coarsest mesh and by a factor of two and a third at the finest,
# because an application of it is cheaper even where it converges more
# slowly.
#
# The cost of an application explains the rest. $M + aK$ is fixed, so
# its multigrid hierarchy is built once for the whole run, whereas
# $A_{10} \, {\rm diag}(A_{00})^{-1} A_{01}$ contains $f''(c)$ and has
# to be re-formed and re-coarsened at every Newton step. That product
# is also denser than $M + aK$, which is where the extra memory goes.
#
# The demo is not threaded, but it is parallel. The 549 250-unknown
# case takes 185.8 s on one process, 117.3 s on two and 69.0 s on four.
# The iteration count is unchanged; the Newton count moves by a few per
# cent only because the random initial condition is drawn per process
# and so depends on how many there are.
#
# The usable range of $\Delta t$ and $\lambda$ here is set by the
# nonlinear problem rather than by the preconditioner: at
# $\Delta t = 5 \times 10^{-5}$, or at $\lambda = 10^{-3}$ with this time
# step, Newton's method fails with a direct solver too.
#
# ## References
# ```{bibliography}
# :filter: docname in docnames
# ```
