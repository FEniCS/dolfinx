# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
# ---

# # The eikonal equation on a Möbius strip
#
# This demo computes the distance to the boundary of a Möbius strip by
# solving the eikonal equation with the latent variable proximal point
# (LVPP) algorithm of {cite:t}`eikonal-dokken2025lvpp`. It illustrates
# how to
#
# * solve a problem with a pointwise inequality constraint, here
#   $|\nabla_\Omega u| \le 1$, without ever forming the constraint,
# * use an $H(\mathrm{div})$-conforming latent variable on a surface
#   that cannot be oriented.
#
# ```{admonition} Difference from the paper
# :class: note
# {cite:t}`eikonal-dokken2025lvpp` solves the eikonal equation in this
# Raviart-Thomas mixed form on flat surfaces, and on manifolds without a
# Raviart-Thomas space. What is new here is the combination: a
# Raviart-Thomas latent variable on a manifold, and one that cannot be
# oriented at that.
# ```
#
# ```{admonition} Download sources
# :class: download
# * {download}`Python script <./demo_mobius_eikonal.py>`
# * {download}`Jupyter notebook <./demo_mobius_eikonal.ipynb>`
# ```
#
# ## Equation and problem definition
#
# The distance to the boundary of a domain $\Omega$ solves the eikonal
# equation
#
# $$
#   |\nabla_\Omega u| = 1 \text{ in } \Omega, \qquad
#   u = 0 \text{ on } \partial\Omega,
# $$
#
# where $\nabla_\Omega$ is the gradient along the surface. It is the
# limit of a problem with an inequality constraint: among all $u$ with
# $|\nabla_\Omega u| \le 1$ and $u = 0$ on the boundary, the distance
# function is the largest, so it maximises $\int_\Omega f u$ for
# $f \equiv 1$.
#
# ### The latent variable
#
# LVPP never imposes the constraint directly. It introduces a *latent*
# variable $\psi$, unconstrained, and recovers the gradient from it
# through a pointwise map whose range is exactly the feasible set. For
# the unit ball that map is
#
# $$
#   \nabla_\Omega u = \frac{\psi}{\sqrt{1 + |\psi|^2}},
# $$
#
# so $|\nabla_\Omega u| < 1$ holds for every $\psi$, and the constraint
# becomes active only in the limit $|\psi| \to \infty$. The
# non-smoothness of the constraint is thereby traded for the growth of
# $\psi$, which is what makes each subproblem a smooth one that Newton's
# method handles.
#
# The proximal point iteration supplies that limit. Given the previous
# latent variable $\psi^{k-1}$ and a step size $\alpha_k > 0$, the
# $k$-th subproblem is to find $(u^k, \psi^k)$ with
#
# $$
# \begin{aligned}
#   \nabla_\Omega \cdot \psi^k
#     &= \nabla_\Omega \cdot \psi^{k-1} - \alpha_k f, \\
#   \nabla_\Omega u^k
#     &= \frac{\psi^k}{\sqrt{1 + |\psi^k|^2}}.
# \end{aligned}
# $$
#
# Increasing $\alpha_k$ drives $|\psi^k|$ up and $|\nabla_\Omega u^k|$
# towards one. The iteration stops when $u$ settles.
#
# ### Mixed form
#
# Taking $\psi$ in an $H(\mathrm{div})$-conforming space and $u$ in a
# discontinuous Lagrange space, and testing the second equation against
# $\tau$ with an integration by parts that leaves no boundary term
# because $u = 0$ on $\partial\Omega$, the subproblem reads: find
# $(u, \psi) \in Q \times V$ with
#
# $$
# \begin{aligned}
#   \int_\Omega (\nabla_\Omega \cdot \psi) \, v ~\mathrm{d}x
#     &= \int_\Omega (\nabla_\Omega \cdot \psi^{k-1}) \, v ~\mathrm{d}x
#      - \alpha_k \int_\Omega f \, v ~\mathrm{d}x, \\
#   \int_\Omega u \, (\nabla_\Omega \cdot \tau) ~\mathrm{d}x
#     + \int_\Omega \frac{\psi \cdot \tau}{\sqrt{1 + |\psi|^2}}
#       ~\mathrm{d}x &= 0,
# \end{aligned}
# $$
#
# for all $(v, \tau) \in Q \times V$. The Dirichlet condition on $u$ is
# natural here, so no boundary condition is imposed on the space.
#
# Raviart-Thomas is mapped by the contravariant Piola map, which on a
# surface needs the two cells of each facet to agree on its normal. On a
# Möbius strip no orientation of the cells achieves that, so the facets
# where they disagree carry a sign instead; see {doc}`demo_mobius`.
#
# ## Implementation

# +

import contextlib

from mpi4py import MPI

import numpy as np

import basix
import basix.ufl
import dolfinx
import dolfinx.fem.petsc
import ufl

# -

# The strip is `n` cells around and `m` across, of half-width `width`,
# on a mesh of degree `geometry_degree`. The latent variable is of
# degree `degree` and $u$ one lower.

n, m = 48, 8
geometry_degree = 2
degree = 3
width = 0.92


# ### The mesh
#
# The strip is built from its parametrisation, the half twist being that
# the column closing the ring is column $0$ read upside down. Equispaced
# Lagrange nodes lie on a lattice of step $1/k$, so taking every node
# from the $k$-times finer parameter grid makes neighbouring cells share
# the nodes on their common edge, which is what keeps the geometry
# continuous. See {doc}`demo_mobius`, which builds the same surface.


def surface_point(t, s):
    """Map the parameters onto the strip."""
    rho = 1 + s * np.cos(t / 2)
    z = s * np.sin(t / 2)
    return np.column_stack([rho * np.cos(t), rho * np.sin(t), z])


def mobius_mesh(n, m, width, k):
    """A Möbius strip, ``n`` cells around and ``m`` across."""
    dtype = dolfinx.default_real_type
    lattice = basix.create_element(
        basix.ElementFamily.P, basix.CellType.triangle, k, basix.LagrangeVariant.equispaced
    ).points
    if MPI.COMM_WORLD.rank == 0:
        ni, nj = k * n, k * m

        def node(i, j):
            return i * (nj + 1) + j if i < ni else nj - j  # the half twist

        i_grid = np.repeat(np.arange(ni), nj + 1)
        j_grid = np.tile(np.arange(nj + 1), ni)
        x = surface_point(2 * np.pi * i_grid / ni, -width + 2 * width * j_grid / nj)
        cells = []
        for i in range(n):
            for j in range(m):
                a = np.array([k * i, k * j])
                b = np.array([k * (i + 1), k * j])
                c = np.array([k * i, k * (j + 1)])
                d = np.array([k * (i + 1), k * (j + 1)])
                for p0, p1, p2 in ((a, b, c), (b, d, c)):
                    ij = np.rint(
                        p0 + np.outer(lattice[:, 0], p1 - p0) + np.outer(lattice[:, 1], p2 - p0)
                    ).astype(np.int64)
                    cells.append([node(int(p), int(q)) for p, q in ij])
        cells = np.asarray(cells, dtype=np.int64)
    else:
        x = np.zeros((0, 3))
        cells = np.zeros((0, len(lattice)), dtype=np.int64)
    domain = ufl.Mesh(
        basix.ufl.element(
            "Lagrange", "triangle", k, basix.LagrangeVariant.equispaced, shape=(3,), dtype=dtype
        )
    )
    return dolfinx.mesh.create_mesh(MPI.COMM_WORLD, cells, domain, x.astype(dtype))


# +
mesh = mobius_mesh(n, m, width, geometry_degree)
mesh.topology.create_facet_orientations()

V = dolfinx.fem.functionspace(mesh, ("RT", degree))
Q = dolfinx.fem.functionspace(mesh, ("DG", degree - 1))
W = ufl.MixedFunctionSpace(Q, V)

u = dolfinx.fem.Function(Q, name="u")
psi = dolfinx.fem.Function(V, name="psi")
v, tau = ufl.TestFunctions(W)

# The previous iterate
u0 = dolfinx.fem.Function(Q)
psi0 = dolfinx.fem.Function(V)
# -

# ### The subproblem
#
# Each step solves the two equations above as one nonlinear system, with
# the step size $\alpha_k$ carried by a {py:class}`Constant
# <dolfinx.fem.Constant>` so that the forms are compiled once.

# +
dx = ufl.Measure("dx", domain=mesh)
f = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(1))
alpha = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(1))

latent_map = 1 / ufl.sqrt(1 + ufl.dot(psi, psi))
F = ufl.inner(ufl.div(psi) - ufl.div(psi0), v) * dx + alpha * ufl.inner(f, v) * dx
F += ufl.inner(u, ufl.div(tau)) * dx + latent_map * ufl.inner(psi, tau) * dx

F_blocked: list[ufl.Form] = ufl.extract_blocks(F)  # type: ignore[assignment]
tol = max(1e-5, 1000 * float(np.finfo(dolfinx.default_real_type).eps))
problem = dolfinx.fem.petsc.NonlinearProblem(
    F_blocked,
    [u, psi],
    bcs=[],
    petsc_options={
        "snes_type": "newtonls",
        "snes_rtol": tol,
        "snes_atol": tol,
        "snes_max_it": 100,
        "snes_error_if_not_converged": True,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    },
    petsc_options_prefix="lvpp_step_",
)
# -

# ### Output
#
# VTX builds its points from the dofmap, so neither a Raviart-Thomas
# function nor the gradient can be written directly. Both are
# interpolated into discontinuous Lagrange spaces of the degree of the
# geometry first.

# +
S = dolfinx.fem.functionspace(mesh, ("DG", mesh.geometry.cmaps[0].degree))
u_out = dolfinx.fem.Function(S, name="u")

S_vec = dolfinx.fem.functionspace(mesh, ("DG", mesh.geometry.cmaps[0].degree, (mesh.geometry.dim,)))
grad_out = dolfinx.fem.Function(S_vec, name="grad_u")
psi_out = dolfinx.fem.Function(S_vec, name="psi")
grad_expr = dolfinx.fem.Expression(ufl.grad(u), S_vec.element.interpolation_points)
# -

# ### The proximal point loop
#
# The step size is increased geometrically, up to a cap, and the
# iteration stops once $u$ stops moving, measured in $L^2$.

# +
change = dolfinx.fem.form(ufl.inner(u - u0, u - u0) * dx)
newton_iterations = []

with contextlib.ExitStack() as stack:
    writers = []
    if dolfinx.has_adios2:
        writers = [
            stack.enter_context(dolfinx.io.VTXWriter(mesh.comm, filename, functions, engine="BP4"))
            for filename, functions in [
                ("mobius_eikonal_u.bp", [u_out]),
                ("mobius_eikonal_psi.bp", [grad_out, psi_out]),
            ]
        ]

    for i in range(1, 100):
        alpha.value = min(2**i, 10)
        problem.solve()
        newton_iterations.append(problem.solver.getIterationNumber())

        diff = np.sqrt(abs(mesh.comm.allreduce(dolfinx.fem.assemble_scalar(change), op=MPI.SUM)))
        if mesh.comm.rank == 0:
            print(
                f"LVPP {i}: alpha={alpha.value.real:g}, "
                f"Newton iterations={newton_iterations[-1]}, |u - u_prev|={diff:.3e}"
            )

        u0.x.array[:] = u.x.array
        psi0.x.array[:] = psi.x.array

        u_out.interpolate(u)
        grad_out.interpolate(grad_expr)
        psi_out.interpolate(psi)
        for writer in writers:
            writer.write(i)

        if diff < 5 * tol:
            break


if mesh.comm.rank == 0:
    print(f"{i} LVPP iterations, {sum(newton_iterations)} Newton iterations in total")
# -

# The constraint is satisfied throughout, and becomes active away from
# the kink of the distance function. We report the mean slope, which
# approaches one as the iteration converges.

# +
one = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(1))
grad_norm = ufl.sqrt(ufl.inner(ufl.grad(u), ufl.grad(u)))


def integrate(expr):
    """Integrate a real-valued expression over the surface."""
    local = dolfinx.fem.assemble_scalar(dolfinx.fem.form(expr * dx))
    return abs(mesh.comm.allreduce(local, op=MPI.SUM))


mean_slope = integrate(grad_norm) / integrate(one)
if mesh.comm.rank == 0:
    print(f"mean |grad(u)| = {mean_slope:.6f}  (1 in the limit)")
# -

# ## References
# ```{bibliography}
#    :filter: cited
#    :labelprefix:
#    :keyprefix: eikonal-
# ```
