# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
# ---

# # Mixed Poisson on a Möbius strip
#
# This demo solves the Poisson equation in mixed form on a Möbius strip,
# a surface that cannot be oriented.
#
# A Möbius strip has no consistent orientation at all, which an
# $H(\mathrm{div})$-conforming space would seem to need. It is made
# conforming one facet at a time instead. The same problem is solved
# first on a cylinder, so that the two can be compared directly.
#
# ```{admonition} Download sources
# :class: download
# * {download}`Python script <./demo_mobius.py>`
# * {download}`Jupyter notebook <./demo_mobius.ipynb>`
# ```
#
# ## Equation and problem definition
#
# On a surface $\Omega$ with boundary, the Poisson equation in mixed
# form is: find the flux $\sigma$ and the potential $u$ with
#
# $$
# \begin{aligned}
#   \sigma - \nabla_\Omega u &= 0, \\
#   \nabla_\Omega \cdot \sigma &= g,
# \end{aligned}
# $$
#
# and $u = 0$ on $\partial\Omega$. Multiplying by test functions $\tau$
# and $v$ and integrating the first equation by parts moves the
# derivative onto $u$ and leaves $\int_{\partial\Omega} u \,
# \tau \cdot n ~\mathrm{d}s$, which vanishes because $u = 0$ there. The
# condition is natural, so it needs no boundary condition on the
# discrete space, and the weak form is
#
# $$
# \begin{aligned}
#   \int_\Omega \sigma \cdot \tau ~\mathrm{d}x
#   + \int_\Omega u \, \nabla_\Omega \cdot \tau ~\mathrm{d}x
#   &= 0, \\
#   \int_\Omega (\nabla_\Omega \cdot \sigma) \, v ~\mathrm{d}x
#   &= \int_\Omega g \, v ~\mathrm{d}x,
# \end{aligned}
# $$
#
# for all $\tau$ and $v$. As on the sphere, $\sigma$ is sought in an
# $H(\mathrm{div})$-conforming Raviart-Thomas space and $u$ in a
# discontinuous Lagrange space. Unlike the sphere, the strip has a
# boundary, so no global constraint and no real space are needed.
#
# ### The surfaces, and an exact solution
#
# The strip is parametrised by $t \in [0, 2\pi)$ along it and
# $s \in [-w, w]$ across it,
#
# $$
#   x = \rho \cos t, \quad
#   y = \rho \sin t, \quad
#   z = s \sin(t/2), \quad
#   \rho = 1 + s \cos(t/2).
# $$
#
# Going once around sends $s$ to $-s$: that half twist is what makes the
# surface non-orientable. The control is the cylinder $\rho = 1$,
# $z = s$, which has the same boundary curves but no twist.
#
# The last two relations give $s\cos(t/2) = \rho - 1$ and
# $s\sin(t/2) = z$, so on either surface
#
# $$
#   s^2 = \left(\sqrt{x^2 + y^2} - 1\right)^2 + z^2 ,
# $$
#
# a formula in the ambient Cartesian coordinates alone. Anything written
# in those coordinates is single-valued on the strip automatically, so
# we can sidestep the question of invariance under the identification
# $(t, s) \mapsto (t + 2\pi, -s)$ entirely, and take
#
# $$
#   u = w^2 - s^2 .
# $$
#
# This vanishes exactly on the boundary curves $s = \pm w$, matching the
# natural condition above. On the cylinder it reduces to $u = w^2 - z^2$
# with $\nabla_\Omega \cdot \nabla_\Omega u = -2$; the demo prints the
# mean of the computed right-hand side over the cylinder as a check.
#
# ```{admonition} grad of a coordinate expression is the ambient gradient
# :class: important
# In UFL $\nabla x$ is the identity, so `grad` applied to an expression
# built from {py:class}`SpatialCoordinate <ufl.SpatialCoordinate>` gives
# the gradient in the surrounding space, **not** along the surface. The
# two differ on a manifold, so the surface operators must be built
# explicitly by projecting with $P = I - n \otimes n$:
#
# $$
#   \nabla_\Omega f = P \nabla f, \qquad
#   \nabla_\Omega \cdot v = \operatorname{tr}\left(P \nabla v\right).
# $$
#
# This is why {doc}`demo_manifold` projects its exact flux too. Note
# that `grad` and `div` applied to a *finite element function* already
# are the surface operators, since those are differentiated through the
# reference cell; only coordinate expressions need the projection.
#
# $P$ is unchanged by flipping $n$, so the exact solution is well
# defined even on the strip, where no global normal exists.
# ```
#
# ## Implementation

# +
from mpi4py import MPI

import numpy as np

import basix.ufl
import dolfinx
import dolfinx.fem.petsc
import ufl

# -

# The degree $k$ of the Raviart-Thomas space, the half-width $w$, and
# the resolutions: cells around the surface and across it.

order = 2
geometry_degree = 2
width = 0.4
resolutions = [(24, 2), (48, 4), (96, 8)]


# ### The mesh
#
# Both surfaces are curved, so a piecewise-flat mesh would commit a
# geometric error of its own and muddy the comparison. The mesh is built
# at degree `geometry_degree` instead, placing every node exactly on the
# surface. The area of the cylinder then converges as $O(h^{2k})$ rather
# than $O(h^2)$, so what is left in the tables below is the
# discretisation, not the geometry.
#
# Equispaced Lagrange nodes sit on a lattice of step $1/k$, so every node
# of every cell is a point of the $k$-times finer parameter grid. Taking
# the nodes from that grid makes neighbouring cells share the nodes on
# their common edge automatically, which is what keeps the geometry
# continuous. The half twist lives in `node`: the column closing the ring
# is column $0$ read upside down, so the last band of cells joins the
# first with $s$ reversed.


def surface_point(t, s, twist):
    """Map the parameters onto the strip, or onto the cylinder."""
    rho = 1 + s * np.cos(t / 2) if twist else np.ones_like(s)
    z = s * np.sin(t / 2) if twist else s
    return np.column_stack([rho * np.cos(t), rho * np.sin(t), z])


def ring_mesh(n, m, twist):
    """A Möbius strip if ``twist``, otherwise a cylinder."""
    dtype = dolfinx.default_real_type
    k = geometry_degree
    lattice = basix.create_element(
        basix.ElementFamily.P, basix.CellType.triangle, k, basix.LagrangeVariant.equispaced
    ).points
    if MPI.COMM_WORLD.rank == 0:
        ni, nj = k * n, k * m

        def node(i, j):
            if i < ni:
                return i * (nj + 1) + j
            return (nj - j) if twist else j  # the half twist

        i_grid = np.repeat(np.arange(ni), nj + 1)
        j_grid = np.tile(np.arange(nj + 1), ni)
        x = surface_point(2 * np.pi * i_grid / ni, -width + 2 * width * j_grid / nj, twist)
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
    # The normal-jump diagnostic below is an interior facet integral, so
    # facets on a process boundary need their neighbouring cell.
    return dolfinx.mesh.create_mesh(
        MPI.COMM_WORLD,
        cells,
        domain,
        x.astype(dtype),
        ghost_mode=dolfinx.mesh.GhostMode.shared_facet,
    )


# ### Signing the facets
#
# A Raviart-Thomas basis function is pushed forward by the contravariant
# Piola map. On a manifold that map scales by the pseudo-determinant
# $\sqrt{\det(J^\top J)}$ of the Jacobian, which is positive by
# construction and so carries no sign. The sign comes instead from the
# order of a cell's vertices, which fixes the direction of its normal,
# and two cells sharing an edge are conforming only if their normals
# agree there.
#
# {py:meth}`create_facet_orientations
# <dolfinx.mesh.Topology.create_facet_orientations>` compares the two
# cells of every facet and, where they disagree, negates the
# degrees-of-freedom of one of them there. Of a cell's basis functions
# only those belonging to a facet have a non-zero normal trace on it, so
# that restores the agreement across that facet and disturbs nothing
# else. Gridap {cite}`mobius-verdugo2022gridap` signs its Raviart-Thomas
# spaces the same way.
#
# Because it looks at one facet at a time and never at the surface as a
# whole, the same call serves both meshes. The cylinder, which could be
# oriented, is treated no differently from the strip, which could not:
# the question of whether a consistent orientation exists never arises.


def solve(n, m, twist):
    """Solve on one mesh: solution, errors and mean Laplacian."""
    mesh = ring_mesh(n, m, twist)
    mesh.topology.create_facet_orientations()

    V = dolfinx.fem.functionspace(mesh, ("RT", order))
    Q = dolfinx.fem.functionspace(mesh, ("DG", order - 1))
    W = ufl.MixedFunctionSpace(V, Q)
    (sigma, u) = ufl.TrialFunctions(W)
    (tau, v) = ufl.TestFunctions(W)

    # Surface operators for the exact solution, which is written in
    # ambient coordinates (see the note above).
    x = ufl.SpatialCoordinate(mesh)
    normal = ufl.CellNormal(mesh)
    P = ufl.Identity(3) - ufl.outer(normal, normal)
    rho = ufl.sqrt(x[0] ** 2 + x[1] ** 2)
    u_exact = width**2 - ((rho - 1) ** 2 + x[2] ** 2)
    sigma_exact = ufl.dot(P, ufl.grad(u_exact))
    g = ufl.tr(ufl.dot(P, ufl.grad(sigma_exact)))

    a = (ufl.inner(sigma, tau) + ufl.inner(ufl.div(sigma), v) + ufl.inner(u, ufl.div(tau))) * ufl.dx
    a_blocked: list[list[ufl.Form | None]] = ufl.extract_blocks(a)  # type: ignore[assignment]
    L_blocked: list[ufl.Form | ufl.ZeroBaseForm] = [
        ufl.ZeroBaseForm((tau,)),
        ufl.inner(g, v) * ufl.dx,
    ]

    problem = dolfinx.fem.petsc.LinearProblem(
        a_blocked,
        L_blocked,
        bcs=[],
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "ksp_error_if_not_converged": True,
        },
        petsc_options_prefix="mobius_",
        kind="mpi",
    )
    sigma_h, u_h = problem.solve()

    # The normal component of a conforming flux is continuous, so this
    # jump is zero up to rounding. `FacetNormal` is built from each
    # cell's own Jacobian and the reference normal, with no reference to
    # a global orientation, so it is a trustworthy diagnostic here.
    jump = ufl.jump(sigma_h, ufl.FacetNormal(mesh))

    def norm(expr, measure=ufl.dx):
        form = dolfinx.fem.form(ufl.inner(expr, expr) * measure)
        return np.sqrt(abs(mesh.comm.allreduce(dolfinx.fem.assemble_scalar(form), op=MPI.SUM)))

    def integral(expr):
        form = dolfinx.fem.form(expr * ufl.dx(domain=mesh))
        return mesh.comm.allreduce(dolfinx.fem.assemble_scalar(form), op=MPI.SUM)

    errors = {
        "L2(u)": norm(u_h - u_exact),
        "L2(sigma)": norm(sigma_h - sigma_exact),
        "normal jump": norm(jump, ufl.dS),
        "mean Laplacian": integral(g) / integral(1.0),
    }
    return mesh, sigma_h, u_h, errors


# ### Results
#
# Both surfaces converge at the same rate, and on both the normal jump
# sits at rounding: the space really is $H(\mathrm{div})$-conforming,
# non-orientable or not. The jump is the quantity to watch, since it is
# what conformity constrains, and unlike a discretisation error it does
# not shrink with the mesh when something is wrong: an unsigned facet
# leaves a jump of roughly the size of the flux itself, however fine the
# mesh.
#
# The error in $\sigma$ is larger on the strip than on the cylinder
# simply because the strip is the harder geometry, not because anything
# is left uncorrected.

# +
columns = ["L2(u)", "L2(sigma)", "normal jump"]
for label, twist in [("cylinder (orientable)", False), ("Möbius strip", True)]:
    if MPI.COMM_WORLD.rank == 0:
        print(f"\n{label}")
        print(f"{'cells':>8}" + "".join(f"{c:>16}" for c in columns))
    errors: dict[str, float] = {}
    for n, m in resolutions:
        errors = solve(n, m, twist)[3]
        if MPI.COMM_WORLD.rank == 0:
            print(f"{2 * n * m:>8}" + "".join(f"{errors[c]:>16.3e}" for c in columns))
    if not twist and MPI.COMM_WORLD.rank == 0:
        print(f"mean surface Laplacian of u: {errors['mean Laplacian']:.4f} (exact: -2)")
# -

# ### Visualisation
#
# VTX builds its points from the dofmap, so a Raviart-Thomas function
# cannot be written directly. We interpolate both fields into
# discontinuous Lagrange spaces of the same degree first: they then
# share a base element and differ only in block size, which lets one
# writer take them together. Opening `mobius.bp` shows $\sigma_h$
# jumping across the seam.

if dolfinx.has_adios2:
    mesh, sigma_h, u_h, _ = solve(*resolutions[-1], True)
    S_out = dolfinx.fem.functionspace(mesh, ("DG", order, (3,)))
    sigma_out = dolfinx.fem.Function(S_out, name="sigma")
    sigma_out.interpolate(sigma_h)

    U_out = dolfinx.fem.functionspace(mesh, ("DG", order))
    u_out = dolfinx.fem.Function(U_out, name="u")
    u_out.interpolate(u_h)

    with dolfinx.io.VTXWriter(mesh.comm, "mobius.bp", [sigma_out, u_out]) as writer:
        writer.write(0.0)

# ## References
# ```{bibliography}
#    :filter: cited
#    :labelprefix:
#    :keyprefix: mobius-
# ```
