# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
# ---

# # Mixed Poisson on the sphere with a rotated $H(\mathrm{curl})$ flux
#
# This demo solves the same problem as {doc}`demo_manifold`, but with
# the flux in an $H(\mathrm{curl})$-conforming Nédélec space, rotated by
# $\pi/2$ in the tangent plane, which recovers the $H(\mathrm{div})$
# space. This is the basis of vorticity-type formulations. It illustrates
# how to
#
# * compute a consistent orientation of the cells of a surface with
#   {py:func}`compute_cell_orientations
#   <dolfinx.mesh.compute_cell_orientations>`,
# * turn the orientation into a consistent unit normal from
#   {py:class}`ufl.CellNormal`,
# * use that normal to build the surface curl of a Nédélec field.
#
# ```{admonition} Download sources
# :class: download
# * {download}`Python script <./demo_manifold_vorticity.py>`
# * {download}`Jupyter notebook <./demo_manifold_vorticity.ipynb>`
# ```
#
# ## Problem definition
#
# As in {doc}`demo_manifold`, $\Omega$ is the unit sphere, and given $g$
# and a constant $h$ we seek $u \in H^1(\Omega)$ and $r \in \mathbb{R}$
# with $\nabla \cdot \nabla u + r = g$ and
# $\int_{\Omega} u ~\mathrm{d}x = h.$ The sphere has no boundary, so
# integrating the equation over $\Omega$ gives $r$ as the mean of $g;$ the
# constant $r$ makes the problem solvable for any $g,$ and $h$ fixes the
# constant in $u.$
#
# We write the problem in mixed form, with the surface gradient
# $q = \nabla u$ as an unknown: find $q \in H(\mathrm{div}; \Omega),$
# $u \in L^2(\Omega)$ and $r \in \mathbb{R}$ such that
#
# $$
# \begin{aligned}
#   q - \nabla u &= 0, \\
#   \nabla \cdot q + r &= g, \\
#   \int_{\Omega} u ~\mathrm{d}x &= h.
# \end{aligned}
# $$ (mixed-form)
#
# On a manifold, here a surface in 3D with unit normal $n,$ Nédélec and
# Raviart-Thomas fields are tangential to the surface. We write the
# rotation of a tangential field $w$ by $\pi/2$ in the tangent plane as
#
# $$
# \begin{aligned}
#   R(w) := n \times w,
# \end{aligned}
# $$
#
# where $\times$ is the cross product. $R$ preserves inner products, and
# applying $R$ twice reverses a field:
#
# $$
# \begin{aligned}
#   R(a) \cdot R(b) &= a \cdot b, &
#   R(R(w)) &= -w.
# \end{aligned}
# $$ (rotation-properties)
#
# ```{admonition} Lemma
# Cell by cell, $R$ maps the Nédélec space of the first kind and degree
# $k$ onto the Raviart-Thomas space of degree $k.$ The degrees follow
# Basix's numbering, in which both lowest-order spaces have degree 1.
# ```
#
# ```{admonition} Informal proof
# :class: dropdown
# The degrees-of-freedom of an $H(\mathrm{curl})$-conforming Nédélec
# element are moments of the tangential component along each edge, which
# is continuous between cells. The degrees-of-freedom of an
# $H(\mathrm{div})$-conforming Raviart-Thomas element are moments of the
# normal component across each edge, which is continuous between cells.
#
# For the unit tangent $t$ of an edge, $\nu = R(t)$ is the unit vector in
# the tangent plane normal to the edge, which up to sign is the facet
# normal ({py:class}`ufl.FacetNormal`) of the surface mesh. By
# {eq}`rotation-properties`,
#
# $$
# \begin{aligned}
#   R(\sigma) \cdot \nu = R(\sigma) \cdot R(t) = \sigma \cdot t,
# \end{aligned}
# $$
#
# so the tangential component of $\sigma$ along the edge is the normal
# component of $R(\sigma)$ across the edge. $R$ therefore turns the
# Nédélec degrees-of-freedom into Raviart-Thomas ones.
# ```
#
# ### Reformulation of the strong form
#
# The flux $q \in H(\mathrm{div}; \Omega)$ in {eq}`mixed-form` would
# naturally be discretised with Raviart-Thomas elements. Here we instead
# write $q = R(\sigma)$ with $\sigma \in H(\mathrm{curl}; \Omega),$ which
# $R$ maps onto $H(\mathrm{div}; \Omega),$ so that $\sigma$ can be
# discretised with Nédélec elements; by the Lemma, the discrete spaces
# correspond too.
# Substituting $q = R(\sigma)$ into {eq}`mixed-form`, we seek
# $\sigma \in H(\mathrm{curl}; \Omega),$ $u \in L^2(\Omega)$ and
# $r \in \mathbb{R}$ such that
#
# $$
# \begin{aligned}
#   R(\sigma) - \nabla u &= 0, \\
#   \nabla \cdot R(\sigma) + r &= g, \\
#   \int_{\Omega} u ~\mathrm{d}x &= h.
# \end{aligned}
# $$ (rotated-form)
#
# The surface curl of a tangential field $w$ is the scalar
#
# $$
# \begin{aligned}
#   \operatorname{rot} w := n \cdot (\nabla \times w).
# \end{aligned}
# $$ (surface-curl)
#
# The divergence of the rotated field is minus the surface curl of
# $\sigma.$ This follows from the product rule for the divergence of a
# cross product, as the unit normal has no curl (the unit normal is the
# gradient of the signed distance to the surface, also on curved cells):
#
# $$
# \begin{aligned}
#   \nabla \cdot R(\sigma)
#   &= \sigma \cdot (\nabla \times n) - n \cdot (\nabla \times \sigma)
#   = -\operatorname{rot} \sigma,
# \end{aligned}
# $$ (divergence-rotation)
#
# so {eq}`rotated-form` becomes
#
# $$
# R(\sigma) - \nabla u = 0,
# $$ (flux-equation)
#
# $$
# -\operatorname{rot} \sigma + r = g,
# $$ (curl-equation)
#
# $$
# \int_{\Omega} u ~\mathrm{d}x = h.
# $$ (mean-equation)
#
# ### Derivation of the weak form
#
# We take the inner product of {eq}`flux-equation` with $R(\tau)$ and
# integrate over $\Omega.$ By {eq}`rotation-properties`,
#
# $$
# \begin{aligned}
#   0 = \int_{\Omega} (R(\sigma) - \nabla u) \cdot R(\tau) ~\mathrm{d}x
#   &= \int_{\Omega} R(\sigma) \cdot R(\tau) ~\mathrm{d}x
#   - \int_{\Omega} \nabla u \cdot R(\tau) ~\mathrm{d}x \\
#   &= \int_{\Omega} \sigma \cdot \tau ~\mathrm{d}x
#   - \int_{\Omega} \nabla u \cdot R(\tau) ~\mathrm{d}x.
# \end{aligned}
# $$
#
# Integrating the last term by parts, as the sphere has no boundary,
# and applying {eq}`divergence-rotation` to $\tau,$
#
# $$
# \begin{aligned}
#   \int_{\Omega} \nabla u \cdot R(\tau) ~\mathrm{d}x
#   = -\int_{\Omega} u \, \nabla \cdot R(\tau) ~\mathrm{d}x
#   = \int_{\Omega} u \operatorname{rot} \tau ~\mathrm{d}x.
# \end{aligned}
# $$
#
# Multiplying {eq}`curl-equation` by $v$ and {eq}`mean-equation` by $s$
# gives the weak form. We discretise it with $V,$ the Nédélec space of
# the first kind and degree $k,$ and $Q,$ the discontinuous Lagrange space
# of degree $k - 1,$ which needs no continuity between cells: find
# $(\sigma, u, r) \in V \times Q \times \mathbb{R}$ such that
#
# $$
# \begin{aligned}
#   \int_{\Omega} \sigma \cdot \tau - u \operatorname{rot} \tau
#   + (\operatorname{rot} \sigma) v - r v + s u ~\mathrm{d}x
#   = -\int_{\Omega} g v ~\mathrm{d}x + h s
#   \quad \forall (\tau, v, s) \in V \times Q \times \mathbb{R}.
# \end{aligned}
# $$
#
# ```{note}
# The rotation $R$ depends on the cell normal $n,$ the normal to the
# surface on each cell (not the facet normal $\nu,$ which lies in the
# tangent plane). If the cell normals of two cells sharing an edge point
# to opposite sides of the surface, $R(\sigma)$ changes sign across the
# edge and its normal component is discontinuous, so $R(\sigma)$ is not
# in $H(\mathrm{div}).$ The cell normals must therefore be consistently
# oriented for this formulation to give the same $u_h$ as the
# Raviart-Thomas formulation of {doc}`demo_manifold`.
# ```
#
# ## Implementation

# +
from mpi4py import MPI
from petsc4py import PETSc

import gmsh
import numpy as np

import basix.ufl
import dolfinx
import dolfinx.fem.petsc
import dolfinx.io.gmsh
import ufl

# -

# ### Mesh generation
#
# We use the same degree, geometry and mesh as {doc}`demo_manifold`:
# `surface_mesh` is the boundary of `volume_mesh`, a tetrahedral mesh of
# the unit ball.

# +
order = 3
res = 0.2

gmsh.initialize()
if MPI.COMM_WORLD.rank == 0:
    ball = gmsh.model.occ.addSphere(0, 0, 0, 1)
    gmsh.model.occ.synchronize()
    gmsh.model.addPhysicalGroup(3, [ball], tag=1)
    gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", res)
    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.setOrder(order)
volume_mesh = dolfinx.io.gmsh.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=3).mesh
gmsh.finalize()
volume_mesh.topology.create_connectivity(2, 3)
boundary_facets = dolfinx.mesh.exterior_facet_indices(volume_mesh.topology)
surface_mesh = dolfinx.mesh.create_submesh(volume_mesh, 2, boundary_facets)[0]
# -

# ### Computing a consistent cell normal
#
# {py:class}`ufl.CellNormal` is computed from the Jacobian
# $J = \partial x / \partial X$ of the map $x(X)$ from the reference
# coordinates $X$ to each cell. The columns
# $J_0 = \partial x / \partial X_0$ and $J_1 = \partial x / \partial X_1$
# are tangent to the cell, and
#
# $$
# \begin{aligned}
#   n = \frac{J_0 \times J_1}{\vert J_0 \times J_1 \vert}.
# \end{aligned}
# $$
#
# The columns follow the order of the cell's vertices, and swapping two
# vertices flips $n.$ The cells of `surface_mesh` are facets of
# `volume_mesh` and keep the vertex order they have in their tetrahedra,
# so on the sphere some cell normals point out and others in.
# {py:func}`compute_cell_orientations
# <dolfinx.mesh.compute_cell_orientations>` returns, for each owned and
# ghost cell, the sign that makes all of them agree.
#
# ```{admonition} Partitioning and non-orientable surfaces
# :class: dropdown
# The signs are computed from the vertex orders alone, so whether the
# agreed normal points out or in depends on the partitioning, which does
# not change $u_h.$ The function raises an error on a surface that cannot
# be oriented, such as the Möbius strips of {doc}`demo_mobius` and
# {doc}`demo_mobius_eikonal`. There, only the Raviart-Thomas formulation
# works, as that formulation needs consistent signs per facet rather than
# a consistent normal.
# ```
#
# The signs are in local cell order. We store them in a piecewise
# constant function, whose degrees-of-freedom are not guaranteed to be
# numbered in the same order as the cells, so the signs are placed
# through its dofmap.

# +
orientations = dolfinx.mesh.compute_cell_orientations(surface_mesh.topology)
S = dolfinx.fem.functionspace(surface_mesh, ("DG", 0))
o_h = dolfinx.fem.Function(S)
o_h.x.array[S.dofmap.list[:, 0]] = orientations
n_h = o_h * ufl.CellNormal(surface_mesh)

num_owned = surface_mesh.topology.index_map(2).size_local
num_flipped = surface_mesh.comm.allreduce(int(np.sum(orientations[:num_owned] < 0)), op=MPI.SUM)
if surface_mesh.comm.rank == 0:
    print(f"{num_flipped} of {surface_mesh.topology.index_map(2).size_global} cell normals flipped")
# -

# We implement the surface curl {eq}`surface-curl` with the oriented
# normal $n_h$ in place of $n.$


# +
def rot(w):
    """Surface curl of a tangential field."""
    return ufl.dot(n_h, ufl.curl(w))


# -

# ### Manufactured problem
#
# For the manufactured solution we take $g = x_0 x_1 x_2,$ which has zero
# mean by symmetry, and $h = 0,$ so that the exact solution is
# $u = -x_0 x_1 x_2 / 12$ and $r = 0.$ $u$ is a spherical harmonic of
# degree 3, so $\nabla \cdot \nabla u = -12 u = g.$

x = ufl.SpatialCoordinate(surface_mesh)
g = x[0] * x[1] * x[2]
u_exact = -x[0] * x[1] * x[2] / 12

# ### Variational formulation
#
# We use the spaces $V,$ $Q$ and $\mathbb{R}$ of the weak form, with
# $k = 3.$

# +
V = dolfinx.fem.functionspace(surface_mesh, ("N1curl", order))
Q = dolfinx.fem.functionspace(surface_mesh, ("DG", order - 1))
r_el = basix.ufl.real_element(
    surface_mesh.basix_cell(), value_shape=(), dtype=dolfinx.default_real_type
)
R = dolfinx.fem.functionspace(surface_mesh, r_el)
W = ufl.MixedFunctionSpace(V, Q, R)
(sigma, u, r) = ufl.TrialFunctions(W)
(tau, v, s) = ufl.TestFunctions(W)

a = (
    ufl.inner(sigma, tau)
    - ufl.inner(u, rot(tau))
    + ufl.inner(rot(sigma), v)
    - ufl.inner(r, v)
    + ufl.inner(u, s)
) * ufl.dx
a_blocked: list[list[ufl.Form | None]] = ufl.extract_blocks(a)  # type: ignore[assignment]
L_blocked: list[ufl.Form | ufl.ZeroBaseForm] = [
    ufl.ZeroBaseForm((tau,)),
    -ufl.inner(g, v) * ufl.dx,
    ufl.ZeroBaseForm((s,)),  # h = 0
]
petsc_options = {
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "ksp_error_if_not_converged": True,
}
problem = dolfinx.fem.petsc.LinearProblem(
    a_blocked,
    L_blocked,
    bcs=[],
    petsc_options=petsc_options,
    petsc_options_prefix="vorticity_",
    kind="mpi",
)
sigma_h, u_h, r_h = problem.solve()
# -

# ### Errors
#
# Applying $R$ to {eq}`flux-equation` and using
# $R(R(\sigma)) = -\sigma$ from {eq}`rotation-properties`, the exact flux
# is $\sigma = -R(\nabla u) = -n \times \nabla u,$ with $\nabla u$ the
# surface gradient. Taking $n = n_h$ there too makes the comparison
# independent of whether $n_h$ points out or in.

# +
n_exact = x / ufl.sqrt(ufl.dot(x, x))
grad_u_exact = ufl.grad(u_exact) - ufl.dot(ufl.grad(u_exact), n_exact) * n_exact
sigma_exact = -ufl.cross(n_h, grad_u_exact)


def norm(expr):
    """The L2 norm of ``expr`` over the sphere."""
    form = dolfinx.fem.form(ufl.inner(expr, expr) * ufl.dx)
    return np.sqrt(abs(surface_mesh.comm.allreduce(dolfinx.fem.assemble_scalar(form), op=MPI.SUM)))


E_u = norm(u_h - u_exact)
E_sigma = norm(sigma_h - sigma_exact)
if surface_mesh.comm.rank == 0:
    print("L2 error in u:", E_u)
    print("L2 error in sigma:", E_sigma)
# -

# Finally, we free the PETSc objects held by the problem.

del problem
PETSc.garbage_cleanup(surface_mesh.comm)  # type: ignore[bad-argument-type]
