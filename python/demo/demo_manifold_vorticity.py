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
# This demo solves the problem of {doc}`demo_manifold` again, but with
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
# ## Equation and problem definition
#
# As in {doc}`demo_manifold`, $\Omega$ is the unit sphere, $g = x_0 x_1
# x_2$, and we seek $u$ and a constant $r$ with
# $\nabla \cdot \nabla u + r = g$ and zero mean of $u$. The exact solution
# is $u = -x_0 x_1 x_2 / 12$ and $r = 0$.
#
# On a manifold, here a surface in 3D with unit normal $n$, the Nédélec
# and Raviart-Thomas fields are tangential to the surface: their Piola
# maps take them into the tangent plane of each cell. We write
#
# $$
# \begin{aligned}
#   R(w) = n \times w
# \end{aligned}
# $$
#
# for any vector field $w$. $R(w)$ is tangential: $R$ drops the normal
# component of $w$ and rotates the rest by $90^\circ$ in the tangent
# plane. We only apply $R$ to tangential fields, on which it is a
# rotation: it preserves inner products, $R(a) \cdot R(b) = a \cdot b$,
# and rotating twice reverses a field, $R(R(w)) = -w$.
#
# Let $t$ be the unit tangent of an edge of the mesh. Its rotation
# $\nu = R(t)$ is the unit vector in the tangent plane normal to the
# edge, which up to sign is the facet normal
# ({py:class}`ufl.FacetNormal`) of the surface mesh. For $z = R(\sigma)$,
#
# $$
# \begin{aligned}
#   z \cdot \nu = R(\sigma) \cdot R(t) = \sigma \cdot t,
# \end{aligned}
# $$
#
# so $R$ turns the tangential component of $\sigma$ along the edge into
# the normal component of $z$ across it. The
# degrees-of-freedom of an $H(\mathrm{curl})$-conforming Nédélec element
# are moments of the tangential component, which is continuous between
# cells; those of an $H(\mathrm{div})$-conforming Raviart-Thomas element
# are moments of the normal component, which is continuous across edges.
# Rotating a Nédélec field therefore gives a Raviart-Thomas field,
# provided both cells of each edge use the same $n$: flipping $n$ on one
# cell flips $R(\sigma)$ there, and its normal component jumps.
#
# The divergence of the rotated field is minus the surface curl
# $\operatorname{rot} \sigma$ of $\sigma$. By the product rule for the
# divergence of a cross product, and as the unit normal has no curl (it
# is the gradient of the signed distance to the surface, also on curved
# cells; on an affine cell it is simply constant),
#
# $$
# \begin{aligned}
#   \nabla \cdot R(\sigma)
#   &= \sigma \cdot (\nabla \times n) - n \cdot (\nabla \times \sigma)
#   = -n \cdot (\nabla \times \sigma)
#   = -\operatorname{rot} \sigma.
# \end{aligned}
# $$
#
# We therefore seek the flux as $R(\sigma) = \nabla u$, i.e.
# $\sigma = -R(\nabla u)$, in a Nédélec space $V$ of the first kind,
# and substitute it into the weak form of {doc}`demo_manifold`. As the
# rotation preserves inner products, this gives: find
# $(\sigma, u, r) \in V \times Q \times \mathbb{R}$ such that
#
# $$
# \begin{aligned}
#   \int_{\Omega} \sigma \cdot \tau - u \operatorname{rot} \tau
#   + (\operatorname{rot} \sigma) v - r v + s u ~\mathrm{d}x
#   = -\int_{\Omega} g v ~\mathrm{d}x
#   \quad \forall (\tau, v, s) \in V \times Q \times \mathbb{R}.
# \end{aligned}
# $$
#
# Cell by cell, $R(\sigma)$ for $\sigma$ in the Nédélec space of
# degree $k$ spans the Raviart-Thomas space of degree $k$, so $u_h$ is
# the same as in {doc}`demo_manifold`, provided that $n$ is the same on
# every cell. That is where the orientation comes in.
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

# We use the same degree, geometry and mesh as {doc}`demo_manifold`: the
# surface mesh is the boundary of a tetrahedral mesh of the unit ball,
# `ball_mesh`.

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
ball_mesh = dolfinx.io.gmsh.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=3).mesh
gmsh.finalize()
ball_mesh.topology.create_connectivity(2, 3)
boundary_facets = dolfinx.mesh.exterior_facet_indices(ball_mesh.topology)
mesh = dolfinx.mesh.create_submesh(ball_mesh, 2, boundary_facets)[0]
# -

# ### A consistent normal
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
# vertices flips $n$. The cells of the surface mesh are facets of
# `ball_mesh` and keep the vertex order they have in their tetrahedra, so
# on the sphere some cell normals point out and others in.
# {py:func}`compute_cell_orientations
# <dolfinx.mesh.compute_cell_orientations>` returns, for each owned and
# ghost cell, the sign that makes all of them agree. The signs are
# computed from the vertex orders alone, so whether the agreed normal
# points out or in depends on the partitioning, which does not change
# $u_h$. The function raises an error on a surface that cannot be
# oriented, such as the Möbius strips of
# {doc}`demo_mobius` and {doc}`demo_mobius_eikonal`. There, only the
# Raviart-Thomas formulation works, as that formulation needs consistent
# signs per facet rather than a consistent normal.
#
# The signs are in local cell order. We store them in a piecewise
# constant function, whose degrees-of-freedom are not numbered in the
# same order as the cells, so the signs are placed through its dofmap.

# +
orientations = dolfinx.mesh.compute_cell_orientations(mesh.topology)
S = dolfinx.fem.functionspace(mesh, ("DG", 0))
o_h = dolfinx.fem.Function(S)
o_h.x.array[S.dofmap.list[:, 0]] = orientations
n_h = o_h * ufl.CellNormal(mesh)

num_owned = mesh.topology.index_map(2).size_local
num_flipped = mesh.comm.allreduce(int(np.sum(orientations[:num_owned] < 0)), op=MPI.SUM)
if mesh.comm.rank == 0:
    print(f"{num_flipped} of {mesh.topology.index_map(2).size_global} cell normals flipped")
# -

# With the oriented normal, we define the surface curl
# $\operatorname{rot} w = n_h \cdot (\nabla \times w)$.


# +
def rot(w):
    """Surface curl of a tangential field."""
    return ufl.dot(n_h, ufl.curl(w))


# -

# ### Function spaces and variational problem

# +
V = dolfinx.fem.functionspace(mesh, ("N1curl", order))
Q = dolfinx.fem.functionspace(mesh, ("DG", order - 1))
r_el = basix.ufl.real_element(mesh.basix_cell(), value_shape=(), dtype=dolfinx.default_real_type)
R = dolfinx.fem.functionspace(mesh, r_el)
W = ufl.MixedFunctionSpace(V, Q, R)
(sigma, u, r) = ufl.TrialFunctions(W)
(tau, v, s) = ufl.TestFunctions(W)

x = ufl.SpatialCoordinate(mesh)
g = x[0] * x[1] * x[2]
u_exact = -x[0] * x[1] * x[2] / 12

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
    ufl.ZeroBaseForm((s,)),
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
# The exact flux is $\sigma = -R(\nabla u) = -n \times \nabla u$, with
# $\nabla u$ the surface gradient. Taking $n = n_h$ there too makes the
# comparison independent of whether $n_h$ points out or in.

# +
n_exact = x / ufl.sqrt(ufl.dot(x, x))
grad_u_exact = ufl.grad(u_exact) - ufl.dot(ufl.grad(u_exact), n_exact) * n_exact
sigma_exact = -ufl.cross(n_h, grad_u_exact)


def norm(expr):
    """The L2 norm of ``expr`` over the sphere."""
    form = dolfinx.fem.form(ufl.inner(expr, expr) * ufl.dx)
    return np.sqrt(abs(mesh.comm.allreduce(dolfinx.fem.assemble_scalar(form), op=MPI.SUM)))


E_u = norm(u_h - u_exact)
E_sigma = norm(sigma_h - sigma_exact)
if mesh.comm.rank == 0:
    print("L2 error in u:", E_u)
    print("L2 error in sigma:", E_sigma)
# -

# Finally, we free the PETSc objects held by the problem.

del problem
PETSc.garbage_cleanup(mesh.comm)  # type: ignore[bad-argument-type]
