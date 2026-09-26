# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
# ---

# # Mixed Poisson equation on the sphere
#
# This demo solves the Poisson equation in mixed form on the unit
# sphere, following Section 5.1.1 of {cite:t}`rognes2013manifolds`.
# It illustrates how to
#
# * solve a PDE on a manifold, here a surface embedded in 3D,
# * use $H(\mathrm{div})$-conforming Raviart-Thomas elements on a
#   surface, which requires consistently oriented cells,
# * impose a global constraint with a real (constant) space.
#
# ```{admonition} Download sources
# :class: download
# * {download}`Python script <./demo_manifold.py>`
# * {download}`Jupyter notebook <./demo_manifold.ipynb>`
# ```
#
# ## Equation and problem definition
#
# Let $\Omega$ be the surface of the unit sphere centred at the origin.
# Given $g$, we seek $u$, its surface gradient $\sigma$ and a constant
# $r$ such that
#
# $$
# \begin{aligned}
#   \sigma - \nabla u &= 0 \quad \text{on } \Omega, \\
#   \nabla \cdot \sigma + r &= g \quad \text{on } \Omega, \\
#   \int_{\Omega} u \, \mathrm{d}x &= 0,
# \end{aligned}
# $$
#
# where $\nabla$ and $\nabla \cdot$ are the surface gradient and
# divergence. The sphere has no boundary, so $u$ is determined up to a
# constant, which the last equation fixes. Integrating the second
# equation over $\Omega$ shows that $r$ is the mean of $g$.
#
# We multiply by test functions $\tau$ and $v$ and a constant $t$,
# integrate over $\Omega$, and integrate $\nabla u \cdot \tau$ by parts,
# which gives no boundary term on a closed surface. The weak form reads:
# find $(\sigma, u, r) \in V \times Q \times \mathbb{R}$ such that
#
# $$
# \int_{\Omega} \sigma \cdot \tau + (\nabla \cdot \sigma) v
#   + (\nabla \cdot \tau) u + r v + t u \, \mathrm{d}x
#   = \int_{\Omega} g v \, \mathrm{d}x
#   \quad \forall (\tau, v, t) \in V \times Q \times \mathbb{R}.
# $$
#
# $V$ must be $H(\mathrm{div})$-conforming, while $Q$ needs no
# continuity. We take Raviart-Thomas elements of degree $k$ for $V$
# and discontinuous Lagrange elements of degree $k-1$ for $Q$, a stable
# pair since the divergence maps $V$ onto $Q$.
#
# As in {cite:t}`rognes2013manifolds`, we take $g = x_0 x_1 x_2$. It is
# the restriction of a harmonic, homogeneous polynomial of degree 3 to
# the sphere, and hence an eigenfunction of the surface Laplacian with
# eigenvalue $-3(3+1) = -12$. As $g$ has zero mean, $r = 0$ and the
# exact solution is $u = -x_0 x_1 x_2 / 12$.
#
# ## Implementation
#
# We start by importing the required modules

# +
from mpi4py import MPI

import gmsh
import numpy as np

import basix.ufl
import dolfinx
import dolfinx.fem.petsc
import dolfinx.io.gmsh
import ufl

# -

# and set the degree $k$ of the Raviart-Thomas space, which is also the
# degree of the geometry, and the mesh resolution.

order = 3
res = 0.2
orient_cells = True  # Set to False to see what goes wrong on a mesh that is not oriented

# ### The mesh
#
# We mesh the unit ball with Gmsh on rank 0, distribute it, and take
# the sphere as the submesh of its exterior facets.

# +
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

# ### Orienting the cells
#
# The vertex order of a cell on a surface fixes its normal, and thereby
# which side of the surface the cell regards as "out". The
# contravariant Piola map, which maps Raviart-Thomas basis functions
# from the reference cell, ties the direction of the normal flux across
# each edge to this side. The normal component of $\sigma_h$ is thus
# continuous across an edge only if the two cells sharing it agree on
# the side, i.e. if the mesh is consistently oriented.
#
# There is no guarantee that an input mesh has a consistent orientation.
# An example is a submesh of the exterior boundary of a volume mesh:
# each facet keeps the vertex order it has in its volume cell.
# {py:meth}`create_cell_orientations
# <dolfinx.mesh.Topology.create_cell_orientations>`
# computes a consistent orientation from the vertex orders, in serial and
# in parallel, and the Raviart-Thomas (and Brezzi-Douglas-Marini) basis
# functions follow it. Whether the result points inwards or outwards is
# arbitrary, but does not change the solution. It raises an error for a
# non-orientable surface, such as a Möbius strip, where no
# $H(\mathrm{div})$-conforming space of this kind exists.
#
# The orientation must be requested explicitly, as it is needed only by
# $H(\mathrm{div})$ elements on surfaces. Lagrange and Nédélec elements
# do not depend on it. In legacy FEniCs the cells were oriented relative
# to a normal field given by the user {cite}`rognes2013manifolds`.

if orient_cells:
    mesh.topology.create_cell_orientations()
    # The orientation information is stored in the last bit of the
    # cell permutation info.
    # We count the number of reversed cells, which should be zero for a
    # consistently oriented mesh.
    reversed_cells = mesh.topology.get_cell_permutation_info() >> 31
    num_owned = mesh.topology.index_map(2).size_local
    num_reversed = mesh.comm.allreduce(int(reversed_cells[:num_owned].sum()), op=MPI.SUM)
    if mesh.comm.rank == 0:
        print(f"{num_reversed} of {mesh.topology.index_map(2).size_global} cells reversed")

# ### Function spaces
#
# The constants $r$ and $t$ live in a real space, with a single degree
# of freedom shared by all cells. The three spaces are combined in a
# {py:class}`MixedFunctionSpace <ufl.MixedFunctionSpace>`.

# +
V = dolfinx.fem.functionspace(mesh, ("RT", order))
Q = dolfinx.fem.functionspace(mesh, ("DG", order - 1))
r_el = basix.ufl.real_element(mesh.basix_cell(), value_shape=(), dtype=dolfinx.default_real_type)
R = dolfinx.fem.functionspace(mesh, r_el)
W = ufl.MixedFunctionSpace(V, Q, R)

(sigma, u, r) = ufl.TrialFunctions(W)
(tau, v, t) = ufl.TestFunctions(W)
# -

# ### Source and exact solution
#
# The exact flux is the surface gradient of $u$, i.e. the gradient of
# its polynomial extension projected onto the tangent plane of the
# sphere, whose normal is $x / |x|$.

# +
x = ufl.SpatialCoordinate(mesh)
g = x[0] * x[1] * x[2]
u_exact = -x[0] * x[1] * x[2] / 12

n = x / ufl.sqrt(ufl.dot(x, x))
grad_u_exact = ufl.grad(u_exact)
sigma_exact = grad_u_exact - ufl.dot(grad_u_exact, n) * n
# -

# ### Variational problem and solve
#
# The right-hand side has a block per test function, the first and last
# of which are zero. We solve the blocked system with a direct solver.

# +
a = (
    ufl.inner(sigma, tau)
    + ufl.inner(ufl.div(sigma), v)
    + ufl.inner(u, ufl.div(tau))
    + ufl.inner(r, v)
    + ufl.inner(u, t)
) * ufl.dx
L = [ufl.ZeroBaseForm((tau,)), ufl.inner(g, v) * ufl.dx, ufl.ZeroBaseForm((t,))]

petsc_options = {
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "ksp_error_if_not_converged": True,
}
problem = dolfinx.fem.petsc.LinearProblem(
    ufl.extract_blocks(a),
    L,
    bcs=[],
    petsc_options=petsc_options,
    petsc_options_prefix="mixed_poisson_",
    kind="mpi",
)
sigma_h, u_h, r_h = problem.solve()
# -

# For visualisation, if DOLFINx is built with ADIOS2, we interpolate
# $u_h$ into a discontinuous Lagrange space of the degree of the
# geometry and write it to file.

if dolfinx.has_adios2:
    V_out = dolfinx.fem.functionspace(mesh, ("DG", order))
    v_out = dolfinx.fem.Function(V_out)
    v_out.interpolate(u_h)
    v_out.name = "u_h"
    with dolfinx.io.VTXWriter(mesh.comm, "u.bp", [v_out]) as vtx_writer:
        vtx_writer.write(0.0)

# ### Errors
#
# Finally, we compute the error of $u_h$ in the $L^2$ norm, and of
# $\sigma_h$ in the $L^2$ and $H(\mathrm{div})$ norms. As
# $\nabla \cdot \sigma = g - r$ and $r = 0$, the divergence of the exact
# flux is $g$. Without the orientation, these errors are several orders
# of magnitude larger.

# +
L2_error_u = dolfinx.fem.form(ufl.inner(u_h - u_exact, u_h - u_exact) * ufl.dx)
L2_error_sigma = dolfinx.fem.form(ufl.inner(sigma_h - sigma_exact, sigma_h - sigma_exact) * ufl.dx)
Hdiv_error_sigma = dolfinx.fem.form(
    ufl.inner(ufl.div(sigma_h) - g, ufl.div(sigma_h) - g) * ufl.dx
    + ufl.inner(sigma_h - sigma_exact, sigma_h - sigma_exact) * ufl.dx
)

E_u = np.sqrt(mesh.comm.allreduce(dolfinx.fem.assemble_scalar(L2_error_u), op=MPI.SUM))
E_sigma = np.sqrt(mesh.comm.allreduce(dolfinx.fem.assemble_scalar(L2_error_sigma), op=MPI.SUM))
E_sigma_hdiv = np.sqrt(
    mesh.comm.allreduce(dolfinx.fem.assemble_scalar(Hdiv_error_sigma), op=MPI.SUM)
)
if mesh.comm.rank == 0:
    print("L2 error in u:", E_u)
    print("L2 error in sigma:", E_sigma)
    print("H(div) error in sigma:", E_sigma_hdiv)
# -

# ## References
# ```{bibliography}
# :filter: docname in docnames
# ```
