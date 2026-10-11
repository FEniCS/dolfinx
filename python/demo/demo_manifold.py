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
#   surface, which requires the two cells of every facet to agree on its
#   normal,
# * widen the ghost layer beyond what
#   {py:class}`GhostMode <dolfinx.mesh.GhostMode>` offers, by writing a
#   partitioner that ghosts a cell wherever it shares a *vertex*. A
#   surface taken as a submesh inherits its ghost layer from the volume,
#   and `shared_facet` leaves it one dimension too narrow to orient the
#   cells in parallel,
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
# degree of the geometry, the mesh resolution, and how the cells are
# and the mesh resolution.

order = 3
res = 0.2

# ### The mesh
#
# We mesh the unit ball with Gmsh on rank 0, distribute it, and take
# the sphere as the submesh of its exterior facets.
#
# ### Widening the ghost layer
#
# A submesh cell is a ghost exactly when the parent entity it came from
# was, so the surface inherits its ghost layer from the ball. That layer
# is one dimension too narrow. `GhostMode.shared_facet` ghosts the tets
# that share a *facet* with an owned tet, but two neighbouring cells of
# the surface are two boundary facets sharing an *edge*, and the tet
# behind the second one usually meets the owned side along that edge
# alone. It is never ghosted, its boundary facet does not exist here,
# and the surface ends up with cells whose neighbour across an edge is
# missing -- which is exactly what
# {py:meth}`create_facet_orientations
# <dolfinx.mesh.Topology.create_facet_orientations>` needs.
#
# {py:class}`GhostMode <dolfinx.mesh.GhostMode>` offers no wider layer,
# but a partitioner can build one. A partitioner returns the destination
# ranks of each cell, the first being its owner and the rest ghosting
# it, so ghosting more widely only means adding ranks to those lists. It
# is free to ignore the dual graph it is handed and return a layout
# decided elsewhere, which is what lets us keep a partition already
# computed and only widen it.
#
# So we partition once as usual, then rebuild: each cell stays with its
# owner and is ghosted on every rank that shares one of its vertices,
# found with {py:meth}`index_to_dest_ranks
# <dolfinx.common.IndexMap.index_to_dest_ranks>`. A tet meeting the
# owned side along an edge shares both of that edge's vertices, so it is
# now ghosted and its boundary facet comes with it.


def ghost_by_vertex(msh):
    """Rebuild ``msh`` so a cell is ghosted wherever it shares a vertex."""
    tdim = msh.topology.dim
    num_owned = msh.topology.index_map(tdim).size_local
    cell_to_vertex = msh.topology.connectivity(tdim, 0)
    dest, offset = msh.topology.index_map(0).index_to_dest_ranks()

    rank = msh.comm.rank
    destinations, counts = [], []
    for c in range(num_owned):
        extra = set()
        for v in cell_to_vertex.links(c):
            extra.update(dest[offset[v] : offset[v + 1]])
        extra.discard(rank)
        destinations.append([rank, *sorted(extra)])
        counts.append(1 + len(extra))
    offsets = np.zeros(len(counts) + 1, dtype=np.int32)
    np.cumsum(counts, out=offsets[1:])
    layout = dolfinx.graph.adjacencylist(
        np.fromiter((r for row in destinations for r in row), dtype=np.int32), offsets
    )

    # The cells are handed back in the order they are passed below, so
    # the layout can be returned as it stands
    def keep_partition(comm, nparts, dual_graph, node_weights, edge_weights, ghosting):
        return layout

    # create_mesh numbers the points it is given by position, so supply
    # the owned nodes and refer to them through the numbering of their
    # index map, whose local ranges are that same running count
    geometry = msh.geometry
    node_map = geometry.index_map()
    nodes = node_map.local_to_global(geometry.dofmaps[0][:num_owned].reshape(-1).astype(np.int32))
    domain = ufl.Mesh(
        basix.ufl.element(
            "Lagrange",
            msh.topology.cell_name(),
            geometry.cmaps[0].degree,
            basix.LagrangeVariant(geometry.cmaps[0].variant),
            shape=(geometry.dim,),
            dtype=geometry.x.dtype,
        )
    )
    return dolfinx.mesh.create_mesh(
        msh.comm,
        nodes.reshape(num_owned, -1),
        domain,
        geometry.x[: node_map.size_local],
        partitioner=keep_partition,
    )


# +
gmsh.initialize()
if MPI.COMM_WORLD.rank == 0:
    ball = gmsh.model.occ.addSphere(0, 0, 0, 1)
    gmsh.model.occ.synchronize()
    gmsh.model.addPhysicalGroup(3, [ball], tag=1)
    gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", res)
    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.setOrder(order)
ball_mesh = dolfinx.io.gmsh.model_to_mesh(
    gmsh.model, MPI.COMM_WORLD, 0, gdim=3, ghost_mode=dolfinx.io.gmsh.GhostMode.shared_facet
).mesh
gmsh.finalize()
ball_mesh = ghost_by_vertex(ball_mesh)
ball_mesh.topology.create_connectivity(2, 3)
fdim = ball_mesh.topology.dim - 1
assert fdim == 2
facet_map = ball_mesh.topology.index_map(fdim)
facet_vector = dolfinx.la.vector(facet_map, dtype=np.int32)
boundary_facets = dolfinx.mesh.exterior_facet_indices(ball_mesh.topology)
facet_vector.array[boundary_facets] = 1
facet_vector.scatter_forward()
all_boundary_facets = np.flatnonzero(facet_vector.array).astype(np.int32)
mesh = dolfinx.mesh.create_submesh(ball_mesh, fdim, all_boundary_facets)[0]
# -

# ### Signing the facets
#
# The vertex order of a cell on a surface fixes its normal, and thereby
# which side of the surface the cell regards as "out". The
# contravariant Piola map, which maps Raviart-Thomas basis functions
# from the reference cell, ties the direction of the normal flux across
# each edge to this side. The normal component of $\sigma_h$ is thus
# continuous across an edge only if the two cells sharing it agree on
# the side. Two cells agree
# when, going round each in the order of its vertices, they run their
# shared edge in opposite directions, as cells $(0, 1, 2)$ and
# $(1, 3, 2)$ do with edge $\{1, 2\}$.
#
# There is no guarantee that the cells of an input mesh agree. An
# example is a submesh of the exterior boundary of a volume mesh:
# each facet keeps the vertex order it has in its volume cell.
# {py:meth}`create_facet_orientations
# <dolfinx.mesh.Topology.create_facet_orientations>` neither reorders the
# cells nor reverses them. It signs the facets on which two cells
# disagree: negating the degrees-of-freedom of one of the two there
# makes their normal components match again, and disturbs nothing else,
# since only a facet's own basis functions have a non-zero normal trace
# on it. The Raviart-Thomas (and Brezzi-Douglas-Marini) basis functions
# follow those signs.
#
# Because this asks nothing of the surface as a whole, only of each
# facet in turn, it works even where no consistent orientation exists at
# all, as on the Möbius strip of {doc}`demo_mobius`. Which of the two
# cells carries the sign is settled by the global numbering, so it can
# change with the partitioning; the space, and so the solution, do
# not.
#
# The signs must be requested explicitly, as they are needed only by
# $H(\mathrm{div})$ elements on surfaces. Lagrange and Nédélec elements
# do not depend on them, and creating an $H(\mathrm{div})$ space on a
# surface before they have been computed raises an error.

# +
cell_map = mesh.topology.index_map(2)
num_cells = cell_map.size_local + cell_map.num_ghosts
mesh.topology.create_facet_orientations()

# No cell is reordered or reversed: the signs sit on the facets whose
# two cells disagree. There are plenty, as the facets of the ball are
# not consistently ordered.
signs = mesh.topology.facet_orientations()
per_cell = signs.size // num_cells
num_signed = mesh.comm.allreduce(
    int(np.sum(signs[: cell_map.size_local * per_cell] < 0)), op=MPI.SUM
)
if mesh.comm.rank == 0:
    print(f"{num_signed} of {cell_map.size_global * per_cell} facets signed")
# -

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
a_blocked: list[list[ufl.Form | None]] = ufl.extract_blocks(a)  # type: ignore[assignment]
L_blocked: list[ufl.Form | ufl.ZeroBaseForm] = [
    ufl.ZeroBaseForm((tau,)),
    ufl.inner(g, v) * ufl.dx,
    ufl.ZeroBaseForm((t,)),
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
# flux is $g$.

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
