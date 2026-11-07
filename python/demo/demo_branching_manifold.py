# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
# ---

# # Mixed Poisson on a branching surface
#
# This demo solves the Poisson equation in mixed form on three sheets
# meeting along a line, a surface that branches. It illustrates how to
#
# * give each branch its own $H(\mathrm{div})$-conforming space, using
#   submeshes,
# * couple the branches through a Lagrange multiplier on the joint, a
#   mesh of lower dimension,
# * tag cells and facets while creating a mesh on one process, and
#   transfer the facet tags to a submesh,
# * integrate over the facets of chosen cells, from the side of each
#   cell,
# * interpolate functions from the submeshes into a single function on
#   the parent mesh, for visualisation.
#
# ```{admonition} Download sources
# :class: download
# * {download}`Python script <./demo_branching_manifold.py>`
# * {download}`Jupyter notebook <./demo_branching_manifold.ipynb>`
# ```
#
# ## Equation and problem definition
#
# The surface $\Omega$ consists of three branches $\Omega_b$,
# $b = 0, 1, 2$, joined along a curve $\Gamma$. On each branch the
# Poisson equation in mixed form reads
#
# $$
# \begin{aligned}
#   \sigma_b - \nabla_\Omega u_b &= 0, \\
#   \nabla_\Omega \cdot \sigma_b &= g,
# \end{aligned}
# $$
#
# with $u_b = 0$ on the outer boundary. At the joint, the potential is
# single-valued and the flux is conserved:
#
# $$
# \begin{aligned}
#   u_b &= u_\Gamma \quad \text{on } \Gamma \text{ for every } b, \\
#   \sum_b \sigma_b \cdot \nu_b &= 0 \quad \text{on } \Gamma,
# \end{aligned}
# $$
#
# where $\nu_b$ is the outward unit normal to $\Gamma$ that lies in
# $\Omega_b$, and $u_\Gamma$ is the common value of the potential there.
# The second condition is Kirchhoff's law: whatever flows out of one
# branch flows into the others. These are the interface conditions of
# {cite:t}`branching-hansbo2017composite`, who solve the primal problem on
# such composite surfaces with a Nitsche method.
#
# ### Discretization
#
# Each branch therefore gets its own Raviart-Thomas space, conforming
# up to the joint, and the conditions at the joint are imposed weakly
# with $u_\Gamma$ as a Lagrange multiplier. This is the partially
# hybridized mixed method {cite}`branching-arnold1985hybrid`: $u_\Gamma$
# lives in the normal-trace space of the Raviart-Thomas elements on
# $\Gamma$. {cite:t}`branching-arbogast2000mortar` use it to couple
# blocks that meet in pairs; here three meet on one interface.
#
# ```{admonition} Networks of curves
# :class: note dropdown
# On networks of curves, where the joint is a point, the same
# construction couples the segments at each bifurcation
# {cite}`branching-gjerde2022graphnics`, and
# {cite:t}`branching-dokken2025networks` implements it in DOLFINx in the
# same way as here. A point has no extent, so there any flux has a
# scalar trace; on a curve the multiplier must instead lie in the
# normal-trace space, on meshes that match along $\Gamma$.
# ```
#
# The unknowns live in three kinds of space:
#
# * the potential $u \in V(\Omega)$, a discontinuous Lagrange space on
#   the whole surface, since $u$ needs no continuity, not even across
#   $\Gamma$,
# * the flux $\sigma_b \in Q_b = Q(\Omega_b)$ on each branch, a
#   Raviart-Thomas space,
# * the joint potential $u_\Gamma \in Z(\Gamma)$, a discontinuous
#   Lagrange space on the joint.
#
# ```{admonition} Why one space is not enough
# :class: note dropdown
# The degrees-of-freedom of a Raviart-Thomas space on a facet are shared
# by the cells of that facet and fix the normal flux across it. On a
# surface without branches each edge has two cells, and
# {doc}`demo_mobius` shows how signing the facets makes that flux
# continuous. An edge of $\Gamma$, however, has three cells. Their
# three fluxes satisfy one condition, Kirchhoff's law, which leaves two
# independent fluxes, more than one set of facet degrees-of-freedom can
# carry. No choice of signs helps:
# {py:meth}`create_facet_orientations
# <dolfinx.mesh.Topology.create_facet_orientations>` refuses such a mesh.
# ```
#
# Testing the first equation with $\tau_b \in Q_b$ and integrating by
# parts on each branch moves the gradient from $u_b$ onto $\tau_b$, and
# leaves the boundary term
# $\int_\Gamma u_b \, \tau_b \cdot \nu_b ~\mathrm{d}s$, in which $u_b$
# is replaced by $u_\Gamma$. The weak form is: find $u \in V(\Omega)$,
# $\sigma_b \in Q_b$ and $u_\Gamma \in Z(\Gamma)$ such that
#
# $$
# \begin{aligned}
#   \int_{\Omega_b} \sigma_b \cdot \tau_b ~\mathrm{d}x
#     + \int_{\Omega_b} u \, \nabla_\Omega \cdot \tau_b ~\mathrm{d}x
#     - \int_\Gamma u_\Gamma \, \tau_b \cdot \nu_b ~\mathrm{d}s
#     &= 0 && \forall \tau_b \in Q_b, \quad b = 0, 1, 2, \\
#   \sum_b \int_{\Omega_b} (\nabla_\Omega \cdot \sigma_b) \, v
#     ~\mathrm{d}x &= \int_\Omega g \, v ~\mathrm{d}x
#     && \forall v \in V(\Omega), \\
#   -\sum_b \int_\Gamma \sigma_b \cdot \nu_b \, v_\Gamma ~\mathrm{d}s
#     &= 0 && \forall v_\Gamma \in Z(\Gamma).
# \end{aligned}
# $$
#
# The last equation is Kirchhoff's law, tested against $v_\Gamma$.
#
# ### Method of manufactured solutions
#
# Each branch is a unit square, rotated about the $z$-axis by
# $2\pi b / 3$, with $\Gamma$ the segment $0 \le z \le 1$ of that axis.
# With $r$ the distance from the axis, take
#
# $$
#   u_b = (1 - r)(1 + a_b r) \sin(\pi z).
# $$
#
# It vanishes on the outer boundary, and on $\Gamma$, where $r = 0$, it
# equals $\sin(\pi z)$ on every branch. The flux out of branch $b$ into
# the joint is $\sigma_b \cdot \nu_b = -\partial_r u_b = (1 - a_b)
# \sin(\pi z)$, so Kirchhoff's law holds when $\sum_b a_b = 3$. We take
# $a = (3, 1, -1)$: branch $0$ drains the joint, branch $2$ feeds it,
# and nothing crosses into branch $1$. The branches are flat, so
# $g = \Delta u_b$ is the usual Laplacian in $(r, z)$.
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

# The branches are at angles `angles`, the exact solution has slopes
# `slopes`, `order` is the degree of the Raviart-Thomas spaces, and
# `joint_marker` tags the edges of the joint.

angles = 2 * np.pi * np.arange(3) / 3
slopes = (3.0, 1.0, -1.0)
order = 2
joint_marker = 3

# ### Mesh-creation
#
# The surface is built by hand: each branch is a grid of `n` by `m`
# squares, cut into triangles, with the column of vertices on the axis
# shared by all three. Every edge on the axis then has three cells,
# which `max_facet_to_cell_links=3` allows.
#
# Each input cell is marked with its branch, and each edge of the joint
# with `joint_marker`. Like the cells themselves, the markers exist only
# on the process that creates the mesh, so they are sent to the
# processes that hold the entities, owned or ghost, with
# {py:func}`distribute_entity_data <dolfinx.io.distribute_entity_data>`
# and turned into {py:class}`MeshTags <dolfinx.mesh.MeshTags>`. An edge
# is given by its two vertices, in the numbering of the input.


def branching_mesh(n, m):
    """Three unit squares joined along the z-axis, tagged by branch."""
    if MPI.COMM_WORLD.rank == 0:
        z = np.linspace(0, 1, m + 1)
        r = np.linspace(0, 1, n + 1)
        points = [np.column_stack([np.zeros_like(z), np.zeros_like(z), z])]
        cells = []
        for b, theta in enumerate(angles):
            rr, zz = np.meshgrid(r[1:], z, indexing="ij")
            rr, zz = rr.ravel(), zz.ravel()
            points.append(np.column_stack([rr * np.cos(theta), rr * np.sin(theta), zz]))

            # Column 0 is the axis, numbered first and shared
            offset = (m + 1) * (1 + b * n)

            def node(i, j, offset=offset):
                return j if i == 0 else offset + (i - 1) * (m + 1) + j

            for i in range(n):
                for j in range(m):
                    cells.append([node(i, j), node(i + 1, j), node(i, j + 1)])
                    cells.append([node(i + 1, j), node(i + 1, j + 1), node(i, j + 1)])
        x = np.vstack(points)
        cells = np.array(cells, dtype=np.int64)
        markers = np.repeat(np.arange(3, dtype=np.int32), 2 * n * m)
        # The joint edges join consecutive vertices on the axis
        joint_edges = np.column_stack([np.arange(m), np.arange(1, m + 1)]).astype(np.int64)
    else:
        x = np.zeros((0, 3))
        cells = np.zeros((0, 3), dtype=np.int64)
        markers = np.zeros(0, dtype=np.int32)
        joint_edges = np.zeros((0, 2), dtype=np.int64)
    domain = ufl.Mesh(
        basix.ufl.element("Lagrange", "triangle", 1, shape=(3,), dtype=dolfinx.default_real_type)
    )
    msh = dolfinx.mesh.create_mesh(
        MPI.COMM_WORLD,
        cells,
        domain,
        x.astype(dolfinx.default_real_type),
        max_facet_to_cell_links=3,
    )
    tdim = msh.topology.dim
    local_cells, local_markers = dolfinx.io.distribute_entity_data(msh, tdim, cells, markers)
    cell_tags = dolfinx.mesh.meshtags_from_entities(
        msh, tdim, dolfinx.graph.adjacencylist(local_cells), local_markers
    )
    local_edges, local_values = dolfinx.io.distribute_entity_data(
        msh, tdim - 1, joint_edges, np.full(len(joint_edges), joint_marker, dtype=np.int32)
    )
    msh.topology.create_connectivity(tdim - 1, tdim)
    facet_tags = dolfinx.mesh.meshtags_from_entities(
        msh, tdim - 1, dolfinx.graph.adjacencylist(local_edges), local_values
    )
    return msh, cell_tags, facet_tags


# ### Submesh creation of the joint and branches
#
# Each branch is extracted from its cell tags as a submesh with
# {py:func}`create_submesh <dolfinx.mesh.create_submesh>`. A branch is
# an ordinary surface with a boundary, so its facets can be signed. The
# joint is a submesh of the tagged edges.
# Each submesh comes with an {py:class}`EntityMap
# <dolfinx.mesh.EntityMap>` relating its entities to those of the whole
# surface, which assembly needs to combine them.
#
# The coupling terms are integrals over $\Gamma$ seen from one branch at
# a time, so that $\nu_b$ is the normal of that branch. They are
# integrals over cell facets, identified as `(cell, local facet)` pairs,
# one integration domain per branch. On a branch the joint is part of
# the boundary, so the facet tags are moved to the branch with
# {py:func}`transfer_meshtags_to_submesh
# <dolfinx.mesh.transfer_meshtags_to_submesh>` and the pairs found with
# {py:func}`compute_integration_domains
# <dolfinx.fem.compute_integration_domains>`, as for any boundary. The
# integrals are taken over the whole surface, which the joint mesh is
# related to, so the cells are then mapped back to it with the entity
# map of the branch.


def solve(n, m):
    """Solve on the surface with ``n`` by ``m`` squares per branch."""
    msh, cell_tags, facet_tags = branching_mesh(n, m)
    tdim = msh.topology.dim

    branches, entity_maps, joint_facets = [], [], []
    for b in range(3):
        sub, sub_map, vertex_map = dolfinx.mesh.create_submesh(msh, tdim, cell_tags.find(b))[:3]
        # Required for a Raviart-Thomas space on a surface. Here it signs
        # no facets, as the cells of each branch share an orientation.
        sub.topology.create_facet_orientations()
        branches.append(sub)
        entity_maps.append(sub_map)

        # The joint is part of the boundary of the branch: find its
        # (cell, local facet) pairs there, and map the cells back to the
        # whole surface, where the local facet index is the same
        sub_tags = dolfinx.mesh.transfer_meshtags_to_submesh(facet_tags, sub, sub_map, vertex_map)
        sub.topology.create_connectivity(tdim - 1, tdim)
        pairs = dolfinx.fem.compute_integration_domains(
            dolfinx.fem.IntegralType.exterior_facet, sub.topology, sub_tags.find(joint_marker)
        ).reshape(-1, 2)
        pairs[:, 0] = sub_map.sub_topology_to_topology(pairs[:, 0], inverse=False)
        joint_facets.append((b, pairs.reshape(-1)))

    on_axis = facet_tags.find(joint_marker)
    joint, joint_map = dolfinx.mesh.create_submesh(msh, tdim - 1, on_axis)[:2]
    entity_maps.append(joint_map)

    # Spaces, with one Raviart-Thomas space per branch
    Q = [dolfinx.fem.functionspace(sub, ("RT", order)) for sub in branches]
    V = dolfinx.fem.functionspace(msh, ("DG", order - 1))
    Z = dolfinx.fem.functionspace(joint, ("DG", order - 1))
    W = ufl.MixedFunctionSpace(*Q, V, Z)
    *sigma, u, u_Gamma = ufl.TrialFunctions(W)
    *tau, v, v_Gamma = ufl.TestFunctions(W)

    # dx(b) is branch b, ds(b) the joint seen from branch b, whose
    # outward normal there is nu
    dx = ufl.Measure("dx", domain=msh, subdomain_data=cell_tags)
    ds = ufl.Measure("ds", domain=msh, subdomain_data=joint_facets)
    nu = ufl.FacetNormal(msh)

    x = ufl.SpatialCoordinate(msh)

    def u_exact(b):
        """The exact solution on branch ``b``."""
        r = x[0] * np.cos(angles[b]) + x[1] * np.sin(angles[b])
        return (1 - r) * (1 + slopes[b] * r) * ufl.sin(ufl.pi * x[2])

    a = 0
    for b in range(3):
        a += ufl.inner(sigma[b], tau[b]) * dx(b)
        a += ufl.inner(u, ufl.div(tau[b])) * dx(b) + ufl.inner(ufl.div(sigma[b]), v) * dx(b)
        a -= ufl.inner(u_Gamma, ufl.dot(tau[b], nu)) * ds(b)
        a -= ufl.inner(ufl.dot(sigma[b], nu), v_Gamma) * ds(b)
    g = [ufl.div(ufl.grad(u_exact(b))) for b in range(3)]
    L = ufl.inner(g[0], v) * dx(0) + ufl.inner(g[1], v) * dx(1) + ufl.inner(g[2], v) * dx(2)

    # Only the equation tested with v has a source. The others get an
    # explicit zero, so that the right-hand side has a block per space.
    a_blocked: list[list[ufl.Form | None]] = ufl.extract_blocks(a)  # type: ignore[assignment]
    L_blocked: list[ufl.Form | ufl.ZeroBaseForm] = [
        *(ufl.ZeroBaseForm((t,)) for t in tau),
        L,
        ufl.ZeroBaseForm((v_Gamma,)),
    ]

    problem = dolfinx.fem.petsc.LinearProblem(
        a_blocked,
        L_blocked,
        entity_maps=entity_maps,
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "ksp_error_if_not_converged": True,
        },
        petsc_options_prefix="branching_",
        kind="mpi",
    )
    *sigma_h, u_h, u_Gamma_h = problem.solve()

    def norm(expr, measure):
        """The L2 norm of ``expr`` over ``measure``."""
        # UFL estimates the degree of sin as that of its argument plus
        # two, too low to measure the error of a high-degree space: at
        # the few points it gives, the multiplier superconverges.
        integral = dolfinx.fem.form(
            ufl.inner(expr, expr) * measure(metadata={"quadrature_degree": 2 * order + 6}),
            entity_maps=entity_maps,
        )
        return np.sqrt(abs(msh.comm.allreduce(dolfinx.fem.assemble_scalar(integral))))

    x_joint = ufl.SpatialCoordinate(joint)
    errors = {
        "L2(u)": np.hypot.reduce([norm(u_h - u_exact(b), dx(b)) for b in range(3)]),
        "L2(sigma)": np.hypot.reduce(
            [norm(sigma_h[b] - ufl.grad(u_exact(b)), dx(b)) for b in range(3)]
        ),
        "L2(u_Gamma)": norm(u_Gamma_h - ufl.sin(ufl.pi * x_joint[2]), ufl.dx(domain=joint)),
    }
    return msh, entity_maps, sigma_h, u_h, errors


# ### Results
#
# The errors converge at the optimal rate, $O(h^k)$ for Raviart-Thomas
# of degree $k$, in the flux, the potential and the multiplier alike.
# Each error is followed by its rate, computed from the previous mesh.

# +
resolutions = [4, 8, 16]
columns = ["L2(u)", "L2(sigma)", "L2(u_Gamma)"]
results = [solve(n, n) for n in resolutions]
if MPI.COMM_WORLD.rank == 0:
    print(f"{'n':>4}" + "".join(f"{c:>14}{'rate':>6}" for c in columns))
    for i, (n, (*_, errors)) in enumerate(zip(resolutions, results, strict=True)):
        row = f"{n:>4}"
        for c in columns:
            if i == 0:
                rate = ""
            else:
                ratio = results[i - 1][-1][c] / errors[c]
                rate = f"{np.log(ratio) / np.log(n / resolutions[i - 1]):.2f}"
            row += f"{errors[c]:>14.3e}{rate:>6}"
        print(row)
# -

# ### Visualisation: interpolation from submeshes to the parent mesh
#
# The fluxes live on the branches. To view them together they are
# interpolated into one discontinuous Lagrange space on the whole
# surface, branch by branch, mapping each branch's cells to those of
# the surface with its entity map.

if dolfinx.has_adios2:
    msh, entity_maps, sigma_h, u_h, _ = results[-1]
    S = dolfinx.fem.functionspace(msh, ("DG", order - 1, (3,)))
    sigma_out = dolfinx.fem.Function(S, name="sigma")
    for sigma_b, sub_map in zip(sigma_h, entity_maps[:3], strict=True):
        sub_cell_map = sigma_b.function_space.mesh.topology.index_map(2)
        sub_cells = np.arange(sub_cell_map.size_local + sub_cell_map.num_ghosts, dtype=np.int32)
        sigma_out.interpolate(
            sigma_b,
            cells0=sub_cells,
            cells1=sub_map.sub_topology_to_topology(sub_cells, inverse=False),
        )
    with dolfinx.io.VTXWriter(msh.comm, "branching_manifold.bp", [sigma_out, u_h]) as writer:
        writer.write(0.0)

# ## References
# ```{bibliography}
#    :filter: cited
#    :labelprefix:
#    :keyprefix: branching-
# ```
