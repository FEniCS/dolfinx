# Copyright (C) 2026 DOLFINx contributors
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

# # Linear Naghdi shells
#
# ```{admonition} Download sources
# :class: download
# * {download}`Python script <./demo_shell.py>`
# * {download}`Jupyter notebook <./demo_shell.ipynb>`
# ```
#
# This demo solves a linear Naghdi shell problem: a cylindrical panel,
# clamped at one axial end, under inward normal pressure. It illustrates
# an analytic surface parametrisation, a mixed displacement/director
# space, and membrane, bending and transverse shear energies. The same
# panel is then solved again with a TDNNS formulation of the bending
# energy, which carries the bending moment as an independent unknown.
# A second example uses a Gmsh surface mesh of a flared, wavy panel
# with bores, a rounded slot and a scalloped edge, refined around the
# cutouts. That panel is solved both with displacements and rotations
# and with the TDNNS shell of Neunteufel and Schöberl, which handles a
# general curved surface whose discrete normal jumps between elements.
# This demo requires Gmsh.
#
# The model includes transverse shear deformation and assumes small
# displacements and rotations. Cubic bubble enrichment and partial
# selective reduced integration (PSRI) alleviate membrane and shear
# locking. This demo uses the practical PSRI variant of
# {cite:t}`hale2018shells`; it does not establish thickness-uniform
# convergence for this geometry.
# Running on multiple MPI ranks requires PETSc with MUMPS or SuperLU_DIST.
#
# ## Geometry and kinematics
#
# We mesh a rectangle in coordinates $(s,z)$, where $s$ is circumferential
# arc length and $z$ is the axial coordinate. The reference midsurface is
#
# $$
# \Phi(s,z)=(R\sin(s/R),z,R\cos(s/R)),\qquad
# -\pi R/4\leq s\leq\pi R/4,\quad 0\leq z\leq L.
# $$
#
# Its tangent vectors $a_1=\partial_s\Phi$ and
# $a_2=\partial_z\Phi$ are orthonormal, so the surface metric is the
# identity and the surface area element is $\mathrm{d}s\,\mathrm{d}z$.
# Choose the outward director $n=(\sin(s/R),0,\cos(s/R))$.
# The unknowns are the Cartesian displacement $u\in\mathbb{R}^3$ and
# two tangential director components $\beta\in\mathbb{R}^2$, giving
# $\eta=\beta_1 a_1+\beta_2 a_2$. Thus $\eta\cdot n=0$.
# These components describe the change of the director, rather than
# rotations about the tangent axes; no drilling rotation is introduced.
#
# With $A=\nabla\Phi$, linearising the three-dimensional kinematics
# $\Phi+u+\zeta(n+\eta)$ gives the strain measures
#
# $$
# \begin{aligned}
# e(u)&=\operatorname{sym}(A^T\nabla u),\\
# k(u,\beta)&=\operatorname{sym}
#     (A^T\nabla\eta+(\nabla u)^T\nabla n),\\
# \gamma(u,\beta)&=A^T\eta+(\nabla u)^T n.
# \end{aligned}
# $$
#
# All gradients are with respect to $(s,z)$. In particular,
# differentiating $\eta$ also differentiates the tangent frame.
# The curvature term involving $\nabla n$ must be retained in $k$.
#
# ## Variational problem
#
# For isotropic plane stress, set $\mu=E/[2(1+\nu)]$ and
# $\lambda_{\mathrm{ps}}=E\nu/(1-\nu^2)$, and define
# $C(e)=2\mu e+\lambda_{\mathrm{ps}}\operatorname{tr}(e)I$.
# The displacement and director change vanish at $z=0$; all other
# edges have natural, traction-free boundary conditions. Find $(u,\beta)$
# such that, for all admissible $(v,\delta)$,
#
# $$
# \int_\omega\left[t C(e(u)):e(v)
# +\frac{t^3}{12}C(k(u,\beta)):k(v,\delta)
# +\kappa_s\mu t\,\gamma(u,\beta)\cdot\gamma(v,\delta)\right]
# \,\mathrm{d}s\,\mathrm{d}z
# =\int_\omega(-p n)\cdot v\,\mathrm{d}s\,\mathrm{d}z,
# $$
#
# where $t$ is thickness, $p$ is pressure per unit midsurface area, and
# $\kappa_s=5/6$ is the shear correction factor. The pressure direction
# is fixed in the reference configuration.

# +
from pathlib import Path

from mpi4py import MPI
from petsc4py import PETSc

import gmsh
import numpy as np

import basix
import ufl
from basix.ufl import blocked_element, element, enriched_element, mixed_element
from dolfinx import default_real_type, fem, mesh
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import gmsh as gmshio
from dolfinx.io import vtkhdf

comm = MPI.COMM_WORLD
R, length = 1.0, 2.0
thickness, young, poisson = 0.1, 1.0e7, 0.3
pressure = 1.0
mu = young / (2 * (1 + poisson))
lambda_ps = young * poisson / (1 - poisson**2)
shear_correction = 5 / 6

msh = mesh.create_rectangle(
    comm,
    ((-np.pi * R / 4, 0.0), (np.pi * R / 4, length)),
    (12, 20),
    cell_type=mesh.CellType.triangle,
)
cell = msh.basix_cell()
p2 = element("Lagrange", cell, 2, dtype=default_real_type)
bubble = element("Bubble", cell, 3, dtype=default_real_type)
U = blocked_element(enriched_element([p2, bubble]), shape=(3,))
B = element("Lagrange", cell, 2, shape=(2,), dtype=default_real_type)
W = fem.functionspace(msh, mixed_element([U, B]))
V, _ = W.sub(0).collapse()
Q, _ = W.sub(1).collapse()

x = ufl.SpatialCoordinate(msh)
angle = x[0] / R
position = ufl.as_vector((R * ufl.sin(angle), x[1], R * ufl.cos(angle)))
tangent = ufl.as_vector((ufl.cos(angle), 0, -ufl.sin(angle)))
axis = ufl.as_vector((0, 1, 0))
normal = ufl.as_vector((ufl.sin(angle), 0, ufl.cos(angle)))
A = ufl.grad(position)
dx = ufl.Measure("dx", domain=msh, metadata={"quadrature_degree": 6})
dx_reduced = ufl.Measure("dx", domain=msh, metadata={"quadrature_degree": 2})
h = ufl.CellDiameter(msh)
alpha = ufl.min_value(1.0, (thickness / h) ** 2)
# -

# ## Partial selective reduced integration
#
# Displacement uses $[P_2+B_3]^3$, with one interior cubic bubble per
# component and triangle; director components use $[P_2]^2$. The bubbles
# vanish on edges and therefore do not change the essential boundary
# conditions. Split the membrane and shear stiffness contributions:
#
# $$
# a_h=a_b+\alpha(a_m+a_s)
# +(1-\alpha)(a_m^{\mathrm{red}}+a_s^{\mathrm{red}}),
# \qquad \alpha|_K=\min\{1,(t/h_K)^2\}.
# $$
#
# Here $h_K$ is the cell diameter in the arc-length parameter domain.
# Bending uses degree-six quadrature throughout; the reduced membrane
# and shear terms use degree-two quadrature. Load integrals also retain
# degree-six quadrature. The fully integrated fraction controls modes
# missed by reduced integration, and clipping $\alpha$ keeps both weights
# nonnegative when $t>h_K$. Degree-two quadrature on these triangles uses
# three points; a one-point rule can introduce spurious modes.
# This follows the FEniCS-Shells implementation {cite}`hale2018shells`.


def strains(u, beta):
    """Return membrane, bending and shear strains in the tangent frame."""
    eta = beta[0] * tangent + beta[1] * axis
    e = ufl.sym(A.T * ufl.grad(u))
    k = ufl.sym(A.T * ufl.grad(eta) + ufl.grad(u).T * ufl.grad(normal))
    gamma = A.T * eta + ufl.grad(u).T * normal
    return e, k, gamma


def stress(e):
    """Apply the isotropic plane-stress constitutive tensor."""
    return 2 * mu * e + lambda_ps * ufl.tr(e) * ufl.Identity(2)


def stiffness(u, beta, v, delta, *, psri=True):
    """Return the shell stiffness form with optional PSRI."""
    e, k, gamma = strains(u, beta)
    ev, kv, gammav = strains(v, delta)
    membrane_shear = thickness * ufl.inner(
        stress(e), ev
    ) + shear_correction * mu * thickness * ufl.inner(gamma, gammav)
    bending = thickness**3 / 12 * ufl.inner(stress(k), kv)
    if psri:
        return (bending + alpha * membrane_shear) * dx + (1 - alpha) * membrane_shear * dx_reduced
    else:
        return (bending + membrane_shear) * dx


def integral(expression):
    """Assemble a scalar integral over all MPI ranks."""
    value = fem.assemble_scalar(fem.form(expression))
    return comm.allreduce(value, op=MPI.SUM)


# ## Verification
#
# First check that a rigid rotation has zero strain. For a constant
# rotation vector $w$, take $u=w\times\Phi$ and $\eta=w\times n$.
# This checks the changing tangent frame and the curvature term in $k$.

# +
omega = ufl.as_vector((0.2, -0.3, 0.4))
u_rigid = ufl.cross(omega, position)
eta_rigid = ufl.cross(omega, normal)
beta_rigid = ufl.as_vector((ufl.dot(eta_rigid, tangent), ufl.dot(eta_rigid, axis)))
rigid_strains = strains(u_rigid, beta_rigid)
rigid_norm = integral(sum(ufl.inner(e, e) for e in rigid_strains) * dx).real
assert rigid_norm < 100 * np.finfo(default_real_type).eps ** 2
# -

# Next solve a manufactured problem with exact displacement
# $u_{\mathrm{ex}}=(0,c z^2,0)$ and $\beta_{\mathrm{ex}}=0$.
# Its only nonzero strain is $e_{22}=2cz$. Prescribe this displacement
# and zero director change on every edge. Integration by parts gives
# the required surface force
#
# $$
# f_{\mathrm{ex}}=2ct\left[
# \frac{\lambda_{\mathrm{ps}}z}{R}n
# -(\lambda_{\mathrm{ps}}+2\mu)a_2\right].
# $$
#
# The normal force balances the circumferential membrane stress. Since
# the exact fields lie in the quadratic finite element space, the error
# should be close to solver and quadrature accuracy.
# This check uses full integration to verify the continuous shell model.
# Reduced integration is not exact for the curved-frame terms, even when
# the displacement is polynomial in the parameter coordinates.

# +
u, beta = ufl.TrialFunctions(W)
v, delta = ufl.TestFunctions(W)
a = stiffness(u, beta, v, delta)
zero_beta = fem.Function(Q, dtype=PETSc.ScalarType)
exact = fem.Function(V, dtype=PETSc.ScalarType)
c = 1.0e-3
exact.interpolate(lambda x: np.vstack((np.zeros_like(x[1]), c * x[1] ** 2, np.zeros_like(x[1]))))
msh.topology.create_connectivity(1, 2)
boundary = mesh.exterior_facet_indices(msh.topology)
bc_exact = fem.dirichletbc(exact, fem.locate_dofs_topological((W.sub(0), V), 1, boundary), W.sub(0))
bc_beta = fem.dirichletbc(
    zero_beta, fem.locate_dofs_topological((W.sub(1), Q), 1, boundary), W.sub(1)
)
f_exact = 2 * c * thickness * (lambda_ps * x[1] / R * normal - (lambda_ps + 2 * mu) * axis)
options = {"ksp_type": "preonly", "pc_type": "lu", "ksp_error_if_not_converged": True}
if comm.size > 1:
    if PETSc.Sys().hasExternalPackage("mumps") and PETSc.IntType == np.int32:
        options["pc_factor_mat_solver_type"] = "mumps"
    elif PETSc.Sys().hasExternalPackage("superlu_dist"):
        options["pc_factor_mat_solver_type"] = "superlu_dist"
    else:
        raise RuntimeError("Parallel shell demo requires MUMPS or SuperLU_DIST.")
verification = LinearProblem(
    stiffness(u, beta, v, delta, psri=False),
    ufl.inner(f_exact, v) * dx,
    bcs=[bc_exact, bc_beta],
    petsc_options_prefix="demo_shell_verification_",
    petsc_options=options,
)
verified = verification.solve()
assert isinstance(verified, fem.Function)
u_verified, beta_verified = ufl.split(verified)
error = integral(ufl.inner(u_verified - exact, u_verified - exact) * dx).real
scale = integral(ufl.inner(exact, exact) * dx).real
rotation_error = integral(ufl.inner(beta_verified, beta_verified) * dx).real
tolerance = np.sqrt(np.finfo(default_real_type).eps)
assert np.sqrt(error / scale) < tolerance
assert np.sqrt(rotation_error / scale) < tolerance
if comm.rank == 0:
    print(f"Manufactured displacement relative L2 error: {np.sqrt(error / scale):.3e}")
# -

# ## Pressure-loaded panel
#
# Clamp the end $z=0$ and solve with the normal pressure load.
# The work balance below uses the PSRI energy of the discrete problem.
# It is an identity, so its residual measures the round-off of the
# direct solve amplified by the conditioning of a thin shell. That
# residual scales with the machine epsilon rather than its square root,
# so give it a tolerance of its own. The embedded problem below is far
# more sensitive again, and single precision loses the identity
# entirely, so that one is checked in double precision only.

# +
clamped = mesh.locate_entities_boundary(msh, 1, lambda x: np.isclose(x[1], 0.0))
zero_u = fem.Function(V, dtype=PETSc.ScalarType)
bcs = [
    fem.dirichletbc(zero_u, fem.locate_dofs_topological((W.sub(0), V), 1, clamped), W.sub(0)),
    fem.dirichletbc(zero_beta, fem.locate_dofs_topological((W.sub(1), Q), 1, clamped), W.sub(1)),
]
work_tolerance = 1e6 * np.finfo(default_real_type).eps
double_precision = np.finfo(default_real_type).eps < 1e-10
load = ufl.inner(-pressure * normal, v) * dx
problem = LinearProblem(a, load, bcs=bcs, petsc_options_prefix="demo_shell_", petsc_options=options)
solution = problem.solve()
assert isinstance(solution, fem.Function)
uh, betah = ufl.split(solution)
energy_twice = integral(stiffness(uh, betah, uh, betah)).real
work = integral(ufl.inner(-pressure * normal, uh) * dx).real
assert np.isfinite(energy_twice) and energy_twice > 0
assert abs(energy_twice - work) / energy_twice < work_tolerance
if comm.rank == 0:
    print(f"Elastic energy: {energy_twice / 2:.6e}")
    print(f"Relative work balance error: {abs(energy_twice - work) / energy_twice:.3e}")
# -

# ## A TDNNS formulation of the bending energy
#
# The formulation above puts the bending energy in terms of the
# curvature $k(u,\beta)$, so the director change must lie in $H^1$. The
# tangential-displacement normal-normal-stress (TDNNS) method instead
# carries the bending moment $\sigma$ as an independent unknown and
# weakens both continuity requirements: $\sigma$ need only have a
# continuous normal-normal component $\sigma_{nn}$, and $\beta$ only a
# continuous tangential component $\beta_t$. See
# {cite:t}`pechstein2017tdnns`.
#
# Because the tangent frame here is orthonormal and parallel along the
# surface, $a_i\cdot\partial_j a_k=0$, so $A^T\nabla\eta=\nabla\beta$
# and the curvature splits as
#
# $$
# k(u,\beta)=\operatorname{sym}(\nabla\beta)
#   +\operatorname{sym}\!\left((\nabla u)^T\nabla n\right).
# $$
#
# Only the first term differentiates $\beta$. Replacing the bending
# energy by its Hellinger--Reissner complementary form, with
# $C_b=(t^3/12)C$, gives
#
# $$
# -\frac{1}{2}\int_\omega C_b^{-1}\sigma:\sigma
# +\int_\omega\sigma:k(u,\beta),
# $$
#
# and the term that differentiates $\beta$ is given its distributional
# meaning element by element,
#
# $$
# \langle\sigma,\nabla\beta\rangle_h=\sum_T\left[
# \int_T\sigma:\nabla\beta\,\mathrm{d}x
# -\int_{\partial T}\sigma_{nn}\,\beta_n\,\mathrm{d}s\right].
# $$
#
# This is well defined for the chosen spaces: $\sigma_{nn}$ is
# single-valued on a facet, so the element-boundary term pairs it with
# the jump of $\beta_n$, which is the component $\beta$ is free to jump
# in. Writing the volume term as $\int\operatorname{div}\sigma\cdot\beta$
# instead gives a *different* operator, differing by a
# $\sigma_{nt}\beta_t$ jump, whose discrete solution collapses to zero
# under refinement.
#
# Discretise $\sigma$ with Hellan--Herrmann--Johnson elements and
# $\beta$ with second-kind Nédélec elements. The result is a symmetric
# saddle-point system, so it needs a direct solver.
#
# The clamped edge fixes $u$ and $\beta_t$; $\sigma$ is left free there,
# and the boundary term $-\int\sigma_{nn}\beta_n$ weakly enforces
# $\beta_n=0$. On the free edges the natural condition $\sigma_{nn}=0$
# is imposed on the moment degrees of freedom instead.
#
# TDNNS removes shear locking by construction, but this panel is
# membrane-dominated and the membrane term locks on its own. The bubble
# enrichment and PSRI of the previous section are therefore kept for the
# membrane and shear energies. With them the two discretisations agree
# closely at this thickness; as the panel is made thinner, TDNNS is the
# more accurate of the two on coarse meshes.

# +
moment_element = element("HHJ", cell, 2, dtype=default_real_type)
rotation_element = element("N2E", cell, 2, dtype=default_real_type)
W_tdnns = fem.functionspace(msh, mixed_element([moment_element, rotation_element, U]))
S_tdnns, _ = W_tdnns.sub(0).collapse()
Q_tdnns, _ = W_tdnns.sub(1).collapse()
V_tdnns, _ = W_tdnns.sub(2).collapse()


n_facet = ufl.FacetNormal(msh)


def compliance(moment):
    """Apply the inverse of the bending constitutive tensor."""
    trace_factor = lambda_ps / (2 * mu + 2 * lambda_ps)
    deviator = moment - trace_factor * ufl.tr(moment) * ufl.Identity(2)
    return 12 / thickness**3 * deviator / (2 * mu)


def displacement_curvature(u):
    """Return the part of the curvature that depends on displacement."""
    return ufl.sym(ufl.grad(u).T * ufl.grad(normal))


def moment_pairing(moment, beta, *, moment_is_test):
    """Return the distributional pairing of a moment with grad(beta).

    ``ufl.inner`` conjugates its second argument, which in a
    sesquilinear form must be the test function, so the two arguments
    are ordered according to which one carries it.
    """
    facet_normal = n_facet("+")
    moment_nn = ufl.dot(ufl.dot(ufl.avg(moment), facet_normal), facet_normal)
    jump_beta_n = ufl.dot(beta("+"), facet_normal) + ufl.dot(beta("-"), n_facet("-"))
    boundary_nn = ufl.dot(ufl.dot(moment, n_facet), n_facet)
    boundary_beta_n = ufl.dot(beta, n_facet)
    if moment_is_test:
        volume, facet = ufl.inner(ufl.grad(beta), moment), ufl.inner(jump_beta_n, moment_nn)
        boundary = ufl.inner(boundary_beta_n, boundary_nn)
    else:
        volume, facet = ufl.inner(moment, ufl.grad(beta)), ufl.inner(moment_nn, jump_beta_n)
        boundary = ufl.inner(boundary_nn, boundary_beta_n)
    return volume * dx - facet * ufl.dS - boundary * ufl.ds


sigma_t, beta_t, u_t = ufl.TrialFunctions(W_tdnns)
tau_t, delta_t, v_t = ufl.TestFunctions(W_tdnns)
membrane_t = ufl.sym(A.T * ufl.grad(u_t))
membrane_v = ufl.sym(A.T * ufl.grad(v_t))
shear_t = A.T * (beta_t[0] * tangent + beta_t[1] * axis) + ufl.grad(u_t).T * normal
shear_v = A.T * (delta_t[0] * tangent + delta_t[1] * axis) + ufl.grad(v_t).T * normal
membrane_shear = thickness * ufl.inner(
    stress(membrane_t), membrane_v
) + shear_correction * mu * thickness * ufl.inner(shear_t, shear_v)
a_tdnns = (
    -ufl.inner(compliance(sigma_t), tau_t) * dx
    + moment_pairing(tau_t, beta_t, moment_is_test=True)
    + ufl.inner(displacement_curvature(u_t), tau_t) * dx
    + moment_pairing(sigma_t, delta_t, moment_is_test=False)
    + ufl.inner(sigma_t, displacement_curvature(v_t)) * dx
    + alpha * membrane_shear * dx
    + (1 - alpha) * membrane_shear * dx_reduced
)
# -

# The moment degrees of freedom attached to a boundary facet are exactly
# its normal-normal moments, so constraining them on the free edges
# imposes $\sigma_{nn}=0$ there.

# +
free_edges = np.setdiff1d(mesh.exterior_facet_indices(msh.topology), clamped)
bcs_tdnns = [
    fem.dirichletbc(
        fem.Function(S_tdnns, dtype=PETSc.ScalarType),
        fem.locate_dofs_topological((W_tdnns.sub(0), S_tdnns), 1, free_edges),
        W_tdnns.sub(0),
    ),
    fem.dirichletbc(
        fem.Function(Q_tdnns, dtype=PETSc.ScalarType),
        fem.locate_dofs_topological((W_tdnns.sub(1), Q_tdnns), 1, clamped),
        W_tdnns.sub(1),
    ),
    fem.dirichletbc(
        fem.Function(V_tdnns, dtype=PETSc.ScalarType),
        fem.locate_dofs_topological((W_tdnns.sub(2), V_tdnns), 1, clamped),
        W_tdnns.sub(2),
    ),
]
problem_tdnns = LinearProblem(
    a_tdnns,
    ufl.inner(-pressure * normal, v_t) * dx,
    bcs=bcs_tdnns,
    petsc_options_prefix="demo_shell_tdnns_",
    petsc_options=options,
)
solution_tdnns = problem_tdnns.solve()
assert isinstance(solution_tdnns, fem.Function)
u_mixed = ufl.split(solution_tdnns)[2]
# -

# Both discretisations approximate the same shell model, so the work
# done by the pressure must agree to discretisation error. On this mesh
# they differ by about $5\times10^{-5}$. The tolerance also has to
# cover the round-off of the single-precision direct solve of this
# indefinite system, which is around $3\times10^{-4}$.

# +
work_tdnns = integral(ufl.inner(-pressure * normal, u_mixed) * dx).real
work_naghdi = integral(ufl.inner(-pressure * normal, uh) * dx).real
assert abs(work_tdnns - work_naghdi) / abs(work_naghdi) < 1e-2
if comm.rank == 0:
    print(f"TDNNS external work: {work_tdnns:.6e}")
    print(
        f"Relative difference from the displacement formulation: "
        f"{abs(work_tdnns - work_naghdi) / abs(work_naghdi):.3e}"
    )
# -

# ## Output
#
# Write a cubic visualization mesh on the physical cylinder, with the
# displacement as point data. Equispaced cubic nodes preserve the bubble
# contribution and match VTK's Lagrange node convention. Each cell has
# its own nodes, so the interpolated displacement is preserved cellwise.
# The VTKHDF writer names its point-data array `u`.
#
# Open `out_shell/shell.vtkhdf` in ParaView, click **Apply**, and use
# **Surface** representation and **Reset Camera**. Apply **Warp By Vector**
# to `u`, increasing the scale factor to make the displacement visible.


# +
def keep_cells_on_rank(
    comm: MPI.Comm, nparts: int, centroids: np.ndarray, weights: np.ndarray | None
) -> np.ndarray:
    """Keep visualization cells on their source rank."""
    return np.full(centroids.shape[0], comm.rank, dtype=np.int32)


def write_shell_vtkhdf(filename: Path, displacement: fem.Function, coordinates: np.ndarray) -> None:
    """Write cubic geometry and displacement with matching nodes."""
    V_output = displacement.function_space
    source = V_output.mesh
    index_map = V_output.dofmap.index_map
    num_points = index_map.size_local
    start = index_map.local_range[0]
    num_cells = source.topology.index_map(source.topology.dim).size_local
    cells = V_output.dofmap.list[:num_cells].astype(np.int64) + start
    output_domain = ufl.Mesh(
        element(
            "Lagrange",
            source.basix_cell(),
            3,
            lagrange_variant=basix.LagrangeVariant.equispaced,
            shape=(3,),
            dtype=default_real_type,
        )
    )
    output_mesh = mesh.create_mesh(
        source.comm,
        cells,
        output_domain,
        coordinates[:num_points],
        partitioner=mesh.create_geometric_cell_partitioner(keep_cells_on_rank),
    )
    # Mesh construction can reorder nodes. Recover the input ordering.
    num_output_points = output_mesh.geometry.index_map().size_local
    input_nodes = output_mesh.geometry.input_global_indices[:num_output_points] - start
    assert np.all((input_nodes >= 0) & (input_nodes < num_points))
    values = displacement.x.array.reshape(-1, 3)[input_nodes].real.copy()
    vtkhdf.write_mesh(filename, output_mesh)
    vtkhdf.write_point_data(filename, output_mesh, values.ravel(), 0.0)


output_element = element(
    "DG",
    msh.basix_cell(),
    3,
    lagrange_variant=basix.LagrangeVariant.equispaced,
    shape=(3,),
    dtype=default_real_type,
)
V_out = fem.functionspace(msh, output_element)
displacement = fem.Function(V_out, dtype=PETSc.ScalarType)
displacement.interpolate(solution.sub(0))
reference_position = fem.Function(V_out, dtype=default_real_type)
reference_position.interpolate(
    fem.Expression(position, V_out.element.interpolation_points, dtype=default_real_type)
)
out_folder = Path("out_shell")
out_folder.mkdir(exist_ok=True)
write_shell_vtkhdf(
    out_folder / "shell.vtkhdf", displacement, reference_position.x.array.reshape(-1, 3)
)
# -

# ## Starting from a Gmsh surface mesh
#
# Construct a separate shell directly in physical space using OpenCASCADE.
# Revolve a spline profile through a quarter turn to make a flared panel
# with a waist and varying curvature in both surface directions. Cut
# three circular bores, a rounded slot and a scallop at the free end.
# These cuts are made by solids passing through the surface along the
# Cartesian $z$ direction; their rims follow the curved CAD surface.
# A distance-based size field refines around the openings and free rim.
#
# The quadratic triangles approximate the CAD surface: their nodes lie on
# it, while points between nodes need not. The mesh has topological
# dimension two and geometric dimension three. Tag the shell, clamped
# arc, opening rims and free edges as physical groups. Only the clamped
# arc has essential boundary conditions; all other edges are traction-free.
# The TDNNS formulation below integrates over interior facets, which in
# parallel requires cells ghosted across facets, so the mesh is built with
# `shared_facet` ghosting rather than the default.

# +
gmsh.initialize()
cad_area = 0.0
if comm.rank == 0:
    gmsh.model.add("flared_perforated_shell")
    occ = gmsh.model.occ
    profile_points = [
        occ.addPoint(-radius / np.sqrt(2), y, radius / np.sqrt(2))
        for y, radius in zip(
            length * np.array([0, 0.18, 0.38, 0.60, 0.82, 1]),
            R * np.array([1, 0.82, 1.18, 0.86, 1.12, 1.35]),
            strict=True,
        )
    ]
    profile = occ.addSpline(profile_points)
    revolved = occ.revolve([(1, profile)], 0, 0, 0, 0, 1, 0, np.pi / 2)
    panel = [(dim, tag) for dim, tag in revolved if dim == 2]
    tools = [
        (3, occ.addCylinder(x * R, y * length, 0, 0, 0, 3 * R, radius * R))
        for x, y, radius in [(-0.35, 0.28, 0.13), (0.35, 0.38, 0.16), (0.42, 0.55, 0.10)]
    ]
    slot_radius = 0.11 * R
    slot_y = 0.72 * length
    slot_start, slot_end = -0.15 * R, 0.20 * R
    slot_box = occ.addBox(
        slot_start, slot_y - slot_radius, 0, slot_end - slot_start, 2 * slot_radius, 3 * R
    )
    slot_caps = [
        (3, occ.addCylinder(x, slot_y, 0, 0, 0, 3 * R, slot_radius)) for x in (slot_start, slot_end)
    ]
    slot, _ = occ.fuse([(3, slot_box)], slot_caps)
    scallop = occ.addCylinder(0, 1.03 * length, 0, 0, 0, 3 * R, 0.25 * R)
    perforated, _ = occ.cut(panel, tools + slot + [(3, scallop)])
    occ.synchronize()
    surfaces = [tag for dim, tag in perforated if dim == 2]
    cad_area = sum(occ.getMass(2, tag) for tag in surfaces)
    boundary_curves = gmsh.model.getBoundary(perforated, oriented=False)
    clamped_curves, opening_curves, free_curves = [], [], []
    for dim, tag in boundary_curves:
        _, y_min, _, _, y_max, _ = gmsh.model.getBoundingBox(dim, tag)
        if np.isclose(y_min, 0, atol=1e-6) and np.isclose(y_max, 0, atol=1e-6):
            clamped_curves.append(tag)
        elif y_min > 1e-6 and y_max < length - 1e-6:
            opening_curves.append(tag)
        else:
            free_curves.append(tag)
    assert clamped_curves and opening_curves
    gmsh.model.addPhysicalGroup(2, surfaces, tag=1)
    gmsh.model.addPhysicalGroup(1, clamped_curves, tag=2)
    gmsh.model.addPhysicalGroup(1, opening_curves, tag=3)
    gmsh.model.addPhysicalGroup(1, free_curves, tag=4)
    gmsh.model.setPhysicalName(2, 1, "shell")
    gmsh.model.setPhysicalName(1, 2, "clamped")
    gmsh.model.setPhysicalName(1, 3, "openings")
    gmsh.model.setPhysicalName(1, 4, "free_edges")

    distance = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(distance, "CurvesList", opening_curves + free_curves)
    gmsh.model.mesh.field.setNumber(distance, "Sampling", 100)
    threshold = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(threshold, "InField", distance)
    gmsh.model.mesh.field.setNumber(threshold, "SizeMin", 0.035 * R)
    gmsh.model.mesh.field.setNumber(threshold, "SizeMax", 0.10 * R)
    gmsh.model.mesh.field.setNumber(threshold, "DistMin", 0.08 * R)
    gmsh.model.mesh.field.setNumber(threshold, "DistMax", 0.35 * R)
    gmsh.model.mesh.field.setAsBackgroundMesh(threshold)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(2)
cad_area = comm.bcast(cad_area, root=0)
mesh_data = gmshio.model_to_mesh(
    gmsh.model, comm, 0, gdim=3, ghost_mode=mesh.GhostMode.shared_facet
)
gmsh.finalize()
surface = mesh_data.mesh
assert surface.topology.dim == 2 and surface.geometry.dim == 3
facet_tags = mesh_data.facet_tags
assert facet_tags is not None
# -

# ### Geometry and rotations
#
# The $3\times2$ cell Jacobian supplies two tangent vectors. Their cross
# product gives the normal, whose sign is chosen to point away from the
# axis of revolution. Unlike a normal on planar triangles, this normal
# varies within a quadratic cell. Differentiating it retains the
# curvature.
# We form an orthonormal tangent basis $T=[t_1,t_2]$ using the axial
# direction; the profile stays away from the axis and its normal is
# never parallel to the axial direction.
#
# Use three Cartesian rotation components $\theta$, with director change
# $\eta=\theta\times n$. Cartesian coefficients do not depend on the
# tangent basis chosen by a cell. The mesh is only $C^0$, so its normal
# can differ across cell edges; strains use the cellwise geometry.
# Surface gradients $\nabla_\Sigma$ are supplied by UFL on the embedded
# mesh, and the projected strain measures are
#
# $$
# \begin{aligned}
# e&=\operatorname{sym}(T^T\nabla_\Sigma u\,T),\\
# k&=\operatorname{sym}\left[T^T\left(\nabla_\Sigma\eta
#       +(\nabla_\Sigma u)^T\nabla_\Sigma n\right)T\right],\\
# \gamma&=T^T\left[\eta+(\nabla_\Sigma u)^T n\right].
# \end{aligned}
# $$
#
# Rotation about $n$ does not change the director. Constrain this drilling
# component to the in-plane displacement spin using
# $d=\theta\cdot n-\tfrac12\operatorname{curl}_\Sigma u\cdot n$.
# Add $\int_\Sigma (Et^3/h_K^2)d(u,\theta)\overline{d(v,\psi)}$
# to the bilinear form, using full quadrature. This term vanishes for a
# rigid rotation. For the curved mesh, compute cell diameters using
# {py:meth}`Mesh.h <dolfinx.mesh.Mesh.h>` and store them in a cellwise
# constant function for PSRI and drilling. The stabilisation follows the
# linear shell tour of {cite:t}`bleyer2024tours`.
#
# Ordering is undefined for complex numbers, so UFL rejects a comparison
# unless it can see that both sides are real. Geometry and cell sizes are
# real, but UFL cannot infer this through a square root or a coefficient,
# so wrap those comparisons in `ufl.real`. The cell-size function itself
# uses the form scalar type, since a form and its coefficients must agree.
# The demo then also runs in a complex-scalar build.

# +
X = ufl.SpatialCoordinate(surface)
J = ufl.Jacobian(surface)
j0 = ufl.as_vector((J[0, 0], J[1, 0], J[2, 0]))
j1 = ufl.as_vector((J[0, 1], J[1, 1], J[2, 1]))
cross = ufl.cross(j0, j1)
n_cell = cross / ufl.sqrt(ufl.dot(cross, cross))
radial = ufl.as_vector((X[0], 0, X[2]))
n_surface = ufl.conditional(ufl.gt(ufl.real(ufl.dot(n_cell, radial)), 0), n_cell, -n_cell)
t1 = ufl.cross(axis, n_surface)
t1 = t1 / ufl.sqrt(ufl.dot(t1, t1))
t2 = ufl.cross(n_surface, t1)
T = ufl.as_matrix(((t1[0], t2[0]), (t1[1], t2[1]), (t1[2], t2[2])))

surface_rotation_element = element(
    "Lagrange", surface.basix_cell(), 2, shape=(3,), dtype=default_real_type
)
W_surface = fem.functionspace(surface, mixed_element([U, surface_rotation_element]))
V_surface, _ = W_surface.sub(0).collapse()
Q_surface, _ = W_surface.sub(1).collapse()
dx_surface = ufl.Measure("dx", domain=surface, metadata={"quadrature_degree": 6})
dx_surface_reduced = ufl.Measure("dx", domain=surface, metadata={"quadrature_degree": 2})
cell_map = surface.topology.index_map(2)
assert cell_map is not None
surface_cells = np.arange(cell_map.size_local + cell_map.num_ghosts, dtype=np.int32)
H = fem.functionspace(surface, ("DG", 0))
h_surface = fem.Function(H, dtype=PETSc.ScalarType)
h_surface.x.array[H.dofmap.list[:, 0]] = surface.h(2, surface_cells)
h_surface.x.scatter_forward()
alpha_surface = ufl.min_value(1.0, (thickness / ufl.real(h_surface)) ** 2)
# -


def surface_strains(u, theta):
    """Return shell strains and drilling strain on the embedded mesh."""
    eta = ufl.cross(theta, n_surface)
    gradient = ufl.grad(u)
    e = ufl.sym(T.T * gradient * T)
    k = ufl.sym(T.T * (ufl.grad(eta) + gradient.T * ufl.grad(n_surface)) * T)
    gamma = T.T * (eta + gradient.T * n_surface)
    drilling = ufl.dot(theta, n_surface) - ufl.dot(ufl.curl(u), n_surface) / 2
    return e, k, gamma, drilling


def surface_stiffness(u, theta, v, psi):
    """Return the embedded shell PSRI stiffness, including drilling."""
    e, k, gamma, drilling = surface_strains(u, theta)
    ev, kv, gammav, drillingv = surface_strains(v, psi)
    membrane_shear = thickness * ufl.inner(
        stress(e), ev
    ) + shear_correction * mu * thickness * ufl.inner(gamma, gammav)
    bending = thickness**3 / 12 * ufl.inner(stress(k), kv)
    stabilisation = young * thickness**3 / h_surface**2 * ufl.inner(drilling, drillingv)
    return (bending + stabilisation + alpha_surface * membrane_shear) * dx_surface + (
        1 - alpha_surface
    ) * membrane_shear * dx_surface_reduced


# ### Verification, solve and output
#
# Check surface area and a rigid rotation, including drilling. The wavy
# profile has regions of positive and negative Gaussian curvature.
# Clamp the tagged arc and apply the same inward pressure as before.
# Compare the integrated mesh area with the CAD area to within 0.5%.
# The perforated panel has a different domain from the first example.
# Report its work balance, and check it in double precision only.

# +
surface_area = integral(1 * dx_surface).real
assert cad_area > 0
assert abs(surface_area / cad_area - 1) < 0.005
curvature_squared = integral(ufl.inner(ufl.grad(n_surface), ufl.grad(n_surface)) * dx_surface).real
assert np.isfinite(curvature_squared) and curvature_squared > 0
gaussian_curvature = ufl.real(ufl.det(T.T * ufl.grad(n_surface) * T))
positive_curvature = integral(ufl.max_value(gaussian_curvature, 0) * dx_surface).real
negative_curvature = integral(-ufl.min_value(gaussian_curvature, 0) * dx_surface).real
assert positive_curvature > 0 and negative_curvature > 0
rigid_surface = surface_strains(ufl.cross(omega, X), omega)
surface_rigid_norm = integral(sum(ufl.inner(e, e) for e in rigid_surface) * dx_surface).real
assert surface_rigid_norm < 1000 * np.finfo(default_real_type).eps ** 2

u_s, theta = ufl.TrialFunctions(W_surface)
v_s, psi = ufl.TestFunctions(W_surface)
clamped_surface = facet_tags.find(2)
zero_surface_u = fem.Function(V_surface, dtype=PETSc.ScalarType)
zero_surface_rotation = fem.Function(Q_surface, dtype=PETSc.ScalarType)
bcs_surface = [
    fem.dirichletbc(
        zero_surface_u,
        fem.locate_dofs_topological((W_surface.sub(0), V_surface), 1, clamped_surface),
        W_surface.sub(0),
    ),
    fem.dirichletbc(
        zero_surface_rotation,
        fem.locate_dofs_topological((W_surface.sub(1), Q_surface), 1, clamped_surface),
        W_surface.sub(1),
    ),
]
surface_problem = LinearProblem(
    surface_stiffness(u_s, theta, v_s, psi),
    ufl.inner(-pressure * n_surface, v_s) * dx_surface,
    bcs=bcs_surface,
    petsc_options_prefix="demo_shell_surface_",
    petsc_options=options,
)
surface_solution = surface_problem.solve()
assert isinstance(surface_solution, fem.Function)
u_surface, theta_surface = ufl.split(surface_solution)
surface_energy_twice = integral(
    surface_stiffness(u_surface, theta_surface, u_surface, theta_surface)
).real
surface_work = integral(ufl.inner(-pressure * n_surface, u_surface) * dx_surface).real
assert np.isfinite(surface_energy_twice) and surface_energy_twice > 0
surface_balance = abs(surface_energy_twice - surface_work) / surface_energy_twice
if double_precision:
    assert surface_balance < 1e-6
if comm.rank == 0:
    print(f"Gmsh shell elastic energy: {surface_energy_twice / 2:.6e}")
    print(f"Gmsh relative work balance error: {surface_balance:.3e}")
    print(f"Gmsh surface area / CAD area: {surface_area / cad_area:.6f}")
# -

# ### The TDNNS shell on a general curved surface
#
# The TDNNS section above relied on the cylinder's parametrisation, whose
# tangent frame is orthonormal and parallel, so that $A^T\nabla\eta$
# collapsed to $\nabla\beta$ and the director change could be carried as
# an independent unknown. A general curved surface has no such frame, and
# a mesh of it is only $C^0$: the normal jumps across element edges.
# Measuring bending with the element-wise gradient alone then misses the
# curvature carried by those kinks, and the computed response converges to
# the wrong answer.
#
# {cite:t}`neunteufel2024shells` give a TDNNS shell that works on such a
# mesh. Its kinematics differ from the section above: there is no
# independent rotation field. Bending is instead driven by the rotation
# induced by the displacement, $\nabla_\Sigma u^T n$, and
# the tangential-continuous unknown that remains is the transverse shear
# strain $\gamma$, which enters hierarchically on top of Kirchhoff--Love
# kinematics. Both formulations carry the bending moment in
# Hellan--Herrmann--Johnson elements, which is why the method is often
# named after them; dropping $\gamma$ here would leave the HHJ method for
# a Koiter shell.
#
# The second ingredient is a distributional Weingarten tensor, in which
# the element-wise term is accompanied by the dihedral angle between the
# normals of neighbouring elements. Linearising about the reference
# configuration, the element-wise curvature is
#
# $$
# H_n(u)=\nabla_\Sigma\!\left(\nabla_\Sigma u^T n\right)
#        -\nabla_\Sigma u^T\nabla_\Sigma n,
# $$
#
# while the linearised change of dihedral angle on an edge is the sum of
# the element-local values of $(\nabla_\Sigma u^T n)\cdot\mu$, each taken
# with its own outward co-normal $\mu$. Pairing both with the bending
# moment $\sigma$ gives the bending work
#
# $$
# \langle\kappa,\sigma\rangle_h
#   =\int_\Sigma(H_n(u)-\nabla_\Sigma\gamma):\sigma
#   -\sum_E\int_E
#     [\![(\nabla_\Sigma u^T n-\gamma)\cdot\mu]\!]\,\sigma_{\mu\mu}.
# $$
#
# The bracket $[\![\cdot]\!]$ is that sum over the two elements meeting at
# $E$, not a difference: the co-normals of the two sides point opposite
# ways, so summing them is what measures the kink. Because $\gamma$ is
# itself the shear strain, the shear energy is simply
# $\kappa_s\mu t\|\gamma\|^2$.
#
# Discretise $\sigma$ with Hellan--Herrmann--Johnson elements, whose
# co-normal--co-normal component $\sigma_{\mu\mu}$ is continuous, and
# $\gamma$ with Nédélec elements. On this embedded mesh both map into the
# tangent plane automatically, so no tangent basis is needed for them.
#
# Because the only rotational unknown is the tangential $\gamma$, there is
# no drilling degree of freedom and the stabilisation term used above is
# not needed. The essential conditions are $u$ and $\gamma$ on the clamped
# arc; the openings and free edges carry $\sigma_{\mu\mu}=0$ instead.
#
# The formulation removes shear locking but says nothing about membrane
# locking, so the PSRI treatment of the membrane term is kept.

# +
W_surface_tdnns = fem.functionspace(
    surface,
    mixed_element(
        [
            element("HHJ", surface.basix_cell(), 1, dtype=default_real_type),
            element("N2E", surface.basix_cell(), 1, dtype=default_real_type),
            element("Lagrange", surface.basix_cell(), 2, shape=(3,), dtype=default_real_type),
        ]
    ),
)
S_surface_tdnns, _ = W_surface_tdnns.sub(0).collapse()
G_surface_tdnns, _ = W_surface_tdnns.sub(1).collapse()
V_surface_tdnns, _ = W_surface_tdnns.sub(2).collapse()
co_normal = ufl.FacetNormal(surface)
dS_surface = ufl.Measure("dS", domain=surface, metadata={"quadrature_degree": 6})
ds_surface = ufl.Measure("ds", domain=surface, metadata={"quadrature_degree": 6})


def induced_rotation(w):
    """Return the rotation induced by a displacement, grad_S(w)^T n."""
    return ufl.dot(ufl.grad(w).T, n_surface)


def element_curvature(w):
    """Return the element-wise linearised curvature of a displacement."""
    return ufl.grad(induced_rotation(w)) - ufl.grad(w).T * ufl.grad(n_surface)


def bending_work(moment, w, shear, *, moment_is_test):
    """Pair a bending moment with the distributional linearised curvature.

    The element-wise Weingarten term is corrected on every edge by the
    linearised change of dihedral angle, which is the sum of the
    element-local values of ``(grad_S(w)^T n - shear) . mu``. ``ufl.inner``
    conjugates its second argument, so the arguments are ordered by which
    one carries the test function.
    """
    volume = element_curvature(w) - ufl.grad(shear)

    # The normal is derived from the cell Jacobian, so it is discontinuous
    # and has to be restricted per side rather than reusing the helper.
    def angle(side):
        rotation = ufl.dot(ufl.grad(w)(side).T, n_surface(side)) - shear(side)
        return ufl.dot(rotation, co_normal(side))

    def moment_nn(side):
        return ufl.dot(ufl.dot(moment(side), co_normal(side)), co_normal(side))

    edge = ufl.dot(induced_rotation(w) - shear, co_normal)
    boundary_nn = ufl.dot(ufl.dot(moment, co_normal), co_normal)
    if moment_is_test:
        return (
            ufl.inner(volume, moment) * dx_surface
            # sigma_mumu is continuous, so either side will do.
            - ufl.inner(angle("+") + angle("-"), moment_nn("+")) * dS_surface
            - ufl.inner(edge, boundary_nn) * ds_surface
        )
    return (
        ufl.inner(moment, volume) * dx_surface
        - ufl.inner(moment_nn("+"), angle("+") + angle("-")) * dS_surface
        - ufl.inner(boundary_nn, edge) * ds_surface
    )


sigma_s, gamma_s, u_s_tdnns = ufl.TrialFunctions(W_surface_tdnns)
tau_s, delta_s, v_s_tdnns = ufl.TestFunctions(W_surface_tdnns)
membrane_s = ufl.sym(T.T * ufl.grad(u_s_tdnns) * T)
membrane_vs = ufl.sym(T.T * ufl.grad(v_s_tdnns) * T)
membrane_only = thickness * ufl.inner(stress(membrane_s), membrane_vs)
a_surface_tdnns = (
    -ufl.inner(compliance(T.T * sigma_s * T), T.T * tau_s * T) * dx_surface
    + bending_work(tau_s, u_s_tdnns, gamma_s, moment_is_test=True)
    + bending_work(sigma_s, v_s_tdnns, delta_s, moment_is_test=False)
    + shear_correction * mu * thickness * ufl.inner(gamma_s, delta_s) * dx_surface
    + alpha_surface * membrane_only * dx_surface
    + (1 - alpha_surface) * membrane_only * dx_surface_reduced
)
# -

# The moment degrees of freedom on a boundary facet are its
# co-normal--co-normal moments, so constraining them on every edge that is
# not clamped imposes $\sigma_{\mu\mu}=0$ there. The saddle-point system is
# larger and more ill-conditioned than the displacement one, so MUMPS is
# given extra working space when it is the solver in use.

# +
surface.topology.create_connectivity(1, 2)
unclamped = np.setdiff1d(mesh.exterior_facet_indices(surface.topology), clamped_surface)
bcs_surface_tdnns = [
    fem.dirichletbc(
        fem.Function(S_surface_tdnns, dtype=PETSc.ScalarType),
        fem.locate_dofs_topological((W_surface_tdnns.sub(0), S_surface_tdnns), 1, unclamped),
        W_surface_tdnns.sub(0),
    ),
    fem.dirichletbc(
        fem.Function(G_surface_tdnns, dtype=PETSc.ScalarType),
        fem.locate_dofs_topological((W_surface_tdnns.sub(1), G_surface_tdnns), 1, clamped_surface),
        W_surface_tdnns.sub(1),
    ),
    fem.dirichletbc(
        fem.Function(V_surface_tdnns, dtype=PETSc.ScalarType),
        fem.locate_dofs_topological((W_surface_tdnns.sub(2), V_surface_tdnns), 1, clamped_surface),
        W_surface_tdnns.sub(2),
    ),
]
surface_tdnns_options = dict(options)
if surface_tdnns_options.get("pc_factor_mat_solver_type") == "mumps":
    surface_tdnns_options["mat_mumps_icntl_14"] = "400"
surface_tdnns_problem = LinearProblem(
    a_surface_tdnns,
    ufl.inner(-pressure * n_surface, v_s_tdnns) * dx_surface,
    bcs=bcs_surface_tdnns,
    petsc_options_prefix="demo_shell_surface_tdnns_",
    petsc_options=surface_tdnns_options,
)
surface_tdnns_solution = surface_tdnns_problem.solve()
assert isinstance(surface_tdnns_solution, fem.Function)
u_surface_tdnns = ufl.split(surface_tdnns_solution)[2]
work_surface_tdnns = integral(ufl.inner(-pressure * n_surface, u_surface_tdnns) * dx_surface).real
assert abs(work_surface_tdnns - surface_work) / abs(surface_work) < 5e-2
if comm.rank == 0:
    print(f"Gmsh TDNNS external work: {work_surface_tdnns:.6e}")
    print(
        "Gmsh TDNNS relative difference from the displacement formulation: "
        f"{abs(work_surface_tdnns - surface_work) / abs(surface_work):.3e}"
    )
# -

# +
V_surface_out = fem.functionspace(surface, output_element)
surface_displacement = fem.Function(V_surface_out, dtype=PETSc.ScalarType)
surface_displacement.interpolate(surface_solution.sub(0))
write_shell_vtkhdf(
    out_folder / "shell_gmsh.vtkhdf",
    surface_displacement,
    V_surface_out.tabulate_dof_coordinates(),
)
# -

# Open `out_shell/shell_gmsh.vtkhdf` with the same ParaView steps.
# Its cubic visualization mesh samples the quadratic Gmsh geometry.

# Release factorisations in the same order on every rank: their PETSc
# teardown is collective, whereas interpreter shutdown order can vary.

del surface_problem, surface_tdnns_problem, problem, problem_tdnns, verification

# ## PyVista visualization
#
# If PyVista is installed, read the two VTKHDF files on rank zero and
# plot the complete shells, including when the solver runs under MPI.
# Show displacement magnitude on the deformed surface, with deformation
# amplified by at most 1000, capped at 15% of the panel length, and the
# reference mesh in grey. Annotate the actual amplification factor. A
# fixed off-axis camera keeps the curvature of both shells visible.
# Save `out_shell/shell.png` and `out_shell/shell_gmsh.png` as well.
# Set `PYVISTA_OFF_SCREEN=true` to save images without opening windows.

# +
if comm.rank == 0:
    try:
        import pyvista
    except ModuleNotFoundError:
        print("Install 'pyvista' to plot the shells: 'python3 -m pip install pyvista'")
    else:
        for name in ("shell", "shell_gmsh"):
            grid = pyvista.read(out_folder / f"{name}.vtkhdf")
            grid.point_data["displacement_magnitude"] = np.linalg.norm(grid.point_data["u"], axis=1)
            warp_factor = min(
                1000.0, 0.15 * length / grid.point_data["displacement_magnitude"].max()
            )
            warped = grid.warp_by_vector("u", factor=warp_factor)
            plotter = pyvista.Plotter(window_size=(1000, 800))
            plotter.add_text(f"{name}: displacement x {warp_factor:.0f}", font_size=10)
            plotter.add_mesh(grid, style="wireframe", color="grey", opacity=0.1)
            plotter.add_mesh(
                warped,
                scalars="displacement_magnitude",
                show_edges=True,
                edge_color="grey",
                edge_opacity=0.2,
                scalar_bar_args={
                    "title": "Displacement magnitude",
                    "fmt": "%.2e",
                    "n_labels": 4,
                    "position_x": 0.2,
                    "position_y": 0.04,
                    "width": 0.6,
                },
            )
            plotter.camera_position = [
                (2 * R, 2.6 * R, 4.7 * R),
                (0, 0.45 * length, 0.45 * R),
                (0, 1, 0),
            ]
            plotter.reset_camera()
            plotter.camera.zoom(0.85)
            plotter.show(screenshot=out_folder / f"{name}.png")
# -

# ## References
# ```{bibliography}
# :filter: docname in docnames
# ```
