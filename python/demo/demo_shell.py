# Copyright (C) 2026 DOLFINx contributors
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

# # Linear cylindrical shell
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
# space, and membrane, bending and transverse shear energies.
#
# The model includes transverse shear deformation and assumes small
# displacements and rotations. Cubic bubble enrichment and partial
# selective reduced integration (PSRI) alleviate membrane and shear
# locking. This demo uses the practical FEniCS-Shells PSRI variant;
# it does not establish thickness-uniform convergence for this geometry.
# Running on multiple MPI ranks requires PETSc with MUMPS or SuperLU_DIST.
# For background, see the
# [FEniCS-Shells Naghdi formulation](https://fenics-shells.readthedocs.io/en/latest/demo/nonlinear-naghdi-cylindrical/demo_nonlinear-naghdi-cylindrical.py.html).
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

import numpy as np

import ufl
from basix.ufl import blocked_element, element, enriched_element, mixed_element
from dolfinx import default_real_type, fem, io, mesh
from dolfinx.fem.petsc import LinearProblem

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
# This follows the
# [FEniCS-Shells PSRI implementation](https://fenics-shells.readthedocs.io/en/latest/demo/nonlinear-naghdi-cylindrical/demo_nonlinear-naghdi-cylindrical.py.html).


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

# +
clamped = mesh.locate_entities_boundary(msh, 1, lambda x: np.isclose(x[1], 0.0))
zero_u = fem.Function(V, dtype=PETSc.ScalarType)
bcs = [
    fem.dirichletbc(zero_u, fem.locate_dofs_topological((W.sub(0), V), 1, clamped), W.sub(0)),
    fem.dirichletbc(zero_beta, fem.locate_dofs_topological((W.sub(1), Q), 1, clamped), W.sub(1)),
]
load = ufl.inner(-pressure * normal, v) * dx
problem = LinearProblem(a, load, bcs=bcs, petsc_options_prefix="demo_shell_", petsc_options=options)
solution = problem.solve()
assert isinstance(solution, fem.Function)
uh, betah = ufl.split(solution)
energy_twice = integral(stiffness(uh, betah, uh, betah)).real
work = integral(ufl.inner(-pressure * normal, uh) * dx).real
assert np.isfinite(energy_twice) and energy_twice > 0
assert abs(energy_twice - work) / energy_twice < tolerance
if comm.rank == 0:
    print(f"Elastic energy: {energy_twice / 2:.6e}")
    print(f"Relative work balance error: {abs(energy_twice - work) / energy_twice:.3e}")
# -

# ## Output
#
# Write the Cartesian displacement and an offset from the parameter
# rectangle to the reference cylinder. In ParaView, first apply
# **Warp By Vector** to `reference_offset` with scale factor 1, then
# apply a second **Warp By Vector** to `displacement`. Increase the
# second scale factor to make the small displacement visible. The first
# warp displays the analytic geometry sampled at the output nodes.
# Interpolate the enriched cubic displacement into discontinuous cubic
# Lagrange elements for VTK output. This preserves its cellwise polynomial
# rather than discarding the bubble contribution in a quadratic space.

# +
V_out = fem.functionspace(msh, ("Discontinuous Lagrange", 3, (3,)))
displacement = fem.Function(V_out, dtype=PETSc.ScalarType)
displacement.interpolate(solution.sub(0))
displacement.name = "displacement"
reference_offset = fem.Function(V_out, dtype=PETSc.ScalarType)
reference_offset.name = "reference_offset"
reference_offset.interpolate(
    fem.Expression(position - ufl.as_vector((x[0], x[1], 0)), V_out.element.interpolation_points)
)
out_folder = Path("out_shell")
out_folder.mkdir(exist_ok=True)
with io.VTKFile(comm, out_folder / "shell.pvd", "w") as output:
    output.write_function([displacement, reference_offset])
# -
