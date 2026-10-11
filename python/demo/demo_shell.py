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
# space, and membrane, bending and transverse shear energies. A second
# example starts from a Gmsh surface mesh embedded in three dimensions.
# This demo requires Gmsh.
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
    coordinate_element = element(
        "Lagrange",
        source.basix_cell(),
        3,
        lagrange_variant=basix.LagrangeVariant.equispaced,
        shape=(3,),
        dtype=default_real_type,
    )
    output_mesh = mesh.create_mesh(
        source.comm,
        cells,
        coordinate_element,
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
reference_position.interpolate(fem.Expression(position, V_out.element.interpolation_points))
out_folder = Path("out_shell")
out_folder.mkdir(exist_ok=True)
write_shell_vtkhdf(
    out_folder / "shell.vtkhdf", displacement, reference_position.x.array.reshape(-1, 3)
)
# -

# ## Starting from a Gmsh surface mesh
#
# Now mesh the physical cylinder directly, rather than its parameter
# rectangle. Gmsh extrudes a circular arc along the axis and creates
# quadratic triangles. The geometry is a polynomial approximation of the
# CAD surface; its nodes lie on the cylinder, but points between nodes
# need not. The mesh has topological dimension two and geometric
# dimension three. The surface and clamped arc are physical groups.

# +
gmsh.initialize()
if comm.rank == 0:
    gmsh.model.add("cylindrical_panel")
    geo = gmsh.model.geo
    center = geo.addPoint(0, 0, 0)
    start = geo.addPoint(-R / np.sqrt(2), 0, R / np.sqrt(2))
    end = geo.addPoint(R / np.sqrt(2), 0, R / np.sqrt(2))
    arc = geo.addCircleArc(start, center, end)
    geo.mesh.setTransfiniteCurve(arc, 13)
    extruded = geo.extrude([(1, arc)], 0, length, 0, numElements=[20])
    geo.synchronize()
    surfaces = [tag for dim, tag in extruded if dim == 2]
    gmsh.model.addPhysicalGroup(2, surfaces, tag=1)
    gmsh.model.addPhysicalGroup(1, [arc], tag=2)
    gmsh.model.setPhysicalName(2, 1, "shell")
    gmsh.model.setPhysicalName(1, 2, "clamped")
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(2)
mesh_data = gmshio.model_to_mesh(gmsh.model, comm, 0, gdim=3)
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
# cylinder axis. Unlike a normal on planar triangles, this normal varies
# within a quadratic cell. Differentiating it retains the curvature.
# We form an orthonormal tangent basis $T=[t_1,t_2]$ using the axial
# direction; this construction is suitable for this cylindrical panel.
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
# [embedded shell formulation](https://bleyerj.github.io/comet-fenicsx/tours/shells/linear_shell/linear_shell.html).

# +
X = ufl.SpatialCoordinate(surface)
J = ufl.Jacobian(surface)
j0 = ufl.as_vector((J[0, 0], J[1, 0], J[2, 0]))
j1 = ufl.as_vector((J[0, 1], J[1, 1], J[2, 1]))
cross = ufl.cross(j0, j1)
n_cell = cross / ufl.sqrt(ufl.dot(cross, cross))
radial = ufl.as_vector((X[0], 0, X[2]))
n_surface = ufl.conditional(ufl.gt(ufl.dot(n_cell, radial), 0), n_cell, -n_cell)
t1 = ufl.cross(axis, n_surface)
t1 = t1 / ufl.sqrt(ufl.dot(t1, t1))
t2 = ufl.cross(n_surface, t1)
T = ufl.as_matrix(((t1[0], t2[0]), (t1[1], t2[1]), (t1[2], t2[2])))

rotation_element = element("Lagrange", surface.basix_cell(), 2, shape=(3,), dtype=default_real_type)
W_surface = fem.functionspace(surface, mixed_element([U, rotation_element]))
V_surface, _ = W_surface.sub(0).collapse()
Q_surface, _ = W_surface.sub(1).collapse()
dx_surface = ufl.Measure("dx", domain=surface, metadata={"quadrature_degree": 6})
dx_surface_reduced = ufl.Measure("dx", domain=surface, metadata={"quadrature_degree": 2})
cell_map = surface.topology.index_map(2)
assert cell_map is not None
surface_cells = np.arange(cell_map.size_local + cell_map.num_ghosts, dtype=np.int32)
H = fem.functionspace(surface, ("DG", 0))
h_surface = fem.Function(H, dtype=default_real_type)
h_surface.x.array[H.dofmap.list[:, 0]] = surface.h(2, surface_cells)
h_surface.x.scatter_forward()
alpha_surface = ufl.min_value(1.0, (thickness / h_surface) ** 2)
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
# Check surface area, curvature and a rigid rotation, including drilling.
# Clamp the tagged arc and apply the same inward pressure as before.
# The two solutions use different rotation spaces and approximate
# geometries; compare their integrated pressure work as a consistency
# check at this resolution, rather than expecting identical solutions.

# +
surface_area = integral(1 * dx_surface).real
assert abs(surface_area / (np.pi * R * length / 2) - 1) < 1e-4
curvature_squared = integral(ufl.inner(ufl.grad(n_surface), ufl.grad(n_surface)) * dx_surface).real
assert abs(curvature_squared * R**2 / surface_area - 1) < 0.01
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
assert abs(surface_energy_twice - surface_work) / surface_energy_twice < tolerance
assert abs(surface_work / work - 1) < 0.05
if comm.rank == 0:
    print(f"Gmsh shell elastic energy: {surface_energy_twice / 2:.6e}")
    print(f"Gmsh/parametric pressure work ratio: {surface_work / work:.6f}")

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

del surface_problem, problem, verification
