# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.6
# ---

# # Smoothed TV image inpainting
#
# ```{admonition} Download sources
# :class: download
# * {download}`Python script <./demo_smoothed_tv_inpainting.py>`
# * {download}`Jupyter notebook <./demo_smoothed_tv_inpainting.ipynb>`
# ```
# Rough draft notes
# This demo solves a variational image inpainting problem on the unit square.
# A synthetic image is masked on an irregular interior region, and the missing
# values are reconstructed using smoothed total variation regularization.
#
# ## Equation and problem definition
#
# Let $\Omega = [0,1]^2$ be the image domain. We define:
#
# - $u_{\mathrm{true}}$: synthetic ground-truth image
# - $m$: mask, equal to 1 on known data and 0 on the missing region
# - $f = m u_{\mathrm{true}}$: observed incomplete image
# - $u$: reconstructed image
#
# We compute $u$ by minimising
#
# $$
# J(u)= {1 \over 2}\int_\Omega m(u-f)^2\,\mathrm{d}x
# + \alpha \int_\Omega \sqrt{|\nabla u|^2 + \varepsilon^2}\,\mathrm{d}x.
# $$
#
# The first term enforces agreement with the known image data, while the
# second term is a smoothed total variation regularisation term.
#
# ## Weak formulation
#
# The weak form reads: find $u$ such that
#
# $$
# \int_\Omega m(u-f)v\,\mathrm{d}x
# + \alpha \int_\Omega
# {\nabla u\cdot\nabla v \over \sqrt{|\nabla u|^2+\varepsilon^2}}
# \,\mathrm{d}x = 0
# $$
#
# for all test functions $v$.
#
# ## Implementation
#
# We use a first-order Lagrange space on a triangular mesh of the unit square.
# The nonlinear problem is solved with PETSc SNES through
# {py:class}`NonlinearProblem <dolfinx.fem.petsc.NonlinearProblem>`

# +
from mpi4py import MPI

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import ufl
from dolfinx import fem, mesh
from dolfinx.fem.petsc import NonlinearProblem

# -


# We begin by creating a mesh of the unit square and a first-order
# Lagrange function space on the mesh.

nx = 100
ny = 100
msh = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny)
V = fem.functionspace(msh, ("Lagrange", 1))

# Next, we define a synthetic ground-truth image and a mask describing
# the missing region.

def true_image(x):
    X = x[0]
    Y = x[1]
    return ((X > 0.2) & (X < 0.8) & (Y > 0.2) & (Y < 0.8)).astype(np.float64)



# Mask: 1 outside circle, 0 inside

def mask_function(x):
    X = x[0]
    Y = x[1]
    mask = np.ones_like(X, dtype=np.float64)
    # random seed for reproducibility
    np.random.seed(0)
    # number of speckles
    num_speckles = 150
    # random centers
    cx = np.random.uniform(0.2, 0.8, num_speckles)
    cy = np.random.uniform(0.2, 0.8, num_speckles)
    # random radii (small + varied)
    radii = np.random.uniform(0.005, 0.02, num_speckles)
    # create holes
    for i in range(num_speckles):
        r2 = (X - cx[i])**2 + (Y - cy[i])**2
        mask[r2 < radii[i]**2] = 0.0
    return mask

# We interpolate the exact image and the mask into the finite element
# space, and construct the observed damaged image.

# True image
u_true = fem.Function(V)
u_true.interpolate(true_image)

# Mask over the true image
m = fem.Function(V)
m.interpolate(mask_function)

# The observed image which is the 'damaged image'
f = fem.Function(V)
f.x.array[:] = m.x.array * u_true.x.array

# The unknown reconstruction is initialised with the observed image.
u = fem.Function(V)
u.x.array[:] = f.x.array.copy()

# We now define the nonlinear variational problem corresponding to the
# smoothed total variation regularised inpainting model.

alpha = fem.Constant(msh, 0.01)
beta = fem.Constant(msh, 1.0)
eps = fem.Constant(msh, 10.0e-4)

# TV inpainting weak form
v = ufl.TestFunction(V)
du = ufl.TrialFunction(V)

grad_u = ufl.grad(u)
tv_denom = ufl.sqrt(ufl.inner(grad_u, grad_u) + eps**2)

F = (
    beta * m * (u - f) * v * ufl.dx
    + alpha * ufl.inner(grad_u, ufl.grad(v)) / tv_denom * ufl.dx
)

# Jacobian
J = ufl.derivative(F, u, du)

# A nonlinear PETSc problem is created and solved with a Newton line
# search method.

petsc_options = {
    "snes_type": "newtonls",
    "snes_linesearch_type": "bt",
    "snes_rtol": 1.0e-8,
    "snes_atol": 1.0e-8,
    "ksp_type": "preonly",
    "pc_type": "lu",
}

problem = NonlinearProblem(
    F,
    u,
    bcs=[],
    J=J,
    petsc_options_prefix="tv_inpainting_",
    petsc_options=petsc_options,
)

problem.solve()

# We print a few basic diagnostics.

# Data fidelity (known region only)
data_error = fem.assemble_scalar(
    fem.form(m * (u - f)**2 * ufl.dx)
)
data_error = np.sqrt(msh.comm.allreduce(data_error, op=MPI.SUM))

# TV seminorm
tv_energy = fem.assemble_scalar(
    fem.form(ufl.sqrt(ufl.inner(ufl.grad(u), ufl.grad(u)) + eps**2) * ufl.dx)
)
tv_energy = msh.comm.allreduce(tv_energy, op=MPI.SUM)

# True error 
true_error = fem.assemble_scalar(
    fem.form((u - u_true)**2 * ufl.dx)
)
true_error = np.sqrt(msh.comm.allreduce(true_error, op=MPI.SUM))

# Hole error 
hole_error = fem.assemble_scalar(
    fem.form((1 - m) * (u - u_true)**2 * ufl.dx)
)
hole_error = np.sqrt(msh.comm.allreduce(hole_error, op=MPI.SUM))

J = 0.5 * data_error**2 + tv_energy

if msh.comm.rank == 0:
    print(f"Data error (known region): {data_error:.4e}")
    print(f"TV seminorm: {tv_energy:.4e}")
    print(f"True L2 error: {true_error:.4e}")
    print(f"Hole error: {hole_error:.4e}")
    print(f"J(u): {J:.4e}")

# For visualisation, we compute the difference between the reconstructed
# and observed images and build a triangulation from the mesh.

u_minus_f = fem.Function(V)
u_minus_f.x.array[:] = u.x.array - f.x.array

# For visualisation, we compute the difference between the reconstructed
# and observed images and build a triangulation from the mesh.

coords = V.tabulate_dof_coordinates()
x, y = coords[:, 0], coords[:, 1]

msh.topology.create_connectivity(msh.topology.dim, 0)
cells = msh.topology.connectivity(msh.topology.dim, 0)
triangles = np.array(cells.array, dtype=np.int32).reshape(-1, 3)
triang = mtri.Triangulation(x, y, triangles)

def plot_field(ax, data, title, fig, cmap="viridis", vmin=0.0, vmax=1.0):
    im = ax.tripcolor(triang, data, shading="gouraud", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_aspect("equal")
    fig.colorbar(im, ax=ax)

fig, axes = plt.subplots(2, 3, figsize=(12, 8))

plot_field(axes[0, 0], u_true.x.array, "u_true", fig)
plot_field(axes[0, 1], m.x.array, "mask", fig, cmap="gray")
plot_field(axes[0, 2], f.x.array, "f", fig)
plot_field(axes[1, 0], u.x.array, "u", fig)

lim = np.max(np.abs(u_minus_f.x.array))
plot_field(axes[1, 1], u_minus_f.x.array, "u_minus_f", fig, cmap="coolwarm", vmin=-lim, vmax=lim)

axes[1, 2].axis("off")

plt.tight_layout()
plt.show()