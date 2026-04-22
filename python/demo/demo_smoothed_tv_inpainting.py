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
# + \alpha \int_\Omega \sqrt{||\nabla u||^2 + \varepsilon^2}\,\mathrm{d}x.
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
# {\nabla u\cdot\nabla v \over \sqrt{||\nabla u||^2+\varepsilon^2}}
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

nx = 300
ny = 300
msh = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny)
V = fem.functionspace(msh, ("Lagrange", 1))

# Next, we define a synthetic ground-truth image and a mask describing
# the missing region.

def true_image(x):
    X = x[0]
    Y = x[1]
    return ((X > 0.2) & (X < 0.8) & (Y > 0.2) & (Y < 0.8)).astype(np.float64)



# Mask

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
# Our image domain $\Omega =(0,1)^2 \subset \mathbb{R}^$
# Where $u_{true}: \Omega \to \{0,1 \}$ is our true image
# Where $m: \Omega \to \mathbb{R}$ is the mask
# Where $f: \Omega \to \mathbb{R}$ is the observed damaged image
# Where $u:\Omega \to \mathbb{R}$ is the reconstructed image

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
# TV: $||\nabla u||$$
# Smoothed TV regularization:
# $$TV = \sqrt{||\nabla u||^2 +\varepsilon^2}$$
# $\varepsilon$ allows for differentiation of $||\nabla||$
# $\alpha$ is the regularization strength or the TV weight
# - with large $\alpha$ being strong smoothing
# - with small $\alpha$ being weak smoothing
# $\beta$ is the data fidelity weight
# - large $\beta$ sticks closely to the data
# - small $\beta$ allows deviations and more smoothing
# $\varepsilon$ is the smoothing of the TV
# - large $\varepsilon$ smoother more like quadratic diffusion
# - small $\varepsilon$ closer to true TV edge pereserving

alpha = fem.Constant(msh, 0.003)
beta = fem.Constant(msh, 1.0)
eps = fem.Constant(msh, 1.0e-4)

# Smoothed TV inpainting weak form
# where $TV=||\nabla u||_2 +\varepsilon^2$:
# F(u) = \int m (u-f)v dx + \alpha \int {\nabla u \cdot \nabla v \over \sqrt{TV}}
v = ufl.TestFunction(V)
du = ufl.TrialFunction(V)

grad_u = ufl.grad(u)
tv_denom = ufl.sqrt(ufl.inner(grad_u, grad_u) + eps**2)

F = (
    beta * m * (u - f) * v * ufl.dx
    + alpha * ufl.inner(grad_u, ufl.grad(v)) / tv_denom * ufl.dx
)



# A nonlinear PETSc problem is created and solved with a Newton line
# search method.
# We want to find $u$ such that $F(u)=0$ where:
# $$~u_h =\sum_j u_j\phi_j $$
# and:
# # $$F_i(u)= \int_{\Omega}\beta m(u_h-f)\phi_i+$$
# $$\alpha \int_{\Omega}{\nabla u_j \cdot \nabla \phi_i \over \sqrt{||\nabla||^2+\varepsilon^2}} $$
# We then want a step and direction $s_k$, with iteration $k$, such that $u_{k+1}=u_k+s_k $
# If we linearize $F$ around $u_k$ we get by first order Taylor expansion:
# $$ F(u_k+s)\approx F(u_k) +{dF\over du}(u_k)s $$
# $$ {dF \over du}= J(u_k) , ~F(u_k+s)\approx F(u_k) +J(u_k)s$$
# Setting $F(u_k +s_k)\approx 0$. Then we have:
# $$  F(u_k) +J(u_k)s =0 \to J(u_k)s = -F(u_k)$$
# $ J(u_k)s = -F(u_k)$ is solved with LU factorization:
# $$ LU=J, ~LUs=-F(u_k) $$
# Then with forward and backward substitution we can solve for $s$:
# $$ ~Ly=-F(u_k), ~y=Us$$
# Then to get the step size, we use backtracking line search to find step size $\lambda$:
# $$ \phi(\lambda) = {1\over 2}||F(u_k+\lambda)s_k||^2_2$$
# such that $\phi(\lambda)\leq \phi(0)+c\lambda \phi'(0)
# Then we update $u$ with:
# $$ u_{k+1} = u_k +\lambda s_k, ~ 0<\lambda_k\leq 1$$

# Jacobian 
# $$J(u) = {\partial F(u) \over \partial u}$$
# where $TV = \sqrt{||\nabla||^2+\varepsilon^2}$
# $$ J(u) = \int \beta um\delta uv dx$$
# $$+ \alpha \int[{\nabla \delta u \nablda v \over \sqrt{TV}}$$
# $$ - {(\nabla u \cdot \nabla v) (\nabla u \cdot \nabla \delta u) \over (TV)^{3\over2}} ]dx$$
J = ufl.derivative(F, u, du)

# Newton line search with backtracking for step size $\lambda$
# LU factorization solve for step and direction $J(u_k) s= -F(u_k)$

petsc_options = {
    "snes_type": "newtonls",
    "snes_linesearch_type": "bt",
    "snes_rtol": 1.0e-8,
    "snes_atol": 1.0e-8,
    "snes_max_it": 1000,
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

snes = problem.solver
reason = snes.getConvergedReason()
iters = snes.getIterationNumber()
final_residual = snes.getFunctionNorm()

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


objective_value = 0.5 *float(beta)* data_error**2 + float(alpha) * tv_energy

square_idx = np.where(u_true.x.array > 0.5)[0]

mse = np.mean((u.x.array - u_true.x.array)**2)
psnr = np.inf if mse == 0 else 10.0 * np.log10(1.0 / mse)
u0 = fem.Function(V)
u0.x.array[:] = f.x.array.copy()

J0_data = fem.assemble_scalar(fem.form(m * (u0 - f)**2 * ufl.dx))
J0_data = msh.comm.allreduce(J0_data, op=MPI.SUM)

J0_tv = fem.assemble_scalar(
    fem.form(ufl.sqrt(ufl.inner(ufl.grad(u0), ufl.grad(u0)) + eps**2) * ufl.dx)
)
J0_tv = msh.comm.allreduce(J0_tv, op=MPI.SUM)

J0 = 0.5 * float(beta) * J0_data + float(alpha) * J0_tv

if msh.comm.rank == 0:
    print(f"Initial objective J(f): {J0:.4e}")
    print(f"Final objective J(u): {objective_value:.4e}")
    print(f"Relative decrease: {(J0 - objective_value)/J0:.2%}")
    print(f"SNES iteration: {iters}")
    print(f"SNES final residual norm: {final_residual:.4e}")
    print(f"SNES converged reason: {reason}")
    print(f"Data error (known region): {data_error:.4e}")
    print(f"TV seminorm: {tv_energy:.4e}")
    print(f"True L2 error: {true_error:.4e}")
    print(f"Hole error: {hole_error:.4e}")
    print(f"J(u): {objective_value:.4e}")
    print(f"PSNR: {psnr:.2f} dB")
    print("u min:", np.min(u.x.array))
    print("u max:", np.max(u.x.array))
    print("u mean in square:", np.mean(u.x.array[square_idx]))
    print("u min in square:", np.min(u.x.array[square_idx]))
    print("u max in square:", np.max(u.x.array[square_idx]))

# For visualisation, we compute the difference between the reconstructed
# and observed images and build a triangulation from the mesh.

u_minus_u_true = fem.Function(V)
u_minus_u_true.x.array[:] = u.x.array - u_true.x.array

hole_error_field = fem.Function(V)
hole_error_field.x.array[:] = (1.0 - m.x.array)*(u.x.array - u_true.x.array)

# For visualisation, we compute the difference between the reconstructed
# and observed images and build a triangulation from the mesh.

coords = V.tabulate_dof_coordinates()
x, y = coords[:, 0], coords[:, 1]

msh.topology.create_connectivity(msh.topology.dim, 0)
cells = msh.topology.connectivity(msh.topology.dim, 0)
triangles = np.array(cells.array, dtype=np.int32).reshape(-1, 3)
triang = mtri.Triangulation(x, y, triangles)

def plot_field(ax, data, title, fig, cmap="viridis", vmin=0.0, vmax=1.0):
    im = ax.tripcolor(triang, data, shading="flat", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_aspect("equal")
    fig.colorbar(im, ax=ax)

fig, axes = plt.subplots(2, 3, figsize=(12, 8))

plot_field(axes[0, 0], u_true.x.array, "u_true", fig)
plot_field(axes[0, 1], m.x.array, "mask", fig, cmap="gray")
plot_field(axes[0, 2], f.x.array, "f", fig)
plot_field(axes[1, 0], u.x.array, "u", fig)

lim = np.max(np.abs(u_minus_u_true.x.array))
plot_field(axes[1, 1], u_minus_u_true.x.array, "u - u_true", fig, cmap="coolwarm", vmin=-lim, vmax=lim)

lim = np.max(np.abs(hole_error_field.x.array))

plot_field(axes[1, 2],hole_error_field.x.array,"hole-only error",fig,cmap="coolwarm",vmin=-lim,vmax=lim)

plt.tight_layout()
plt.show()
