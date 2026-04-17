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
#
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
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import ufl

from dolfinx import fem, mesh
from dolfinx.fem.petsc import NonlinearProblem

# subdivisions for coordinate directions
nx = 80
ny = 80
domain = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny)
# Finite Element space
V=fem.functionspace(domain, ("Lagrange",1))

# true image
def true_image(x):
    X=x[0]
    Y=x[1]
    # circle $(x-a)^2+(y-b)^2=r^2 $
    disk = ((X-0.35)**2+(Y-0.65)**2 <0.12**2).astype(np.float64)
    # Gaussian bump, $G(x,y)=Aexp(-{(x-x_0)^2+(y-y_0)^2\over 2\sigma^2})$
    bump = np.exp(-40.0*((X-0.75)**2+(Y-0.25)**2))
    return 0.8*disk+0.3*bump

u_true = fem.Function(V)
u_true.interpolate(true_image)

# mask
def mask_function(x):
    X=x[0]
    Y=x[1]
    # irregular hole by union of the two circles, d<r_1+r_2
    hole1=(X-0.5)**2+(Y-0.52)**2<0.12**2
    hole2=(X-0.65)**2+(Y-0.58)**2<0.08**2
    mask = np.ones_like(X,dtype=np.float64)
    mask[hole1]=0.0
    mask[hole2]=0.0
    return mask
m=fem.Function(V)
m.interpolate(mask_function)

# observed image $f=mu_{true}$
def observed_image(x):
    return mask_function(x)*true_image(x)

f=fem.Function(V)
f.interpolate(observed_image)
#unknown 
u=fem.Function(V)
#inital guess
init_fill=0.5
def initial_guess(x):
    return observed_image(x)+(1.0-mask_function(x))*init_fill

u.interpolate(initial_guess)
# weak form 
# $F(u,v)=\int_{\Omega}m(u-f)v\mathrm{d}x+\alpha \int_{\Omega}{\nabla u \cdot \nabla v \over \sqrt{|\nabla u|^2 +\varepsilon^2}}\mathrm{d}x $
v=ufl.TestFunction(V)
du=ufl.TrialFunction(V)
ALPHA = 0.02
EPS = 1.0e-4
alpha=fem.Constant(domain, ALPHA)
eps=fem.Constant(domain, EPS)
grad_u=ufl.grad(u)
tv_denom=ufl.sqrt(ufl.inner(grad_u,grad_u)+eps**2)
F=(
    m*(u-f)*v*ufl.dx + alpha*ufl.inner(grad_u,ufl.grad(v))/tv_denom*ufl.dx
)

# Jacobian
J=ufl.derivative(F,u,du)

# Solve
petsc_options={
    "snes_type":"newtonls",
    "snes_linesearch_type": "bt",
    "snes_rtol": 1.0e-8,
    "snes_atol": 1.0e-8,
    "ksp_type":"preonly",
    "pc_type": "lu"
}
problem= NonlinearProblem(
    F,u,bcs=[],J=J,

    petsc_options_prefix="tv_inpaint_",
    petsc_options=petsc_options
)
problem.solve()