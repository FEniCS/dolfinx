# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Biharmonic equation
#
# Authors: Julius Herb, Ottar Hellan and Jørgen S. Dokken
#
# ```{admonition} Download sources
# :class: download
# * {download}`Python script <./demo_biharmonic.py>`
# * {download}`Jupyter notebook <./demo_biharmonic.ipynb>`
# ```
# This demo illustrates how to:
#
# - Solve a linear partial differential equation
# - Use a discontinuous Galerkin method
# - Solve a fourth-order differential equation
# - Use the method of manufactured solutions
#
# ## Equation and problem definition
#
# ### Strong formulation
#
# The biharmonic equation is a fourth-order elliptic equation.
# On the domain $\Omega \subset \mathbb{R}^{d}$, $1 \le d \le 3$, it reads
#
# $$
# \nabla^{4} u = f \quad {\rm in} \ \Omega,
# $$
#
# where $\nabla^{4} \equiv \nabla^{2} \nabla^{2}=\Delta\Delta$ is the
# biharmonic operator and $f$ is a prescribed source term.
# To formulate a complete boundary value problem, the biharmonic equation
# must be complemented by suitable boundary conditions.
#
#  ### Choice of boundary conditions
# As we have a fourth order partial differential equation, we are required
# to supply two boundary conditions.
# There are two common sets of conditions that people use for the
# biharmonic equation, namely the *clamped* condition and the
# *simply supported* condition, see for instance {cite:t}`Gander2017bcs`.
#
# $$
# \begin{align}
# u &= \frac{\partial u}{\partial n} = 0 \text{ on } \partial\Omega\\
# u &= \Delta u = 0 \text{ on } \partial \Omega
# \end{align}
# $$
#
# In this demo we will consider the clamped boundary conditions
#
# $$
# \begin{align}
# u &= g_D \text{ on } \partial\Omega\\
# \frac{\partial u}{\partial n} &= g_N \text{ on } \partial\Omega\\
# \end{align}
# $$
#
# as the simply supported boundary conditions reduces the system into two
# sequential Poisson problems, named the Ciarlet-Raviart method
# {cite}`ciarlet1974mixed`.


# ### Weak formulation
#
# Multiplying the biharmonic equation by a test function and integrating
# by parts twice leads to a problem of second-order derivatives, which
# would require $H^{2}$ conforming (roughly $C^{1}$ continuous) basis
# functions. To solve the biharmonic equation using Lagrange finite element
# basis functions, the biharmonic equation can be split into two second-
# order equations (see the [Mixed Poisson demo](./demo_mixed-poisson)
# for an example of a mixed method), or a variational
# formulation can be constructed that imposes weak continuity of normal
# derivatives between finite element cells. This demo uses a discontinuous
# Galerkin approach to impose continuity of the normal derivative weakly,
# see for instance {cite:t}`babuska1973penalty` or
# {cite:t}`Georgoulis2009biharmonic`.
#
# In this demo, we consider equation 3.20-3.21 from
# {cite:t}`Georgoulis2009biharmonic`, but instead of use a broken
# (discontinuous) finite element space for the unknown $u$, we use a
# continuous space, which simplifies the formulation to
#
#
# a weak formulation of the biharmonic problem reads: find $u \in V_{g_D}$
# such that
#
# $$
# a(u,v)=L(v) \quad \forall \ v \in V,
# $$
#
# where the bilinear form is
#
# $$
# \begin{align}
# a(u, v) &=
# \sum_{K \in \mathcal{T}} \int_{K} \Delta u \Delta v ~{\rm d}x \\
# &+\sum_{E \in \mathcal{E}_h^{\rm int}}\int_{E}\left(
# - \left<\Delta u \right>[\!\![ \nabla v ]\!\!]
# - [\!\![ \nabla u ]\!\!] \left<\nabla^{2} v \right>
# + \frac{\beta}{h_E} [\!\![ \nabla u ]\!\!] [\!\![ \nabla v ]\!\!]
# \right)~{\rm d}s\\
# &+\sum_{E \in \mathcal{E}_h^{\rm ext}}\int_{E}\left(
# - \Delta u  \nabla v \cdot n - \Delta v \nabla u \cdot n
# + \frac{\beta}{h_E} \nabla u \cdot n \nabla v \cdot n
# \right)~{\rm d}s
# \end{align}
# $$
#
# and the linear form is
#
# $$
# L(v) = \int_{\Omega} fv ~{\rm d}x
# +\sum_{E \in \mathcal{E}_h^{\rm ext}}\int_{E}\left(
# -g_N \Delta v + \frac{\beta}{h_E} g_N \nabla v \cdot n
# \right) ~{\rm d}s
# $$
#
# Furthermore, $\left< u \right> = \frac{1}{2} (u_{+} + u_{-})$,
# $[\!\![ w ]\!\!]  = w_{+} \cdot n_{+} + w_{-} \cdot n_{-}$,
# $\beta \ge 0$ is a penalty parameter and
# $h_E$ is a measure of the cell size.
#
# where $K$ is an element of the mesh, while $\mathcal{E}_h^{\rm int}$
# is the collection of all interior facets, while
# $\mathcal{E}_h^{\rm ext}$ is the set of exterior facets.
#
# Note that the Dirichlet condition for $u$ will be enforced strongly
# in this example.
#
# We follow the example of {cite:t}`Georgoulis2009biharmonic` and use the
# method of manufactured solutions to construct a $f$, $g_D$, $g_N$ that
# satisfies
#
# $$
# u(x, y) = \sin (2\pi x) \sin (2 \pi y) \text{in } [0, 1] \times [0, 1].
# $$
#
# ## Implementation
#
# We first import the modules and functions that the program uses:


# +
from pathlib import Path

from mpi4py import MPI

import numpy as np

import ufl
from dolfinx import default_scalar_type, fem, io, mesh, plot
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import CellType, GhostMode

# -

# We begin by using {py:func}`create_rectangle
# <dolfinx.mesh.create_rectangle>` to create a rectangular
# {py:class}`Mesh <dolfinx.mesh.Mesh>` of the domain, and creating a
# finite element {py:class}`FunctionSpace <dolfinx.fem.FunctionSpace>`
# $V$ on the mesh.

N = 32
errors = []
hs = []
msh = mesh.create_rectangle(
    comm=MPI.COMM_WORLD,
    points=((0.0, 0.0), (1.0, 1.0)),
    n=(N, N),
    cell_type=CellType.triangle,
    ghost_mode=GhostMode.shared_facet,
)

# As noted in {cite:t}`Georgoulis2009biharmonic` second order Lagrange
# elements yield sub-optimal convergence, and therefore we choose to use
# third order elements

degree = 3
V = fem.functionspace(msh, ("Lagrange", degree))

# The second argument to {py:func}`functionspace
# <dolfinx.fem.functionspace>` is a tuple consisting of `(family,
# degree)`, where `family` is the finite element family, and `degree`
# specifies the polynomial degree. in this case `V` consists of
# second-order, continuous Lagrange finite element functions.
# For further details of how one can specify
# finite elements as tuples, see {py:class}`ElementMetaData
# <dolfinx.fem.ElementMetaData>`.
#
# Next, we locate the mesh facets that lie on the boundary
# $\Gamma_D = \partial\Omega$.
# We do this using using {py:func}`exterior_facet_indices
# <dolfinx.mesh.exterior_facet_indices>` which returns all mesh boundary
# facets (Note: if we are only interested in a subset of those, consider
# {py:func}`locate_entities_boundary
# <dolfinx.mesh.locate_entities_boundary>`).

tdim = msh.topology.dim
msh.topology.create_connectivity(tdim - 1, tdim)
facets = mesh.exterior_facet_indices(msh.topology)

# We now find the degrees-of-freedom that are associated with the
# boundary facets using {py:func}`locate_dofs_topological
# <dolfinx.fem.locate_dofs_topological>`

fdim = tdim - 1
dofs = fem.locate_dofs_topological(V=V, entity_dim=fdim, entities=facets)

# and use {py:func}`dirichletbc <dolfinx.fem.dirichletbc>` to create a
# {py:class}`DirichletBC <dolfinx.fem.DirichletBC>`
# class that represents the boundary condition.

# We define the manufactured solution and interpolate it into the function
# space of our unknown to apply it as a strong boundary condition


def u_manufactured(x):
    """Manufactured solution."""
    return np.sin(2 * np.pi * x[0]) ** 2 * np.sin(2 * np.pi * x[1]) ** 2


g_D = fem.Function(V)
g_D.interpolate(u_manufactured)
bc = fem.dirichletbc(value=g_D, dofs=dofs)

# Next, we express the variational problem using UFL.
#
# First, the penalty parameter $\beta$ is defined. In addition, we define
# a variable `h` for the cell diameter $h_E$, a variable `n`for the
# outward-facing normal vector $n$ and a variable `h_avg` for the
# average size of cells sharing a facet
# $\left< h \right> = \frac{1}{2} (h_{+} + h_{-})$. Here, the UFL syntax
# `('+')` and `('-')` restricts a function to the `('+')` and `('-')`
# sides of a facet.

beta = fem.Constant(msh, default_scalar_type(50.0))
h = ufl.CellDiameter(msh)
n = ufl.FacetNormal(msh)
h_avg = (h("+") + h("-")) / 2.0

# After that, we can define the variational problem consisting of the
# bilinear form $a$ and the linear form $L$. We use {py:mod}`ufl` to derive
# the manufactured $f$ and $g_N$.
# Note that with `dS`, integration is carried out over all the interior
# facets $\mathcal{E}_h^{\rm int}$, whereas with `ds` it would be only
# the facets on the boundary of the domain, i.e. $\partial\Omega$.
# The jump operator
# $[\!\![ w ]\!\!] = w_{+} \cdot n_{+} + w_{-} \cdot n_{-}$ w.r.t. the
# outward-facing normal vector $n$ is in UFL available as `jump(w, n)`.

# +
# Define variational problem
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
x = ufl.SpatialCoordinate(msh)
u_ex = ufl.sin(2 * ufl.pi * x[0]) ** 2 * ufl.sin(2 * ufl.pi * x[1]) ** 2
f = ufl.div(ufl.grad(ufl.div(ufl.grad(u_ex))))
g_N = ufl.dot(ufl.grad(u_ex), n)

a = (
    ufl.inner(ufl.div(ufl.grad(u)), ufl.div(ufl.grad(v))) * ufl.dx
    - ufl.inner(ufl.avg(ufl.div(ufl.grad(u))), ufl.jump(ufl.grad(v), n)) * ufl.dS
    - ufl.inner(ufl.jump(ufl.grad(u), n), ufl.avg(ufl.div(ufl.grad(v)))) * ufl.dS
    + beta / h_avg * ufl.inner(ufl.jump(ufl.grad(u), n), ufl.jump(ufl.grad(v), n)) * ufl.dS
    - ufl.inner(ufl.div(ufl.grad(u)), ufl.dot(ufl.grad(v), n)) * ufl.ds
    - ufl.inner(ufl.div(ufl.grad(v)), ufl.dot(ufl.grad(u), n)) * ufl.ds
    + beta / h * ufl.inner(ufl.dot(ufl.grad(u), n), ufl.dot(ufl.grad(v), n)) * ufl.ds
)
L = (
    ufl.inner(f, v) * ufl.dx
    - ufl.inner(g_N, ufl.div(ufl.grad(v))) * ufl.ds
    + beta / h * ufl.inner(g_N, ufl.dot(ufl.grad(v), n)) * ufl.ds
)
# -

# We create a {py:class}`LinearProblem <dolfinx.fem.petsc.LinearProblem>`
# object that brings together the variational problem, the Dirichlet
# boundary condition, and which specifies the linear solver. In this
# case we use a direct (LU) solver. The {py:func}`solve
# <dolfinx.fem.petsc.LinearProblem.solve>` will compute a solution.

problem = LinearProblem(
    a,
    L,
    bcs=[bc],
    petsc_options_prefix="demo_biharmonic_",
    petsc_options={
        "ksp_type": "preonly",
        "pc_type": "lu",
    },
)
uh = problem.solve()
assert isinstance(uh, fem.Function)
assert problem.solver.getConvergedReason() > 0

# We compute the error between the computed and exact solution:

# +
error = fem.form(ufl.inner(uh - u_ex, uh - u_ex) * ufl.dx)
local_error = fem.assemble_scalar(error)
glob_error = np.sqrt(error.mesh.comm.allreduce(local_error, op=MPI.SUM))

print(f"Global_error: {glob_error:.5e}")

assert glob_error < 1e-3
# -

# The solution can be written to a VTX-file using {py:class}`VTXWriter
# <dolfinx.io.VTXWriter>` which can be opened with ParaView

out_folder = Path("out_biharmonic")
out_folder.mkdir(parents=True, exist_ok=True)
with io.VTXWriter(msh.comm, out_folder / "biharmonic.bp", [uh]) as file:
    file.write(0.0)

# and displayed using [pyvista](https://docs.pyvista.org/).

# +
try:
    import pyvista

    cells, types, x = plot.vtk_mesh(V)
    grid = pyvista.UnstructuredGrid(cells, types, x)
    grid.point_data["u"] = uh.x.array.real
    grid.set_active_scalars("u")
    plotter = pyvista.Plotter()
    plotter.add_mesh(grid, show_edges=True)
    warped = grid.warp_by_scalar()
    plotter.add_mesh(warped)
    if pyvista.OFF_SCREEN:
        plotter.screenshot(out_folder / "uh_biharmonic.png")
    else:
        plotter.show()
except ModuleNotFoundError:
    print("'pyvista' is required to visualise the solution")
    print("Install 'pyvista' with pip: 'python3 -m pip install pyvista'")

# -

# ```{bibliography}
# ```
