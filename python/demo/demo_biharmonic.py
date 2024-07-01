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
# This demo is implemented in a single Python file,
# {download}`demo_biharmonic.py`, which contains both the variational forms
# and the solver. It illustrates how to:
#
# - Solve a linear partial differential equation
# - Use a discontinuous Galerkin method
# - Solve a fourth-order differential equation
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
# where $\nabla^{4} \equiv \nabla^{2} \nabla^{2}$ is the biharmonic operator
# and $f$ is a prescribed source term.
# To formulate a complete boundary value problem, the biharmonic equation
# must be complemented by suitable boundary conditions.
#
# ### Weak formulation
#
# Multiplying the biharmonic equation by a test function and integrating
# by parts twice leads to a problem of second-order derivatives, which would
# require $H^{2}$ conforming (roughly $C^{1}$ continuous) basis functions.
# To solve the biharmonic equation using Lagrange finite element basis
# functions, the biharmonic equation can be split into two second-order
# equations (see the Mixed Poisson demo for a mixed method for the Poisson
# equation), or a variational formulation can be constructed that imposes
# weak continuity of normal derivatives between finite element cells.
# This demo uses a discontinuous Galerkin approach to impose continuity
# of the normal derivative weakly.
#
# Consider a triangulation $\mathcal{T}$ of the domain $\Omega$, where
# the set of interior facets is denoted by $\mathcal{E}_h^{\rm int}$.
# Functions evaluated on opposite sides of a facet are indicated by the
# subscripts $+$ and $-$.
# Using the standard continuous Lagrange finite element space
#
# $$
# V = \left\{v \in H^{1}_{0}(\Omega)\,:\, v \in P_{k}(K) \
# \forall \ K \in \mathcal{T} \right\}
# $$
#
# and considering the boundary conditions
#
# $$
# \begin{align}
# u &= 0 \quad {\rm on} \ \partial\Omega, \\
# \nabla^{2} u &= 0 \quad {\rm on} \ \partial\Omega,
# \end{align}
# $$
#
# a weak formulation of the biharmonic problem reads: find $u \in V$ such that
#
# $$
# a(u,v)=L(v) \quad \forall \ v \in V,
# $$
#
# where the bilinear form is
#
# $$
# a(u, v) =
# \sum_{K \in \mathcal{T}} \int_{K} \nabla^{2} u \nabla^{2} v \, {\rm d}x \
# +\sum_{E \in \mathcal{E}_h^{\rm int}}\left(\int_{E} \frac{\alpha}{h_E}
# [\!\![ \nabla u ]\!\!] [\!\![ \nabla v ]\!\!] \, {\rm d}s
# - \int_{E} \left<\nabla^{2} u \right>[\!\![ \nabla v ]\!\!]  \, {\rm d}s
# - \int_{E} [\!\![ \nabla u ]\!\!] \left<\nabla^{2} v \right> \,
# {\rm d}s\right)
# $$
#
# and the linear form is
#
# $$
# L(v) = \int_{\Omega} fv \, {\rm d}x.
# $$
#
# Furthermore, $\left< u \right> = \frac{1}{2} (u_{+} + u_{-})$,
# $[\!\![ w ]\!\!]  = w_{+} \cdot n_{+} + w_{-} \cdot n_{-}$,
# $\alpha \ge 0$ is a penalty parameter and
# $h_E$ is a measure of the cell size.
#
# The input parameters for this demo are defined as follows:
#
# - $\Omega = [0,1] \times [0,1]$ (a unit square)
# - $\alpha = 8.0$ (penalty parameter)
# - $f = 4.0 \pi^4\sin(\pi x)\sin(\pi y)$ (source term)
#
# ## Implementation
#
# We first import the modules and functions that the program uses:

import importlib.util

if importlib.util.find_spec("petsc4py") is not None:
    import dolfinx

    if not dolfinx.has_petsc:
        print("This demo requires DOLFINx to be compiled with PETSc enabled.")
        exit(0)
    from petsc4py.PETSc import ScalarType  # type: ignore
else:
    print("This demo requires petsc4py.")
    exit(0)

from mpi4py import MPI

# +
import dolfinx
import ufl
from dolfinx import fem, io, mesh, plot
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import CellType, GhostMode
from ufl import CellDiameter, FacetNormal, avg, div, dS, dx, grad, inner, jump, pi, sin

# -

# We begin by using {py:func}`create_rectangle
# <dolfinx.mesh.create_rectangle>` to create a rectangular
# {py:class}`Mesh <dolfinx.mesh.Mesh>` of the domain, and creating a
# finite element {py:class}`FunctionSpace <dolfinx.fem.FunctionSpace>`
# $V$ on the mesh.

msh = mesh.create_rectangle(
    comm=MPI.COMM_WORLD,
    points=((0.0, 0.0), (1.0, 1.0)),
    n=(32, 32),
    cell_type=CellType.triangle,
    ghost_mode=GhostMode.shared_facet,
)
V = fem.functionspace(msh, ("Lagrange", 2))

# The second argument to {py:func}`functionspace
# <dolfinx.fem.functionspace>` is a tuple consisting of `(family,
# degree)`, where `family` is the finite element family, and `degree`
# specifies the polynomial degree. in this case `V` consists of
# second-order, continuous Lagrange finite element functions.
#
# Next, we locate the mesh facets that lie on the boundary
# $\Gamma_D = \partial\Omega$.
# We do this using using {py:func}`locate_entities_boundary
# <dolfinx.mesh.locate_entities_boundary>` and providing  a marker
# function that returns `True` for points `x` on the boundary and
# `False` otherwise.

msh.topology.create_connectivity(msh.topology.dim-1, msh.topology.dim)
facets = mesh.exterior_facet_indices(msh.topology)

# We now find the degrees-of-freedom that are associated with the
# boundary facets using {py:func}`locate_dofs_topological
# <dolfinx.fem.locate_dofs_topological>`

dofs = fem.locate_dofs_topological(V=V, entity_dim=1, entities=facets)

# and use {py:func}`dirichletbc <dolfinx.fem.dirichletbc>` to create a
# {py:class}`DirichletBC <dolfinx.fem.DirichletBC>`
# class that represents the boundary condition. In this case, we impose
# Dirichlet boundary conditions with value $0$ on the entire boundary
# $\partial\Omega$.

bc = fem.dirichletbc(value=ScalarType(0), dofs=dofs, V=V)

# Next, we express the variational problem using UFL.
#
# First, the penalty parameter $\alpha$ is defined. In addition, we define a
# variable `h` for the cell diameter $h_E$, a variable `n`for the
# outward-facing normal vector $n$ and a variable `h_avg` for the
# average size of cells sharing a facet
# $\left< h \right> = \frac{1}{2} (h_{+} + h_{-})$. Here, the UFL syntax
# `('+')` and `('-')` restricts a function to the `('+')` and `('-')`
# sides of a facet.

alpha = ScalarType(8.0)
h = CellDiameter(msh)
n = FacetNormal(msh)
h_avg = (h("+") + h("-")) / 2.0

# After that, we can define the variational problem consisting of the bilinear
# form $a$ and the linear form $L$. The source term is prescribed as
# $f = 4.0 \pi^4\sin(\pi x)\sin(\pi y)$. Note that with `dS`, integration is
# carried out over all the interior facets $\mathcal{E}_h^{\rm int}$, whereas
# with `ds` it would be only the facets on the boundary of the domain, i.e.
# $\partial\Omega$. The jump operator
# $[\!\![ w ]\!\!] = w_{+} \cdot n_{+} + w_{-} \cdot n_{-}$ w.r.t. the
# outward-facing normal vector $n$ is in UFL available as `jump(w, n)`.

# +
# Define variational problem
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
x = ufl.SpatialCoordinate(msh)
f = 4.0 * pi**4 * sin(pi * x[0]) * sin(pi * x[1])

a = (
    inner(div(grad(u)), div(grad(v))) * dx
    - inner(avg(div(grad(u))), jump(grad(v), n)) * dS
    - inner(jump(grad(u), n), avg(div(grad(v)))) * dS
    + alpha / h_avg * inner(jump(grad(u), n), jump(grad(v), n)) * dS
)
L = inner(f, v) * dx
# -

# We create a {py:class}`LinearProblem <dolfinx.fem.petsc.LinearProblem>`
# object that brings together the variational problem, the Dirichlet
# boundary condition, and which specifies the linear solver. In this
# case we use a direct (LU) solver. The {py:func}`solve
# <dolfinx.fem.petsc.LinearProblem.solve>` will compute a solution.

problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()

# The solution can be written to a  {py:class}`XDMFFile
# <dolfinx.io.XDMFFile>` file visualization with ParaView or VisIt

with io.XDMFFile(msh.comm, "out_biharmonic/biharmonic.xdmf", "w") as file:
    V1 = fem.functionspace(msh, ("Lagrange", 1))
    u1 = fem.Function(V1)
    u1.interpolate(uh)
    file.write_mesh(msh)
    file.write_function(u1)

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
        pyvista.start_xvfb(wait=0.1)
        plotter.screenshot("uh_biharmonic.png")
    else:
        plotter.show()
except ModuleNotFoundError:
    print("'pyvista' is required to visualise the solution")
    print("Install 'pyvista' with pip: 'python3 -m pip install pyvista'")
