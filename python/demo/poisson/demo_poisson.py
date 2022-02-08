# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.6
# ---

# # Poisson equation
#
# This demo is implemented in a single Python file,
# {download}`demo_poisson.py`, which contains both the variational forms
# and the solver.
#
# This demo illustrates how to:
#
# - Solve a linear partial differential equation
# - Create and apply Dirichlet boundary conditions
# - Define a FunctionSpace
#
# The solution for $u$ in this demo will look as follows:
#
# ```{image} poisson_u.png
# :scale: 75 %
# ```
#
# ## Equation and problem definition
#
# The Poisson equation is the canonical elliptic partial differential
# equation.  For a domain $\Omega \subset \mathbb{R}^n$ with
# boundary $\partial \Omega = \Gamma_{D} \cup \Gamma_{N}$, the
# Poisson equation with particular boundary conditions reads:
#
# $$
# \begin{align}
# - \nabla^{2} u &= f \quad {\rm in} \ \Omega, \\
# u &= 0 \quad {\rm on} \ \Gamma_{D}, \\
# \nabla u \cdot n &= g \quad {\rm on} \ \Gamma_{N}. \\
# \end{align}
# $$
#
# Here, $f$ and $g$ are input data and $n$ denotes the
# outward directed boundary normal. The most standard variational form
# of Poisson equation reads: find $u \in V$ such that
#
# $$
# a(u, v) = L(v) \quad \forall \ v \in V,
# $$
#
# where $V$ is a suitable function space and
#
# $$
# \begin{align}
# a(u, v) &= \int_{\Omega} \nabla u \cdot \nabla v \, {\rm d} x, \\
# L(v)    &= \int_{\Omega} f v \, {\rm d} x + \int_{\Gamma_{N}} g v \, {\rm d} s.
# \end{align}
# $$
#
# The expression $a(u, v)$ is the bilinear form and $L(v)$
# is the linear form. It is assumed that all functions in $V$
# satisfy the Dirichlet boundary conditions ($u = 0 \ {\rm on} \
# \Gamma_{D}$).
#
# In this demo, we shall consider the following definitions of the input
# functions, the domain, and the boundaries:
#
# - $\Omega = [0,2] \times [0,1]$ (a rectangle)
# - $\Gamma_{D} = \{(0, y) \cup (1, y) \subset \partial \Omega\}$
#   (Dirichlet boundary)
# - $\Gamma_{N} = \{(x, 0) \cup (x, 1) \subset \partial \Omega\}$
#   (Neumann boundary)
# - $g = \sin(5x)$ (normal derivative)
# - $f = 10\exp(-((x - 0.5)^2 + (y - 0.5)^2) / 0.02)$ (source
#   term)
#
# ## Implementation
#
# This description goes through the implementation (in
# {download}`demo_poisson.py`) of a solver for the above described
# Poisson equation step-by-step.
#
# First, the {py:mod}`dolfinx` module is imported:

# +
import numpy as np

import ufl
from dolfinx import fem, io, mesh, plot
from ufl import ds, dx, exp, grad, inner, sin

from mpi4py import MPI
from petsc4py.PETSc import ScalarType
# -

# We begin by defining a mesh of the domain and a finite element
# function space $V$ relative to this mesh. We create a rectangular mesh
# using built-in function provided by class
# {py:class}`create_rectangle<dolfinx.mesh.create_rectangle>`. In order
# to create a mesh consisting of 32 x 16 squares with each square
# divided into two triangles, we do as follows

# +
# Create mesh
rect_mesh = mesh.create_rectangle(
    comm=MPI.COMM_WORLD,
    points=((0.0, 0.0), (2.0, 1.0)),
    n=(32, 16),
    cell_type=mesh.CellType.triangle,
)

# Define function space
V = fem.FunctionSpace(rect_mesh, ("Lagrange", 1))
# -

# The second argument to {py:class}`FunctionSpace
# <dolfinx.fem.FunctionSpace>` is a tuple consisting of `(family,
# degree)`, where `family` is the finite element family, and `degree`
# specifies the polynomial degree. Thus, in this case, our space `V`
# consists of first-order, continuous Lagrange finite element functions
# (or in order words, continuous piecewise linear polynomials).
#
# Next, we want to consider the Dirichlet boundary condition. A simple
# Python function, returning a boolean, can be used to define the
# boundary for the Dirichlet boundary condition ($\Gamma_D$). The
# function should return `True` for those points inside the boundary and
# `False` for the points outside. In our case, we want to say that the
# points $(x, y)$ such that $x = 0$ or $x = 1$ are inside on the inside
# of $\Gamma_D$. (Note that because of rounding-off errors, it is often
# wise to instead specify $x < \epsilon$ or $x > 1 - \epsilon$ where
# $\epsilon$ is a small number (such as machine precision).)
#
# Now, the Dirichlet boundary condition can be created using the class
# {py:class}`DirichletBC <dolfinx.fem.DirichletBC>`. A
# {py:class}`DirichletBC <dolfinx.fem.DirichletBC>` takes three
# arguments: the value of the boundary condition, the part of the
# boundary which the condition apply to, and the function space. This
# boundary part is identified with degrees of freedom in the function
# space to which we apply the boundary conditions.
#
# To identify the degrees of freedom, we first find the facets (entities
# of dimension 1) that likes on the boundary of the mesh, and satisfies
# our criteria for $\Gamma_D$. Then, we use the function
# `locate_dofs_topological` to identify all degrees of freedom that is
# located on the facet (including the vertices). In our example, the
# function space is `V`, the value of the boundary condition (0.0) can
# represented using a {py:class}`Constant <dolfinx.fem.Constant>` and
# the Dirichlet boundary is defined immediately above. The definition of
# the Dirichlet boundary condition then looks as follows:
#
# Define boundary condition on x = 0 or x = 1

# Locate the facets on the boundary matching the provided marker
# function
facets = mesh.locate_entities_boundary(
    rect_mesh,
    dim=1,
    marker=lambda x: np.logical_or(np.isclose(x[0], 0.0), np.isclose(x[0], 2.0)),
)

# Locate the degrees of freedom for the relevant function space that
# intersects the facets where we what to impose the boundary condition
dofs = fem.locate_dofs_topological(V=V, entity_dim=1, entities=facets)
# Define the boundary condition
bc = fem.dirichletbc(value=ScalarType(0), dofs=dofs, V=V)

# Next, we want to express the variational problem.  First, we need to
# specify the trial function $u$ and the test function $v$, both living
# in the function space $V$. We do this by defining a
# {py:class}`TrialFunction <ufl.TrialFunction>` and a
# {py:class}`TestFunction <ufl.TrialFunction>` on the previously defined
# {py:class}`FunctionSpace <dolfinx.fem.FunctionSpace>` `V`.
#
# Further, the source $f$ and the boundary normal derivative $g$ are
# involved in the variational forms, and hence we must specify these.
#
# With these ingredients, we can write down the bilinear form `a` and
# the linear form `L` (using UFL operators). In summary, this reads

# +
# Define variational problem
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
x = ufl.SpatialCoordinate(rect_mesh)
f = 10 * exp(-((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2) / 0.02)
g = sin(5 * x[0])
a = inner(grad(u), grad(v)) * dx
L = inner(f, v) * dx + inner(g, v) * ds
# -

# Now, we have specified the variational forms and can consider the
# solution of the variational problem. First, we need to define a
#  {py:class}`Function <dolfinx.fem.Function>` `u` to represent the
# solution. (Upon initialization, it is simply set to the zero
# function.) A  {py:class}`Function <dolfinx.fem.Function>` represents a
# function living in a finite element function space. Next, we
# initialize a solver using the {py:class}`LinearProblem
#  <dolfinx.fem.LinearProblem>`. This class is initialized with the
# arguments `a`, `L`, and `bc` as follows:

# +
# In this problem, we use a direct LU solver, which is defined through
# the dictionary `petsc_options`.
problem = fem.LinearProblem(
    a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
)

# When we want to compute the solution to the problem, we can specify
# what kind of solver we want to use.
uh = problem.solve()
# -

# The function `u` will be modified during the call to solve. The
# default settings for solving a variational problem have been used.
# However, the solution process can be controlled in much more detail if
# desired.
#
# A {py:class}`Function <dolfinx.fem.Function>` can be manipulated in
# various ways, in particular, it can be plotted and saved to file.
# Here, we output the solution to an XDMF file for later visualization
# and also plot it using pyvista.

# +
# Save solution in XDMF format
with io.XDMFFile(MPI.COMM_WORLD, "poisson.xdmf", "w") as file:
    file.write_mesh(rect_mesh)
    file.write_function(uh)

# Plot solution
try:
    import pyvista

    cells, types, x = plot.create_vtk_mesh(V)
    grid = pyvista.UnstructuredGrid(cells, types, x)
    grid.point_data["u"] = uh.x.array.real
    grid.set_active_scalars("u")

    plotter = pyvista.Plotter()
    plotter.add_mesh(grid, show_edges=True)
    warped = grid.warp_by_scalar()
    plotter.add_mesh(warped)

    plotter.show()
except ModuleNotFoundError:
    print("'pyvista' is required to visualise the solution")
    print("Install 'pyvista' with pip: 'python3 -m pip install pyvista'")
