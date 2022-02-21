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
# and the solver. It illustrates how to:
#
# - Solve a linear partial differential equation
# - Define a {py:class}`FunctionSpace <dolfinx.fem.FunctionSpace>`
# - Create and apply Dirichlet boundary conditions
#
# ## Equation and problem definition
#
# For a domain $\Omega \subset \mathbb{R}^n$ with boundary $\partial
# \Omega = \Gamma_{D} \cup \Gamma_{N}$, the Poisson equation with
# particular boundary conditions reads:
#
# $$
# \begin{align}
# - \nabla^{2} u &= f \quad {\rm in} \ \Omega, \\
# u &= 0 \quad {\rm on} \ \Gamma_{D}, \\
# \nabla u \cdot n &= g \quad {\rm on} \ \Gamma_{N}. \\
# \end{align}
# $$
#
# where $f$ and $g$ are input data and $n$ denotes the outward directed
# boundary normal. The variational problem reads: find $u \in V$ such
# that
#
# $$
# a(u, v) = L(v) \quad \forall \ v \in V,
# $$
#
# where $V$ is a suitable function space and
#
# $$
# \begin{align}
# a(u, v) &:= \int_{\Omega} \nabla u \cdot \nabla v \, {\rm d} x, \\
# L(v)    &:= \int_{\Omega} f v \, {\rm d} x + \int_{\Gamma_{N}} g v \, {\rm d} s.
# \end{align}
# $$
#
# The expression $a(u, v)$ is the bilinear form and $L(v)$
# is the linear form. It is assumed that all functions in $V$
# satisfy the Dirichlet boundary conditions ($u = 0 \ {\rm on} \
# \Gamma_{D}$).
#
# In this demo we consider:
#
# - $\Omega = [0,2] \times [0,1]$ (a rectangle)
# - $\Gamma_{D} = \{(0, y) \cup (1, y) \subset \partial \Omega\}$
# - $\Gamma_{N} = \{(x, 0) \cup (x, 1) \subset \partial \Omega\}$
# - $g = \sin(5x)$
# - $f = 10\exp(-((x - 0.5)^2 + (y - 0.5)^2) / 0.02)$
#
# ## Implementation
#
# The modules that will be used are imported:

# +
import numpy as np

import ufl
from dolfinx import fem, io, mesh, plot
from ufl import ds, dx, grad, inner

from mpi4py import MPI
from petsc4py.PETSc import ScalarType
# -

# We begin by using {py:func}`create_rectangle
# <dolfinx.mesh.create_rectangle>` to create a rectangular
# {py:class}`Mesh <dolfinx.mesh.Mesh>` of the domain, and creating a
# finite element {py:class}`FunctionSpace <dolfinx.fem.FunctionSpace>`
# $V$ on the mesh.

# +
msh = mesh.create_rectangle(comm=MPI.COMM_WORLD,
                            points=((0.0, 0.0), (2.0, 1.0)), n=(32, 16),
                            cell_type=mesh.CellType.triangle,)
V = fem.FunctionSpace(msh, ("Lagrange", 1))
# -

# The second argument to {py:class}`FunctionSpace
# <dolfinx.fem.FunctionSpace>` is a tuple consisting of `(family,
# degree)`, where `family` is the finite element family, and `degree`
# specifies the polynomial degree. in this case `V` consists of
# first-order, continuous Lagrange finite element functions.
#
# Next, we locate the mesh facets that lie on the boundary $\Gamma_D$.
# We do this using using {py:func}`locate_entities_boundary
# <dolfinx.mesh.locate_entities_boundary>` and providing  a marker
# function that returns `True` for points `x` on the boundary and
# `False` otherwise.

facets = mesh.locate_entities_boundary(msh, dim=1,
                                       marker=lambda x: np.logical_or(np.isclose(x[0], 0.0),
                                                                      np.isclose(x[0], 2.0)))

# We now find the degrees-of-freedom that are associated with the
# boundary facets using {py:func}`locate_dofs_topological
# <dolfinx.fem.locate_dofs_topological>`

dofs = fem.locate_dofs_topological(V=V, entity_dim=1, entities=facets)

# and use {py:func}`dirichletbc <dolfinx.fem.dirichletbc>` to create a
# {py:class}`DirichletBCMetaClass <dolfinx.fem.DirichletBCMetaClass>`
# class that represents the boundary condition

bc = fem.dirichletbc(value=ScalarType(0), dofs=dofs, V=V)

# Next, we express the variational problem using UFL.

# +
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
x = ufl.SpatialCoordinate(msh)
f = 10 * ufl.exp(-((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2) / 0.02)
g = ufl.sin(5 * x[0])
a = inner(grad(u), grad(v)) * dx
L = inner(f, v) * dx + inner(g, v) * ds
# -

# We create a {py:class}`LinearProblem <dolfinx.fem.LinearProblem>`
# object that brings together the variational problem, the Dirichlet
# boundary condition, and which specifies the linear solver. In this
# case we use a direct (LU) solver. The {py:func}`solve
# <dolfinx.fem.LinearProblem.solve>` will compute a solution.

# +
problem = fem.petsc.LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()
# -

# The solution can be written to a  {py:class}`XDMFFile
# <dolfinx.io.XDMFFile>` file visualization with ParaView ot VisIt

# +
with io.XDMFFile(msh.comm, "poisson.xdmf", "w") as file:
    file.write_mesh(msh)
    file.write_function(uh)
# -

# and displayed using [pyvista](https://docs.pyvista.org/).

# +
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
# -
