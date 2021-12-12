#
# .. _demo_poisson_equation:
#
# Poisson equation
# ================
#
# This demo is implemented in a single Python file,
# :download:`demo_poisson.py`, which contains both the variational forms
# and the solver.
#
# This demo illustrates how to:
#
# * Solve a linear partial differential equation
# * Create and apply Dirichlet boundary conditions
# * Define a FunctionSpace
#
# The solution for :math:`u` in this demo will look as follows:
#
# .. image:: poisson_u.png
#    :scale: 75 %
#
#
# Equation and problem definition
# -------------------------------
#
# The Poisson equation is the canonical elliptic partial differential
# equation.  For a domain :math:`\Omega \subset \mathbb{R}^n` with
# boundary :math:`\partial \Omega = \Gamma_{D} \cup \Gamma_{N}`, the
# Poisson equation with particular boundary conditions reads:
#
# .. math::
#    - \nabla^{2} u &= f \quad {\rm in} \ \Omega, \\
#                 u &= 0 \quad {\rm on} \ \Gamma_{D}, \\
#                 \nabla u \cdot n &= g \quad {\rm on} \ \Gamma_{N}. \\
#
# Here, :math:`f` and :math:`g` are input data and :math:`n` denotes the
# outward directed boundary normal. The most standard variational form
# of Poisson equation reads: find :math:`u \in V` such that
#
# .. math:: a(u, v) = L(v) \quad \forall \ v \in V,
#
# where :math:`V` is a suitable function space and
#
# .. math:: a(u, v) &= \int_{\Omega} \nabla u \cdot \nabla v \, {\rm d}
#    x, \\
#    L(v)    &= \int_{\Omega} f v \, {\rm d} x
#    + \int_{\Gamma_{N}} g v \, {\rm d} s.
#
# The expression :math:`a(u, v)` is the bilinear form and :math:`L(v)`
# is the linear form. It is assumed that all functions in :math:`V`
# satisfy the Dirichlet boundary conditions (:math:`u = 0 \ {\rm on} \
# \Gamma_{D}`).
#
# In this demo, we shall consider the following definitions of the input
# functions, the domain, and the boundaries:
#
# * :math:`\Omega = [0,1] \times [0,1]` (a unit square)
# * :math:`\Gamma_{D} = \{(0, y) \cup (1, y) \subset \partial \Omega\}`
#   (Dirichlet boundary)
# * :math:`\Gamma_{N} = \{(x, 0) \cup (x, 1) \subset \partial \Omega\}`
#   (Neumann boundary)
# * :math:`g = \sin(5x)` (normal derivative)
# * :math:`f = 10\exp(-((x - 0.5)^2 + (y - 0.5)^2) / 0.02)` (source
#   term)
#
#
# Implementation
# --------------
#
# This description goes through the implementation (in
# :download:`demo_poisson.py`) of a solver for the above described
# Poisson equation step-by-step.
#
# First, the :py:mod:`dolfinx` module is imported: ::

import numpy as np

import ufl
from dolfinx import fem, plot
from dolfinx.fem import (DirichletBC, Function, FunctionSpace,
                         locate_dofs_topological)
from dolfinx.generation import RectangleMesh
from dolfinx.io import XDMFFile
from dolfinx.mesh import CellType, GhostMode, locate_entities_boundary
from ufl import ds, dx, grad, inner

from mpi4py import MPI

# We begin by defining a mesh of the domain and a finite element
# function space :math:`V` relative to this mesh. As the unit square is
# a very standard domain, we can use a built-in mesh provided by the
# class :py:class:`UnitSquareMesh <dolfinx.generation.UnitSquareMesh>`. In
# order to create a mesh consisting of 32 x 32 squares with each square
# divided into two triangles, we do as follows ::

# Create mesh and define function space
mesh = RectangleMesh(
    MPI.COMM_WORLD,
    [np.array([0, 0, 0]), np.array([1, 1, 0])], [32, 32],
    CellType.triangle, GhostMode.none)

V = FunctionSpace(mesh, ("Lagrange", 1))

# The second argument to :py:class:`FunctionSpace
# <dolfinx.fem.FunctionSpace>` is the finite element family, while the
# third argument specifies the polynomial degree. Thus, in this case,
# our space ``V`` consists of first-order, continuous Lagrange finite
# element functions (or in order words, continuous piecewise linear
# polynomials).
#
# Next, we want to consider the Dirichlet boundary condition. A simple
# Python function, returning a boolean, can be used to define the
# boundary for the Dirichlet boundary condition (:math:`\Gamma_D`). The
# function should return ``True`` for those points inside the boundary
# and ``False`` for the points outside. In our case, we want to say that
# the points :math:`(x, y)` such that :math:`x = 0` or :math:`x = 1` are
# inside on the inside of :math:`\Gamma_D`. (Note that because of
# rounding-off errors, it is often wise to instead specify :math:`x <
# \epsilon` or :math:`x > 1 - \epsilon` where :math:`\epsilon` is a
# small number (such as machine precision).)
#
# Now, the Dirichlet boundary condition can be created using the class
# :py:class:`DirichletBC <dolfinx.fem.bcs.DirichletBC>`. A
# :py:class:`DirichletBC <dolfinx.fem.bcs.DirichletBC>` takes two
# arguments: the value of the boundary condition and the part of the
# boundary on which the condition applies. This boundary part is
# identified with degrees of freedom in the function space to which we
# apply the boundary conditions. A method ``locate_dofs_geometrical`` is
# provided to extract the boundary degrees of freedom using a
# geometrical criterium. In our example, the function space is ``V``,
# the value of the boundary condition (0.0) can represented using a
# :py:class:`Function <dolfinx.functions.Function>` and the Dirichlet
# boundary is defined immediately above. The definition of the Dirichlet
# boundary condition then looks as follows: ::

# Define boundary condition on x = 0 or x = 1
u0 = Function(V)
u0.x.array[:] = 0.0
facets = locate_entities_boundary(mesh, 1,
                                  lambda x: np.logical_or(np.isclose(x[0], 0.0),
                                                          np.isclose(x[0], 1.0)))
bc = DirichletBC(u0, locate_dofs_topological(V, 1, facets))

# Next, we want to express the variational problem.  First, we need to
# specify the trial function :math:`u` and the test function :math:`v`,
# both living in the function space :math:`V`. We do this by defining a
# :py:class:`TrialFunction <dolfinx.functions.fem.TrialFunction>` and a
# :py:class:`TestFunction <dolfinx.functions.fem.TrialFunction>` on the
# previously defined :py:class:`FunctionSpace
# <dolfinx.fem.FunctionSpace>` ``V``.
#
# Further, the source :math:`f` and the boundary normal derivative
# :math:`g` are involved in the variational forms, and hence we must
# specify these.
#
# With these ingredients, we can write down the bilinear form ``a`` and
# the linear form ``L`` (using UFL operators). In summary, this reads ::

# Define variational problem
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
x = ufl.SpatialCoordinate(mesh)
f = 10 * ufl.exp(-((x[0] - 0.5)**2 + (x[1] - 0.5)**2) / 0.02)
g = ufl.sin(5 * x[0])
a = inner(grad(u), grad(v)) * dx
L = inner(f, v) * dx + inner(g, v) * ds

# Now, we have specified the variational forms and can consider the
# solution of the variational problem. First, we need to define a
# :py:class:`Function <dolfinx.functions.fem.Function>` ``u`` to
# represent the solution. (Upon initialization, it is simply set to the
# zero function.) A :py:class:`Function
# <dolfinx.functions.fem.Function>` represents a function living in a
# finite element function space. Next, we initialize a solver using the
# :py:class:`LinearProblem <dolfinx.fem.linearproblem.LinearProblem>`.
# This class is initialized with the arguments ``a``, ``L``, and ``bc``
# as follows: :: In this problem, we use a direct LU solver, which is
# defined through the dictionary ``petsc_options``.
problem = fem.LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

# When we want to compute the solution to the problem, we can specify
# what kind of solver we want to use.
uh = problem.solve()

# The function ``u`` will be modified during the call to solve. The
# default settings for solving a variational problem have been used.
# However, the solution process can be controlled in much more detail if
# desired.
#
# A :py:class:`Function <dolfinx.functions.fem.Function>` can be
# manipulated in various ways, in particular, it can be plotted and
# saved to file. Here, we output the solution to an ``XDMF`` file for
# later visualization and also plot it using the :py:func:`plot
# <dolfinx.common.plot.plot>` command: ::

# Save solution in XDMF format
with XDMFFile(MPI.COMM_WORLD, "poisson.xdmf", "w") as file:
    file.write_mesh(mesh)
    file.write_function(uh)


# Update ghost entries and plot
try:
    import pyvista
    uh.x.scatter_forward()
    topology, cell_types = plot.create_vtk_topology(mesh, mesh.topology.dim)
    grid = pyvista.UnstructuredGrid(topology, cell_types, mesh.geometry.x)
    grid.point_data["u"] = uh.compute_point_values().real
    grid.set_active_scalars("u")

    plotter = pyvista.Plotter()
    plotter.add_mesh(grid, show_edges=True)
    warped = grid.warp_by_scalar()
    plotter.add_mesh(warped)

    # If pyvista environment variable is set to off-screen (static)
    # plotting save png
    if pyvista.OFF_SCREEN:
        pyvista.start_xvfb(wait=0.1)
        plotter.screenshot("uh.png")
    else:
        plotter.show()
except ModuleNotFoundError:
    print("pyvista is required to visualise the solution")
