#
# .. _demo_poisson_equation:
#
# Poisson equation
# ================
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
# .. math::
#    a(u, v) = L(v) \quad \forall \ v \in V,
#
# where :math:`V` is a suitable function space and
#
# .. math::
#    a(u, v) &= \int_{\Omega} \nabla u \cdot \nabla v \, {\rm d} x, \\
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
# First, all necessary modules, submodules, classes and methods which are used in the code are
# imported. ::

import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI

import dolfinx
import dolfinx.plotting
import ufl
from dolfinx import DirichletBC, Function, FunctionSpace, RectangleMesh, solve
from dolfinx.cpp.mesh import CellType
from dolfinx.fem import locate_dofs_topological
from dolfinx.io import XDMFFile
from dolfinx.mesh import locate_entities_boundary
from ufl import ds, dx, grad, inner

# Built-in mesh generation
# ------------------------
#
# Dolfin-x comes with several helpers for generation of simple 1D, 2D and
# 3D meshes. All available methods are listed in :py:mod:`dolfinx.generation <dolfinx.generation>`
# submodule. The following creates a mesh consisting of 32 x 32 squares with each square
# divided into two triangles.
#
# .. note::
#    A first call to any mesh generation helper invokes a form compiler, FFCX, and this comes with
#    slight time penalty and necessary file IO operations.
#
# ::

mesh = RectangleMesh(
    MPI.COMM_WORLD,
    [np.array([0, 0, 0]), np.array([1, 1, 0])], [32, 32],
    CellType.triangle, dolfinx.cpp.mesh.GhostMode.none)

# Create function space
# ---------------------
#
# Function space which spans basis function over prepared mesh
# is created using :py:class:`FunctionSpace <dolfinx.function.FunctionSpace>`. ::

V = FunctionSpace(mesh, ("Lagrange", 1))

# The second argument to :py:class:`FunctionSpace <dolfinx.function.FunctionSpace>` is the tuple of
# a finite element family and a polynomial degree. Thus, in this case,
# our space ``V`` consists of first-order, continuous Lagrange finite element functions (or in order words,
# continuous piecewise linear polynomials).
#
# .. note::
#    Creating function space involves the call to FFCX compiler and filesystem IO as well. If you'd like
#    to see when the form compiler is invoked, increase the
#    FFCX verbosity, see :ref:`controlling-compilation-parameters`.
#
#
# Finding mesh entities (facets) and their degrees-of-freedom
# -----------------------------------------------------------
#
# In order to apply Dirichlet boundary condition, we must first find which
# facets lie on the desired boundary. There are two methods to help find mesh entities,
# :py:func:`dolfinx.mesh.locate_entities_boundary` and :py:func:`dolfinx.mesh.locate_entities`.
#
# Facets will be marked using a callback function, which identifies points with :math:`x = 0` or
# :math:`x = 1`. We are using NumPy's boolean operations, because the coordinates ``x`` array
# is a big array with shape ``(gdim, num_vertices)``, where ``num_vertices`` is number of all
# vertices in the mesh which has to be checked. ::

facets = locate_entities_boundary(mesh, 1,
                                  lambda x: np.logical_or(np.isclose(x[0], 0.0),
                                                          np.isclose(x[0], 1.0)))

# Such returned result is a NumPy array containg local numbers of matched facets,
# so it makes it easy to debug.
#
# We need to find which degrees-of-freedom from the space ``V`` belong to the identified
# mesh facets. Dolfin-x provides :py:func:`dolfinx.fem.locate_dofs_topological`
# for this purpose. ::

dofs = locate_dofs_topological(V, 1, facets)

# Preparing Dirichlet boundary condition
# --------------------------------------
#
# For Dirichlet (strong) boundary condition we need a class
# :py:class:`DirichletBC <dolfinx.fem.DirichletBC>`.
# This class will be used to apply boundary conditions for a specific degrees-of-freedom
# and with the values from :py:class:`dolfinx.function.Function`.
# Degrees-of-freedom were found above, so we only need to prepare an empty function. ::

u0 = Function(V)
u0.vector.set(0.0)
bc = DirichletBC(u0, dofs)

# Defining variational problem
# ----------------------------
#
# Next, we want to express the variational problem.  First, we need to
# specify the trial function :math:`u` and the test function :math:`v`,
# both living in the function space :math:`V`. We do this by defining a
# :py:class:`ufl.TrialFunction <ufl.TrialFunction>`
# and a :py:class:`ufl.TestFunction <ufl.TestFunction>` on the previously defined
# :py:class:`FunctionSpace <dolfinx.function.FunctionSpace>` ``V``.
#
# .. note::
#    In Dolfin-x, a class is defined using UFL imports directly, if there is
#    no need to attach a non-UFL specific data to it. Typical examples of
#    direct calls to UFL include test and trial function, mathematical
#    functions and usual weak form tensor operations.
#
# Further, the source :math:`f` and the boundary normal derivative
# :math:`g` are involved in the variational forms, and hence we must
# specify these.
# With these ingredients, we can write down the bilinear form ``a`` and
# the linear form ``L`` (using UFL operators). In summary, this reads ::

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

x = ufl.SpatialCoordinate(mesh)
f = 10 * ufl.exp(-((x[0] - 0.5)**2 + (x[1] - 0.5)**2) / 0.02)
g = ufl.sin(5 * x[0])

a = inner(grad(u), grad(v)) * dx
L = inner(f, v) * dx + inner(g, v) * ds

# Solving linear problem with ``solve()``
# ---------------------------------------
#
# Now, we have specified the variational forms and can consider the
# solution of the variational problem. First, we need to define a
# :py:class:`Function <dolfinx.function.Function>` ``u0`` to
# represent the solution. (Upon initialization, it is simply set to the
# zero function.) A :py:class:`Function
# <dolfinx.function.Function>` represents a function living in
# a finite element function space. Next, we can call the :py:func:`solve
# <dolfinx.fem.solving.solve>` as follows: ::

u0 = Function(V)
solve(a == L, u0, bc, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

# :py:func:`Solve <dolfinx.fem.solving.solve>` will assemble the bilinear and linear
# forms into matrix and vector, apply boundary conditions and call PETSc
# to solve the linear system.
#
# The function ``u0`` will be modified during the call to solve.
# PETSc solvers could be controlled with ``petsc_options`` argument.
# This expects a dictionary which follows the standard PETSc options
# naming. In this demo a direct solver is used.

# Storing function with ``XDMFFile``
# ----------------------------------
#
# A :py:class:`Function <dolfinx.function.Function>` can be
# manipulated in various ways, in particular, it can be plotted and
# saved to file. Here, we output the solution to an ``XDMF`` file
# for later visualization and also plot it using
# the :py:func:`plot <dolfinx.common.plot.plot>` command: ::

with XDMFFile(MPI.COMM_WORLD, "poisson.xdmf", "w") as file:
    file.write_mesh(mesh)
    file.write_function(u0)

# A simple ``matplotlob`` plotting interface is available at :py:func:`dolfinx.plotting.plot`.
# We can store the result into an image file. ::

im = dolfinx.plotting.plot(u0)
plt.colorbar(im)
plt.savefig("poisson.png")

#
# .. image:: poisson.png
#    :scale: 75 %
#    :align: center
#
