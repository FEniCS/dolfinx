#
# .. _demo_hyperelasticity:
#
# Hyperelasticity
# ===============

# This demo is implemented in a single Python file,
# :download:`demo_hyperelasticity.py`, which contains both the
# variational forms and the solver.


# Background
# ----------

# This example demonstrates the solution of a three-dimensional
# elasticity problem. It focuses on how to:

# * Minimise a non-quadratic functional
# * Compute the directional derivative using Automatic Differentiation
# * The built-in Newton solver
# * Use the built-in class ``NonlinearProblem``
# * Interpolate ``python`` functions into ``dolfinx`` functions

# Equation and problem definition
# -------------------------------

# By definition, boundary value problems for hyperelastic media can be
# expressed as minimisation problems on reference or spatial configurations,
# and it is the latter that is adopted in this example. For a domain
# :math:`\Omega \subset \mathbb{R}^{d}`, where :math:`d` denotes the spatial dimension,
# the task is to find the displacement field
# :math:`u: \Omega \rightarrow \mathbb{R}^{d}` that minimises the total potential energy :math:`\Pi`:

# .. math::
#    \min_{u \in V} \Pi,

# where :math:`V` is a suitable function space that satisfies boundary
# conditions on :math:`u`. The total potential energy is given by

# .. math::
#    \Pi = \int_{\Omega} \psi(u) \, {\textrm{d}} x
#    - \int_{\Omega} B \cdot u \, {\textrm{d}} x
#    - \int_{\partial\Omega} T \cdot u \, {\textrm{d}} s,

# where :math:`\psi` is the elastic stored energy density, :math:`bodyForce` is a
# body force (per unit reference volume) and :math:`surfaceTraction` is a traction force
# (per unit reference area).

# At minimum points of :math:`\Pi`, the directional derivative of :math:`\Pi`
# with respect to change in :math:`u`

# .. math::
#     :label: first_variation

# L(u; v) = D_{v} \Pi = \left. \frac{d \Pi(u + \epsilon v)}{d\epsilon}
# \right|_{\epsilon = 0}


# is equal to zero for all :math:`v \in V`:

# .. math::
#    L(u; v) = 0 \quad \forall \ v \in V.

# To minimise the potential energy, a solution to the variational
# equation above is sought. Depending on the potential energy
# :math:`\psi`, :math:`L(u; v)` can be nonlinear in :math:`u`. In such a
# case, the Jacobian of :math:`L` is required in order to solve this
# problem using Newton's method. The Jacobian of :math:`L` is defined as

# .. math::
#    :label: second_variation

# a(u; du, v) = D_{du} L = \left. \frac{d L(u + \epsilon du;
# v)}{d\epsilon} \right|_{\epsilon = 0} .


# Elastic stored energy density
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# To define the elastic stored energy density, consider the deformation
# gradient :math:`F`

# .. math::

#    F = I + \nabla u,

# the right Cauchy-Green tensor :math:`C`

# .. math::

#    C = F^{T} F,

# and the scalar invariants  :math:`J` and :math:`I_{c}`

# .. math::
#    J     &= \det(F), \\
#    I_{c} &= {\textrm{trace}}(C) = F:F.

# This demo considers a common neo-Hookean stored energy model of the form

# .. math::
#    \psi =  \frac{\mu}{2} (I_{c} - 3) - \mu \ln(J) + \frac{\lambda}{2}(J-1)^{2},

# where :math:`\mu` and :math:`\lambda` are the Lame parameters. These
# can be expressed in terms of the more common Young's modulus :math:`E`
# and Poisson ratio :math:`\nu` by:

# .. math::
#     \lambda = \frac{E \nu}{(1 + \nu)(1 - 2\nu)}, \quad  \quad
#     \mu     =  \frac{E}{2(1 + \nu)} .


# Demo parameters
# ^^^^^^^^^^^^^^^

# We consider a unit cube domain:

# * :math:`\Omega = (0, 1) \times (0, 1) \times (0, 1)` (unit cube)

# We use the following definitions of the boundary and boundary conditions:

# * :math:`\Gamma_{D_{0}} = 0 \times (0, 1) \times (0, 1)` (Dirichlet boundary)

# * :math:`\Gamma_{D_{1}} = 1 \times (0, 1) \times (0, 1)` (Dirichlet boundary)

# * :math:`\Gamma_{N} = \partial \Omega \backslash \Gamma_{D}` (Neumann boundary)

# * On :math:`\Gamma_{D_{0}}`:  :math:`u = (0, 0, 0)`

# * On  :math:`\Gamma_{D_{1}}`
#     .. math::
#        u = (&0, \\
#        &(0.5 + (y - 0.5)\cos(\pi/3) - (z - 0.5)\sin(\pi/3) - y)/2, \\
#        &(0.5 + (y - 0.5)\sin(\pi/3) + (z - 0.5)\cos(\pi/3) - z))/2)

# * On :math:`\Gamma_{N}`: :math:`surfaceTraction = (0.1, 0, 0)`

# These are the body forces and material parameters used:

# * :math:`bodyForce = (0, -0.5, 0)`

# * :math:`E    = 10.0`

# * :math:`\nu  = 0.3`


# Implementation
# --------------

# This demo is implemented in the :download:`demo_hyperelasticity.py`
# file.::


import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import BoxMesh, DirichletBC, Function, VectorFunctionSpace, cpp, Constant, NewtonSolver, NonlinearProblem
from dolfinx.cpp.mesh import CellType
from dolfinx.fem import (apply_lifting, assemble_matrix, assemble_vector,
                         locate_dofs_geometrical, set_bc)
from dolfinx.io import XDMFFile
from dolfinx.log import set_log_level, LogLevel, set_output_file
from ufl import (
    Identity,
    TestFunction,
    TrialFunction,
    as_vector,
    dx,
    ds,
    grad,
    inner,
    ln,
    det,
    derivative)

set_log_level(LogLevel.INFO)
set_output_file("logHyperelasticity.txt")

# A subclass of :py:class:`NonlinearProblem <dolfinx.cpp.NonlinearProblem>`,
# which will represent Hyperelasticity in an abstract form to interface with the built-in
# :py:class:`NewtonSolver <dolfinx.cpp.nls.NewtonSolver>`, is now defined. ::


class Hyperelasticity(NonlinearProblem):
    def __init__(self, a, L, bcs):
        super().__init__()
        self.L = L
        self.a = a
        self._F, self._J = None, None
        self._bcs = bcs

    def form(self, x):
        x.ghostUpdate(
            addv=PETSc.InsertMode.INSERT,
            mode=PETSc.ScatterMode.FORWARD)

    def F(self, x):
        if self._F is None:
            self._F = assemble_vector(self.L)

        else:
            with self._F.localForm() as f_local:
                f_local.set(0.0)
            self._F = assemble_vector(self._F, self.L)

        apply_lifting(self._F, [self.a], [self._bcs], [x], -1.0)
        self._F.ghostUpdate(
            addv=PETSc.InsertMode.ADD,
            mode=PETSc.ScatterMode.REVERSE)
        set_bc(self._F, self._bcs, x, -1.0)
        return self._F

    def J(self, x):
        if self._J is None:
            self._J = assemble_matrix(self.a, self._bcs)
        else:
            self._J.zeroEntries()
            self._J = assemble_matrix(self._J, self.a, self._bcs)
        self._J.assemble()
        return self._J

# The constructor (``__init__``) stores references to the bilinear (``a``)
# and linear (``L``) forms, list of boundary conditions (``bcs``). These will used to compute the Jacobian matrix
# and the residual vector, respectively, for use in the Newton solver.  The
# functions ``F`` and ``J`` are virtual member functions of
# :py:class:`NonlinearProblem <dolfinx.cpp.NonlinearProblem>`. The function
# ``F`` computes the residual vector ``b``, and the function ``J``
# computes the Jacobian matrix ``A``.

# A tetrahedral mesh of the domain with 25 ( =
# 24 + 1) vertices in one direction and 17 ( = 16 + 1) vertices in the
# other two directions is created using the built-in function ``BoxMesh``::


mesh = BoxMesh(
    MPI.COMM_WORLD, [np.array([0.0, 0.0, 0.0]),
                     np.array([1.0, 1.0, 1.0])], [24, 16, 16],
    CellType.tetrahedron, cpp.mesh.GhostMode.none)

# boundary tolerance to identify left and right facets
btol = 1.e-6


def left(x):
    return x[0] < btol


def right(x):
    return np.abs(x[0] - 1.) < btol

# function returning the value of non-homogeneous Dirichlet BC on the
# `right` facet


def rightBoundary(x):
    return np.stack((np.zeros(x.shape[1]),
                     (0.5 + (x[1] - 0.5) * np.cos(np.pi / 3) - (x[2] - 0.5) * np.sin(np.pi / 3) - x[1]) / 2.,
                     (0.5 + (x[1] - 0.5) * np.sin(np.pi / 3) + (x[2] - 0.5) * np.cos(np.pi / 3) - x[2]) / 2.))


# Next, various model parameters are defined::
E, nu = Constant(mesh, 10.0), Constant(mesh, 0.3)
mu = E / 2. / (1 + nu)
lmbda = 2 * mu * nu / (1. - 2 * nu)

# To this end, a function space comprising of continuous piecewise
# linear vector polynomials (i.e. a linear Lagrange vector element space)
# is defined::
V = VectorFunctionSpace(mesh, ("Lagrange", 1))

# Note that :py:class:`VectorFunctionSpace <dolfix.function.VectorFunctionSpace>` creates a
# function space of vector fields. The dimension of the vector field
# (the number of components) is assumed to be the same as the spatial
# dimension, unless otherwise specified.

# Trial and test functions along with the approximate displacement
# ``u`` are defined on the finite element space ``V``, and two objects
# of type :py:function:`as_vector <ufl.tensors.as_vector>` are
# declared for the body force (``bodyForce``) and traction
# (``surfaceTraction``) terms::

u = Function(V, name="u")
v = TestFunction(V)
du = TrialFunction(V)
bodyForce = as_vector([0, -0.5, 0])
surfaceTraction = as_vector([0.1, 0, 0])

# With the functions defined, the kinematic quantities involved in the model
# are defined using UFL syntax::

F = Identity(len(u)) + grad(u)    # deformation gradient
I1 = inner(F, F)    # First principal invariant of C = F.T * F
J = det(F)


# Next the strain energy density, the linear form (residual) and the bilinear form (jacobian)
# (see :eq:`first_variation` and :eq:`second_variation`), are computed using Automatic Differentiation,
# again using UFL syntax::

psi = mu / 2. * (I1 - 3) - mu * ln(J) + lmbda / 2. * (J - 1)**2
pi = derivative(psi, u, v) * dx - inner(bodyForce, v) * \
    dx - inner(surfaceTraction, v) * ds
Jac = derivative(pi, u, du)

# Next the function containing the Dirichlet boundary conditions on the right
# facet is interpolated into a ``dolfinx`` function.::

ur = Function(V)
ur.interpolate(rightBoundary)

# and similarly homogeneous boundary conditions are specified on the left
# facet.::

ul = Function(V)
with ul.vector.localForm() as bc_local:
    bc_local.set(0.0)

# Next, the functions for boundary conditions are passed to :py:class:`DirichletBC <dolfinx.fem.DirichletBC>`
# to create corresponding ``DirichletBC`` objects ``bcl`` and ``bcr`` which are then collected in a list ``bcs``.
# These are to be passed to the class ``Hyperelasticity`` which was
# declared in the beginning::

bcl = DirichletBC(ul, locate_dofs_geometrical(V, left))
bcr = DirichletBC(ur, locate_dofs_geometrical(V, right))
bcs = [bcl, bcr]

# Next, corresponding instances of ``Hyperelasticity`` and :py:class:`NewtonSolver <dolfinx.cpp.NewtonSolver>`
# are created ::

problem = Hyperelasticity(Jac, pi, bcs)
solver = NewtonSolver(MPI.COMM_WORLD)

# Next the ``convergence_criterion`` of the ``NewtonSolver`` is set to ``"incremental"``
# which specifies that the Newton solver should compute a norm of the solution increment
# to check for convergence (the other possibility is to use
# ``"residual"``, or to provide a user-defined check). The tolerance for
# convergence is specified by ``rtol``.

solver.convergence_criterion = "incremental"
solver.max_it = 25
solver.rtol = 1.e-8

# The next step is to solve the problem and store the solution vector in
# :py:func:`u.vector<dolfinx.cpp.Function.vector>`.
# This is accomplished by calling :py:func:`solver.solve(problem,
# u.vector)<dolfinx.cpp.NewtonSolver.solve>`::

solver.solve(problem, u.vector)

# Lastly, the solution ``u`` is written to a file for visualization::

with XDMFFile(MPI.COMM_WORLD, "hyperelasticity.xdmf", "w") as wfil:
    wfil.write_mesh(mesh)
    wfil.write_function(u)
