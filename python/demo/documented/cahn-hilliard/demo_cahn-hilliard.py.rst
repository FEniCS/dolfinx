Cahn-Hilliard equation
======================

This demo is implemented in a single Python file,
:download:`demo_cahn-hilliard.py`, which contains both the variational
forms and the solver.

This example demonstrates the solution of a particular nonlinear
time-dependent fourth-order equation, known as the Cahn-Hilliard
equation. In particular it demonstrates the use of

* The built-in Newton solver
* Advanced use of the base class ``NonlinearProblem``
* Automatic linearisation
* A mixed finite element method
* The :math:`\theta`-method for time-dependent equations
* User-defined Expressions as Python classes
* Form compiler options
* Interpolation of functions


Equation and problem definition
-------------------------------

The Cahn-Hilliard equation is a parabolic equation and is typically used
to model phase separation in binary mixtures.  It involves first-order
time derivatives, and second- and fourth-order spatial derivatives.  The
equation reads:

.. math::
   \frac{\partial c}{\partial t} - \nabla \cdot M \left(\nabla\left(\frac{d f}{d c}
             - \lambda \nabla^{2}c\right)\right) &= 0 \quad {\rm in} \ \Omega, \\
   M\left(\nabla\left(\frac{d f}{d c} - \lambda \nabla^{2}c\right)\right) \cdot n &= 0 \quad {\rm on} \ \partial\Omega, \\
   M \lambda \nabla c \cdot n &= 0 \quad {\rm on} \ \partial\Omega.

where :math:`c` is the unknown field, the function :math:`f` is usually
non-convex in :math:`c` (a fourth-order polynomial is commonly used),
:math:`n` is the outward directed boundary normal, and :math:`M` is a
scalar parameter.


Mixed form
^^^^^^^^^^

The Cahn-Hilliard equation is a fourth-order equation, so casting it in
a weak form would result in the presence of second-order spatial
derivatives, and the problem could not be solved using a standard
Lagrange finite element basis.  A solution is to rephrase the problem as
two coupled second-order equations:

.. math::
   \frac{\partial c}{\partial t} - \nabla \cdot M \nabla\mu  &= 0 \quad {\rm in} \ \Omega, \\
   \mu -  \frac{d f}{d c} + \lambda \nabla^{2}c &= 0 \quad {\rm in} \ \Omega.

The unknown fields are now :math:`c` and :math:`\mu`. The weak
(variational) form of the problem reads: find :math:`(c, \mu) \in V
\times V` such that

.. math::
   \int_{\Omega} \frac{\partial c}{\partial t} q \, {\rm d} x + \int_{\Omega} M \nabla\mu \cdot \nabla q \, {\rm d} x
          &= 0 \quad \forall \ q \in V,  \\
   \int_{\Omega} \mu v \, {\rm d} x - \int_{\Omega} \frac{d f}{d c} v \, {\rm d} x - \int_{\Omega} \lambda \nabla c \cdot \nabla v \, {\rm d} x
          &= 0 \quad \forall \ v \in V.


Time discretisation
^^^^^^^^^^^^^^^^^^^

Before being able to solve this problem, the time derivative must be
dealt with. Apply the :math:`\theta`-method to the mixed weak form of
the equation:

.. math::

   \int_{\Omega} \frac{c_{n+1} - c_{n}}{dt} q \, {\rm d} x + \int_{\Omega} M \nabla \mu_{n+\theta} \cdot \nabla q \, {\rm d} x
          &= 0 \quad \forall \ q \in V  \\
   \int_{\Omega} \mu_{n+1} v  \, {\rm d} x - \int_{\Omega} \frac{d f_{n+1}}{d c} v  \, {\rm d} x - \int_{\Omega} \lambda \nabla c_{n+1} \cdot \nabla v \, {\rm d} x
          &= 0 \quad \forall \ v \in V

where :math:`dt = t_{n+1} - t_{n}` and :math:`\mu_{n+\theta} =
(1-\theta) \mu_{n} + \theta \mu_{n+1}`.  The task is: given
:math:`c_{n}` and :math:`\mu_{n}`, solve the above equation to find
:math:`c_{n+1}` and :math:`\mu_{n+1}`.


Demo parameters
^^^^^^^^^^^^^^^

The following domains, functions and time stepping parameters are used
in this demo:

* :math:`\Omega = (0, 1) \times (0, 1)` (unit square)
* :math:`f = 100 c^{2} (1-c)^{2}`
* :math:`\lambda = 1 \times 10^{-2}`
* :math:`M = 1`
* :math:`dt = 5 \times 10^{-6}`
* :math:`\theta = 0.5`


Implementation
--------------

This demo is implemented in the :download:`demo_cahn-hilliard.py`
file.

First, the modules :py:mod:`random` :py:mod:`matplotlib`
:py:mod:`dolfin` module are imported::

    import random
    from dolfin import *
    from dolfin import function
    from dolfin.io import XDMFFile
    from dolfin.fem.assemble import assemble

.. index::
   single: NonlinearProblem; (in Cahn-Hilliard demo)

A class which will represent the Cahn-Hilliard in an abstract from for
use in the Newton solver is now defined. It is a subclass of
:py:class:`NonlinearProblem <dolfin.cpp.NonlinearProblem>`. ::

    # Class for interfacing with the Newton solver
    class CahnHilliardEquation(NonlinearProblem):
        def __init__(self, a, L):
            NonlinearProblem.__init__(self)
            self.L = L
            self.a = a
            self._F = None
            self._J = None

        def form(self, x):
            x.update_ghosts()

        def F(self, x):
            if self._F is None:
                self._F = assemble(self.L)
            else:
                self._F = assemble(self._F, self.L)
            return self._F

        def J(self, x):
            if self._J is None:
                self._J = assemble(self.a)
            else:
                self._J = assemble(self._J, self.a)
            return self._J

The constructor (``__init__``) stores references to the bilinear (``a``)
and linear (``L``) forms. These will used to compute the Jacobian matrix
and the residual vector, respectively, for use in a Newton solver.  The
function ``F`` and ``J`` are virtual member functions of
:py:class:`NonlinearProblem <dolfin.cpp.NonlinearProblem>`. The function
``F`` computes the residual vector ``b``, and the function ``J``
computes the Jacobian matrix ``A``.

Next, various model parameters are defined::

    # Model parameters
    lmbda  = 1.0e-02  # surface parameter
    dt     = 5.0e-06  # time step
    theta  = 0.5      # time stepping family, e.g. theta=1 -> backward Euler, theta=0.5 -> Crank-Nicolson

A unit square mesh with 97 (= 96 + 1) vertices in each direction is
created, and on this mesh a
:py:class:`FunctionSpace<dolfin.functions.functionspace.FunctionSpace>`
``ME`` is built using a pair of linear Lagrangian elements. ::

    # Create mesh and build function space
    mesh = UnitSquareMesh(MPI.comm_world, 96, 96, CellType.Type.triangle)
    P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    ME = FunctionSpace(mesh, P1*P1)

Trial and test functions of the space ``ME`` are now defined::

    # Define trial and test functions
    du    = TrialFunction(ME)
    q, v  = TestFunctions(ME)

.. index:: split functions

For the test functions,
:py:func:`TestFunctions<dolfin.functions.function.TestFunctions>` (note
the 's' at the end) is used to define the scalar test functions ``q``
and ``v``. The
:py:class:`TrialFunction<dolfin.functions.function.TrialFunction>`
``du`` has dimension two. Some mixed objects of the
:py:class:`Function<dolfin.functions.function.Function>` class on ``ME``
are defined to represent :math:`u = (c_{n+1}, \mu_{n+1})` and :math:`u0
= (c_{n}, \mu_{n})`, and these are then split into sub-functions::

    # Define functions
    u   = Function(ME)  # current solution
    u0  = Function(ME)  # solution from previous converged step

    # Split mixed functions
    dc, dmu = split(du)
    c,  mu  = split(u)
    c0, mu0 = split(u0)

The line ``c, mu = split(u)`` permits direct access to the components of
a mixed function. Note that ``c`` and ``mu`` are references for
components of ``u``, and not copies.

.. index::
   single: interpolating functions; (in Cahn-Hilliard demo)

Initial conditions are created by using the evaluate method
then interpolated into a finite element space::

    @function.expression.numba_eval
    def init_cond(values, x, cell):
        values[:, 0] = 0.63 + 0.02*(0.5 - random.random())
        values[:, 1] = 0.0

    # Create intial conditions and interpolate
    u_init = Expression(init_cond, shape=(2,))
    u.interpolate(u_init)

The first line creates an object of type ``InitialConditions``.  The
following two lines make ``u`` and ``u0`` interpolants of ``u_init``
(since ``u`` and ``u0`` are finite element functions, they may not be
able to represent a given function exactly, but the function can be
approximated by interpolating it in a finite element space).

.. index:: automatic differentiation

The chemical potential :math:`df/dc` is computed using automated
differentiation::

    # Compute the chemical potential df/dc
    c = variable(c)
    f    = 100*c**2*(1-c)**2
    dfdc = diff(f, c)

The first line declares that ``c`` is a variable that some function can
be differentiated with respect to. The next line is the function
:math:`f` defined in the problem statement, and the third line performs
the differentiation of ``f`` with respect to the variable ``c``.

It is convenient to introduce an expression for :math:`\mu_{n+\theta}`::

    # mu_(n+theta)
    mu_mid = (1.0-theta)*mu0 + theta*mu

which is then used in the definition of the variational forms::

    # Weak statement of the equations
    L0 = inner(c, q)*dx - inner(c0, q)*dx + dt*inner(grad(mu_mid), grad(q))*dx
    L1 = inner(mu, v)*dx - inner(dfdc, v)*dx - lmbda*inner(grad(c), grad(v))*dx
    L = L0 + L1

This is a statement of the time-discrete equations presented as part of
the problem statement, using UFL syntax. The linear forms for the two
equations can be summed into one form ``L``, and then the directional
derivative of ``L`` can be computed to form the bilinear form which
represents the Jacobian matrix::

    # Compute directional derivative about u in the direction of du (Jacobian)
    a = derivative(L, u, du)

.. index::
   single: Newton solver; (in Cahn-Hilliard demo)

The DOLFIN Newton solver requires a
:py:class:`NonlinearProblem<dolfin.cpp.NonlinearProblem>` object to
solve a system of nonlinear equations. Here, we are using the class
``CahnHilliardEquation``, which was declared at the beginning of the
file, and which is a sub-class of
:py:class:`NonlinearProblem<dolfin.cpp.NonlinearProblem>`. We need to
instantiate objects of both ``CahnHilliardEquation`` and
:py:class:`NewtonSolver <dolfin.cpp.NewtonSolver>`::

    # Create nonlinear problem and Newton solver
    problem = CahnHilliardEquation(a, L)
    solver = NewtonSolver(MPI.comm_world)
    # solver.parameters["linear_solver"] = "lu"
    solver.parameters["convergence_criterion"] = "incremental"
    solver.parameters["relative_tolerance"] = 1e-6

The string ``"lu"`` passed to the Newton solver indicated that an LU
solver should be used.  The setting of
``parameters["convergence_criterion"] = "incremental"`` specifies that
the Newton solver should compute a norm of the solution increment to
check for convergence (the other possibility is to use ``"residual"``,
or to provide a user-defined check). The tolerance for convergence is
specified by ``parameters["relative_tolerance"] = 1e-6``.

To run the solver and save the output to a VTK file for later
visualization, the solver is advanced in time from :math:`t_{n}` to
:math:`t_{n+1}` until a terminal time :math:`T` is reached::

    # Output file
    file = XDMFFile(MPI.comm_world, "output.xdmf")

    # Step in time
    t = 0.0
    T = 50*dt
    u.vector().vec().copy(result=u0.vector().vec())
    u0.vector().update_ghosts()
    while (t < T):
        t += dt
        solver.solve(problem, u.vector())
        u.vector().vec().copy(result=u0.vector().vec())
        file.write(u.sub(0), t)

The string ``"compressed"`` indicates that the output data should be
compressed to reduce the file size. Within the time stepping loop, the
solution vector associated with ``u`` is copied to ``u0`` at the
beginning of each time step, and the nonlinear problem is solved by
calling
:py:func:`solver.solve(problem,u.vector())<dolfin.cpp.NewtonSolver.solve>`,
with the new solution vector returned in
:py:func:`u.vector()<dolfin.cpp.Function.vector>`. The ``c`` component
of the solution (the first component of ``u``) is then written to file
at every time step.
