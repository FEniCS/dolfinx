.. Documentation for the mixed Poisson demo from DOLFIN.

.. _demo_pde_mixed-poisson-dual_cpp_documentation:

Dual-mixed formulation for Poisson equation
======================================

.. include:: ../common.txt

Implementation
--------------

The implementation is split in two files, a form file containing the
definition of the variational forms expressed in UFL and the solver
which is implemented in a C++ file.

Running this demo requires the files: :download:`main.cpp`,
:download:`MixedPoissonDual.ufl` and :download:`CMakeLists.txt`.

UFL form file
^^^^^^^^^^^^^

First we define the variational problem in UFL which we save in the
file called :download:`MixedPoissonDual.ufl`.

We begin by defining the finite element spaces. We define two finite
element spaces :math:`\Sigma_h = DRT` and :math:`V_h = CG` separately,
before combining these into a mixed finite element space:

.. code-block:: python

    DRT = FiniteElement("DRT", triangle, 2)
    CG  = FiniteElement("CG", triangle, 3)
    W = DRT * CG

The first argument to :py:class:`FiniteElement` specifies the type of
finite element family, while the third argument specifies the
polynomial degree. The UFL user manual contains a list of all
available finite element families and more details.  The * operator
creates a mixed (product) space ``W`` from the two separate spaces
``DRT`` and ``CG``. Hence,

.. math::

    W = \{ (\tau, v) \ \text{such that} \ \tau \in DRT, v \in CG \}.

Next, we need to specify the trial functions (the unknowns) and the
test functions on this space. This can be done as follows

.. code-block:: python

    (sigma, u) = TrialFunctions(W)
    (tau, v) = TestFunctions(W)

Further, we need to specify the sources :math:`f` and :math:`g`
(coefficients) that will be used in the linear form of the variational
problem. This coefficient needs be defined on a finite element space,
but ``CG`` of polynmial degree 3 is not necessary. We therefore define
a separate finite element space for these coefficients.

.. code-block:: python

    CG1 = FiniteElement("CG", triangle, 1)
    f = Coefficient(CG1)
    g = Coefficient(CG1)

Finally, we define the bilinear and linear forms according to the
equations:

.. code-block:: python

    a = (dot(sigma, tau) + dot(grad(u), tau) + dot(sigma, grad(v)))*dx
    L = - f*v*dx - g*v*ds


C++ program
^^^^^^^^^^^

The solver is implemented in the :download:`main.cpp` file.

At the top we include the DOLFIN header file and the generated header
file containing the variational forms.  For convenience we also
include the DOLFIN namespace.

.. code-block:: c++

   #include <dolfin.h>
   #include "MixedPoissonDual.h"

   using namespace dolfin;

Then follows the definition of the coefficient functions (for
:math:`f` and :math:`g`), which are derived from the DOLFIN
:cpp:class:`Expression` class.

.. code-block:: c++

   // Source term (right-hand side)
   class Source : public Expression
   {
     void eval(Array<double>& values, const Array<double>& x) const
     {
       double dx = x[0] - 0.5;
       double dy = x[1] - 0.5;
       values[0] = 10*exp(-(dx*dx + dy*dy) / 0.02);
     }
   };

   // Boundary source for Neumann boundary condition
   class BoundarySource : public Expression
   {
     void eval(Array<double>& values, const Array<double>& x) const
     { values[0] = sin(5.0*x[0]); }
   };

Then follows the definition of the essential boundary part of the
boundary of the domain, which is derived from the
:cpp:class:`SubDomain` class.

.. code-block:: c++

    // Sub domain for Dirichlet boundary condition
    class DirichletBoundary : public SubDomain
    {
      bool inside(const Array<double>& x, bool on_boundary) const
      { return x[0] < DOLFIN_EPS || x[0] > 1.0 - DOLFIN_EPS; }
    };

Inside the ``main()`` function we first create the ``mesh`` and then
we define the (mixed) function space for the variational
formulation. We also define the bilinear form ``a`` and linear form
``L`` relative to this function space.

.. code-block:: c++

    // Construct function space
    auto W = std::make_shared<MixedPoissonDual::FunctionSpace>(mesh);
    MixedPoissonDual::BilinearForm a(W, W);
    MixedPoissonDual::LinearForm L(W);

Then we create the sources (:math:`f`, :math:`g`) and assign it to the
linear form.

.. code-block:: c++

    // Create sources and assign to L
    Source f;
    BoundarySource g;
    L.f = f;
    L.g = g;

It only remains to prescribe the boundary condition for :math:``u``.
Essential boundary conditions are specified through the class
:cpp:class:`DirichletBC` which takes three arguments: the function
space the boundary condition is supposed to be applied to, the data
for the boundary condition, and the relevant part of the boundary.

We want to apply the boundary condition to the second subspace of the
mixed space.

.. code-block:: c++

    // Define boundary condition
    auto zero = std::make_shared<Constant>(0.0);
    auto boundary = std::make_shared<DirichletBoundary>();
    DirichletBC bc(W->sub(1), zero, boundary);

To compute the solution we use the bilinear and linear forms, and the
boundary condition, but we also need to create a :cpp:class:`Function`
to store the solution(s). The (full) solution will be stored in the
:cpp:class:`Function` ``w``, which we initialise using the
:cpp:class:`FunctionSpace` ``W``. The actual computation is performed
by calling ``solve``.

.. code-block:: c++

    // Compute solution
    Function w(W);
    solve(a == L, w, bc);

Now, the separate components ``sigma`` and ``u`` of the solution can
be extracted by taking components. These can easily be visualized by
calling ``plot``.

.. code-block:: c++

    // Extract sub functions (function views)
    Function& sigma = w[0];
    Function& u = w[1];

    // Plot solutions
    plot(u);
    plot(sigma);


Complete code
-------------

Complete UFL file
^^^^^^^^^^^^^^^^^

.. literalinclude:: MixedPoissonDual.ufl
   :start-after: # Compile
   :language: python

Complete main file
^^^^^^^^^^^^^^^^^^

.. literalinclude:: main.cpp
   :start-after: // Last changed
   :language: c++
