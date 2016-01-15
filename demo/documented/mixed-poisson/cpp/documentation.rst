.. Documentation for the mixed Poisson demo from DOLFIN.

.. _demo_pde_mixed-poisson_cpp_documentation:

Mixed formulation for Poisson equation
======================================

.. include:: ../common.txt

Implementation
--------------

The implementation is split in two files, a form file containing the definition
of the variational forms expressed in UFL and the solver which is implemented
in a C++ file.

Running this demo requires the files: :download:`main.cpp`,
:download:`MixedPoisson.ufl` and :download:`CMakeLists.txt`.

UFL form file
^^^^^^^^^^^^^

First we define the variational problem in UFL which we save in the
file called :download:`MixedPoisson.ufl`.

We begin by defining the finite element spaces. We define two finite
element spaces :math:`\Sigma_h = BDM` and :math:`V_h = DG` separately,
before combining these into a mixed finite element space:

.. code-block:: python

    BDM = FiniteElement("BDM", triangle, 1)
    DG  = FiniteElement("DG", triangle, 0)
    W = BDM * DG

The first argument to :py:class:`FiniteElement` specifies the type of
finite element family, while the third argument specifies the
polynomial degree. The UFL user manual contains a list of all
available finite element families and more details.  The * operator
creates a mixed (product) space ``W`` from the two separate spaces
``BDM`` and ``DG``. Hence,

.. math::

    W = \{ (\tau, v) \ \text{such that} \ \tau \in BDM, v \in DG \}.

Next, we need to specify the trial functions (the unknowns) and the
test functions on this space. This can be done as follows

.. code-block:: python

    (sigma, u) = TrialFunctions(W)
    (tau, v) = TestFunctions(W)

Further, we need to specify the source :math:`f` (a coefficient) that
will be used in the linear form of the variational problem. This
coefficient needs be defined on a finite element space, but none of
the above defined elements are quite appropriate. We therefore define
a separate finite element space for this coefficient.

.. code-block:: python

    CG = FiniteElement("CG", triangle, 1)
    f = Coefficient(CG)

Finally, we define the bilinear and linear forms according to the equations:

.. code-block:: python

    a = (dot(sigma, tau) + div(tau)*u + div(sigma)*v)*dx
    L = - f*v*dx


C++ program
^^^^^^^^^^^

The solver is implemented in the :download:`main.cpp` file.

At the top we include the DOLFIN header file and the generated header
file containing the variational forms.  For convenience we also
include the DOLFIN namespace.

.. code-block:: c++

   #include <dolfin.h>
   #include "MixedPoisson.h"

   using namespace dolfin;

Then follows the definition of the coefficient functions (for
:math:`f` and :math:`G`), which are derived from the DOLFIN
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

   // Boundary source for flux boundary condition
   class BoundarySource : public Expression
   {
   public:

     BoundarySource(const Mesh& mesh) : Expression(2), mesh(mesh) {}

     void eval(Array<double>& values, const Array<double>& x,
               const ufc::cell& ufc_cell) const
     {
       dolfin_assert(ufc_cell.local_facet >= 0);

       Cell cell(mesh, ufc_cell.index);
       Point n = cell.normal(ufc_cell.local_facet);

       const double g = sin(5*x[0]);
       values[0] = g*n[0];
       values[1] = g*n[1];
     }

    private:

      const Mesh& mesh;

    };


Then follows the definition of the essential boundary part of the
boundary of the domain, which is derived from the
:cpp:class:`SubDomain` class.

.. code-block:: c++

    // Sub domain for essential boundary condition
    class EssentialBoundary : public SubDomain
    {
      bool inside(const Array<double>& x, bool on_boundary) const
      {
        return x[1] < DOLFIN_EPS or x[1] > 1.0 - DOLFIN_EPS;
      }
    };

Inside the ``main()`` function we first create the ``mesh`` and then
we define the (mixed) function space for the variational
formulation. We also define the bilinear form ``a`` and linear form
``L`` relative to this function space.

.. code-block:: c++

    // Construct function space
    auto W = std::make_shared<MixedPoisson::FunctionSpace>(mesh);
    MixedPoisson::BilinearForm a(W, W);
    MixedPoisson::LinearForm L(W);

Then we create the source (:math:`f`) and assign it to the linear form.

.. code-block:: c++

    // Create source and assign to L
    auto f = std::make_shared<Source>();
    L.f = f;

It only remains to prescribe the boundary condition for the
flux. Essential boundary conditions are specified through the class
:cpp:class:`DirichletBC` which takes three arguments: the function
space the boundary condition is supposed to be applied to, the data
for the boundary condition, and the relevant part of the boundary.

We want to apply the boundary condition to the first subspace of the
mixed space. This space can be accessed through the `sub` member
function of the :cpp:class:`FunctionSpace` class.

Next, we need to construct the data for the boundary condition. An
essential boundary condition is handled by replacing degrees of
freedom by the degrees of freedom evaluated at the given data. The
:math:`BDM` finite element spaces are vector-valued spaces and hence
the degrees of freedom act on vector-valued objects. The effect is
that the user is required to construct a :math:`G` such that :math:`G
\cdot n = g`.  Such a :math:`G` can be constructed by letting :math:`G
= g n`. This is what the derived expression class ``BoundarySource``
defined above does.

.. code-block:: c++

    // Define boundary condition
    auto G = std::make_shared<BoundarySource>(*mesh);
    auto boundary = std::make_shared<EssentialBoundary>();
    DirichletBC bc(W->sub(0), G, boundary);

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

.. literalinclude:: MixedPoisson.ufl
   :start-after: # Compile
   :language: python

Complete main file
^^^^^^^^^^^^^^^^^^

.. literalinclude:: main.cpp
   :start-after: // Last changed
   :language: c++
