.. Documentation for the Poisson demo from DOLFIN.

.. _demo_pde_poisson_cpp_documentation:

Poisson equation
================

.. include:: ../common.txt

Implementation
--------------

The implementation is split in two files: a form file containing the
definition of the variational forms expressed in UFL and a C++ file
containing the actual solver.

Running this demo requires the files: :download:`main.cpp`,
:download:`Poisson.ufl` and :download:`CMakeLists.txt`.

UFL form file
^^^^^^^^^^^^^

The first step is to define the variational problem at hand. We define
the variational problem in UFL terms in a separate form file
:download:`Poisson.ufl`.  We begin by defining the finite element:

.. code-block:: python

    element = FiniteElement("Lagrange", triangle, 1)

The first argument to :py:class:`FiniteElement` is the finite element
family, the second argument specifies the domain, while the third
argument specifies the polynomial degree. Thus, in this case, our
element ``element`` consists of first-order, continuous Lagrange basis
functions on triangles (or in order words, continuous piecewise linear
polynomials on triangles).

Next, we use this element to initialize the trial and test functions
(:math:`u` and :math:`v`) and the coefficient functions (:math:`f` and
:math:`g`):

.. code-block:: python

    u = TrialFunction(element)
    v = TestFunction(element)
    f = Coefficient(element)
    g = Coefficient(element)

Finally, we define the bilinear and linear forms according to the
variational formulation of the equations:

.. code-block:: python

    a = inner(grad(u), grad(v))*dx
    L = f*v*dx + g*v*ds

Before the form file can be used in the C++ program, it must be
compiled using FFC by running (on the command-line):

.. code-block:: sh

    ffc -l dolfin Poisson.ufl

Note the flag ``-l dolfin`` which tells FFC to generate
DOLFIN-specific wrappers that make it easy to access the generated
code from within DOLFIN.

C++ program
^^^^^^^^^^^

The main solver is implemented in the :download:`main.cpp` file.

At the top we include the DOLFIN header file and the generated header
file "Poisson.h" containing the variational forms for the Poisson
equation.  For convenience we also include the DOLFIN namespace.

.. code-block:: c++

    #include <dolfin.h>
    #include "Poisson.h"

    using namespace dolfin;

.. index:: Expression

Then follows the definition of the coefficient functions (for
:math:`f` and :math:`g`), which are derived from the
:cpp:class:`Expression` class in DOLFIN.

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

    // Normal derivative (Neumann boundary condition)
    class dUdN : public Expression
    {
      void eval(Array<double>& values, const Array<double>& x) const
      {
        values[0] = sin(5*x[0]);
      }
    };

.. index:: SubDomain

The ``DirichletBoundary`` is derived from the :cpp:class:`SubDomain`
class and defines the part of the boundary to which the Dirichlet
boundary condition should be applied.

.. code-block:: c++

    // Sub domain for Dirichlet boundary condition
    class DirichletBoundary : public SubDomain
    {
      bool inside(const Array<double>& x, bool on_boundary) const
      {
        return x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS;
      }
    };

Inside the ``main`` function, we begin by defining a mesh of the
domain. As the unit square is a very standard domain, we can use a
built-in mesh provided by the class :cpp:class:`UnitSquareMesh`. In order
to create a mesh consisting of 32 x 32 squares with each square
divided into two triangles, and the finite element space (specified in
the form file) defined relative to this mesh, we do as follows

.. code-block:: c++

    // Create mesh and function space
    auto mesh = std::make_shared<UnitSquareMesh>(32, 32);
    auto V = std::make_shared<Poisson::FunctionSpace>(mesh);

.. index:: DirichletBC

Now, the Dirichlet boundary condition (:math:`u = 0`) can be created
using the class :cpp:class:`DirichletBC`. A :cpp:class:`DirichletBC`
takes three arguments: the function space the boundary condition
applies to, the value of the boundary condition, and the part of the
boundary on which the condition applies. In our example, the function
space is ``V``, the value of the boundary condition (0.0) can
represented using a :cpp:class:`Constant`, and the Dirichlet boundary
is defined by the class :cpp:class:`DirichletBoundary` listed
above. The definition of the Dirichlet boundary condition then looks
as follows:

.. code-block:: c++

    // Define boundary condition
    auto u0 = std::make_shared<Constant>(0.0);
    auto boundary = std::make_shared<DirichletBoundary>();
    DirichletBC bc(V, u0, boundary);

.. index::
    triple: forms; attach; expression

Next, we define the variational formulation by initializing the
bilinear and linear forms (:math:`a`, :math:`L`) using the previously
defined :cpp:class:`FunctionSpace` ``V``.  Then we can create the
source and boundary flux term (:math:`f`, :math:`g`) and attach these
to the linear form.

.. code-block:: c++

    // Define variational forms
    Poisson::BilinearForm a(V, V);
    Poisson::LinearForm L(V);
    auto f = std::make_shared<Source>();
    auto g = std::make_shared<dUdN>();
    L.f = f;
    L.g = g;

Now, we have specified the variational forms and can consider the
solution of the variational problem. First, we need to define a
:cpp:class:`Function` ``u`` to store the solution. (Upon
initialization, it is simply set to the zero function.) Next, we can
call the ``solve`` function with the arguments ``a == L``, ``u`` and
``bc`` as follows:

.. code-block:: c++

    // Compute solution
    Function u(V);
    solve(a == L, u, bc);

The function ``u`` will be modified during the call to solve. A
:cpp:class:`Function` can be manipulated in various ways, in
particular, it can be plotted and saved to file. Here, we output the
solution to a ``VTK`` file (using the suffix ``.pvd``) for later
visualization and also plot it using the ``plot`` command:

.. code-block:: c++

    // Save solution in VTK format
    File file("poisson.pvd");
    file << u;

    // Plot solution
    plot(u);


Complete code
-------------

Complete UFL file
^^^^^^^^^^^^^^^^^

.. literalinclude:: Poisson.ufl
   :start-after: # Compile this form
   :language: python

Complete main file
^^^^^^^^^^^^^^^^^^

.. literalinclude:: main.cpp
   :start-after: // du/dn
   :language: c++
