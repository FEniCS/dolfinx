Poisson equation (C++)
======================

This demo illustrates how to:

* Solve a linear partial differential equation
* Create and apply Dirichlet boundary conditions
* Define Expressions
* Define a FunctionSpace
* Create a SubDomain

The solution for :math:`u` in this demo will look as follows:

.. image:: ../poisson_u.png
                         :scale: 75 %


Equation and problem definition
-------------------------------

The Poisson equation is the canonical elliptic partial differential
equation.  For a domain :math:`\Omega \subset \mathbb{R}^n` with
boundary :math:`\partial \Omega = \Gamma_{D} \cup \Gamma_{N}`, the
Poisson equation with particular boundary conditions reads:

.. math::
   - \nabla^{2} u &= f \quad {\rm in} \ \Omega, \\
     u &= 0 \quad {\rm on} \ \Gamma_{D}, \\
     \nabla u \cdot n &= g \quad {\rm on} \ \Gamma_{N}. \\

Here, :math:`f` and :math:`g` are input data and :math:`n` denotes the
outward directed boundary normal. The most standard variational form
of Poisson equation reads: find :math:`u \in V` such that

.. math::
   a(u, v) = L(v) \quad \forall \ v \in V,

where :math:`V` is a suitable function space and

.. math::
   a(u, v) &= \int_{\Omega} \nabla u \cdot \nabla v \, {\rm d} x, \\
   L(v)    &= \int_{\Omega} f v \, {\rm d} x
   + \int_{\Gamma_{N}} g v \, {\rm d} s.

The expression :math:`a(u, v)` is the bilinear form and :math:`L(v)`
is the linear form. It is assumed that all functions in :math:`V`
satisfy the Dirichlet boundary conditions (:math:`u = 0 \ {\rm on} \
\Gamma_{D}`).

In this demo, we shall consider the following definitions of the input
functions, the domain, and the boundaries:

* :math:`\Omega = [0,1] \times [0,1]` (a unit square)
* :math:`\Gamma_{D} = \{(0, y) \cup (1, y) \subset \partial \Omega\}` (Dirichlet boundary)
* :math:`\Gamma_{N} = \{(x, 0) \cup (x, 1) \subset \partial \Omega\}` (Neumann boundary)
* :math:`g = \sin(5x)` (normal derivative)
* :math:`f = 10\exp(-((x - 0.5)^2 + (y - 0.5)^2) / 0.02)` (source term)


Implementation
--------------

The implementation is split in two files: a form file containing the
definition of the variational forms expressed in UFL and a C++ file
containing the actual solver.

Running this demo requires the files: :download:`main.cpp`,
:download:`Poisson.ufl` and :download:`CMakeLists.txt`.


UFL form file
^^^^^^^^^^^^^

The UFL file is implemented in :download:`Poisson.ufl`, and the
explanation of the UFL file can be found at :doc:`here <Poisson.ufl>`.


C++ program
^^^^^^^^^^^

The main solver is implemented in the :download:`main.cpp` file.

At the top we include the DOLFIN header file and the generated header
file "Poisson.h" containing the variational forms for the Poisson
equation.  For convenience we also include the DOLFIN namespace.

.. code-block:: cpp

   #include <dolfin.h>
   #include "Poisson.h"

   using namespace dolfin;

Then follows the definition of the coefficient functions (for
:math:`f` and :math:`g`), which are derived from the
:cpp:class:`Expression` class in DOLFIN

.. code-block:: cpp

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

The ``DirichletBoundary`` is derived from the :cpp:class:`SubDomain`
class and defines the part of the boundary to which the Dirichlet
boundary condition should be applied.

.. code-block:: cpp

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
built-in mesh provided by the class :cpp:class:`UnitSquareMesh`. In
order to create a mesh consisting of 32 x 32 squares with each square
divided into two triangles, and the finite element space (specified in
the form file) defined relative to this mesh, we do as follows

.. code-block:: cpp

   int main()
   {
     // Create mesh and function space
     auto mesh = std::make_shared<UnitSquareMesh>(32, 32);
     auto V = std::make_shared<Poisson::FunctionSpace>(mesh);

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

.. code-block:: cpp

     // Define boundary condition
     auto u0 = std::make_shared<Constant>(0.0);
     auto boundary = std::make_shared<DirichletBoundary>();
     DirichletBC bc(V, u0, boundary);

Next, we define the variational formulation by initializing the
bilinear and linear forms (:math:`a`, :math:`L`) using the previously
defined :cpp:class:`FunctionSpace` ``V``.  Then we can create the
source and boundary flux term (:math:`f`, :math:`g`) and attach these
to the linear form.

.. code-block:: cpp

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

.. code-block:: cpp

     // Compute solution
     Function u(V);
     solve(a == L, u, bc);

The function ``u`` will be modified during the call to solve. A
:cpp:class:`Function` can be manipulated in various ways, in
particular, it can be plotted and saved to file. Here, we output the
solution to a ``VTK`` file (using the suffix ``.pvd``) for later
visualization and also plot it using the ``plot`` command:

.. code-block:: cpp

     // Save solution in VTK format
     File file("poisson.pvd");
     file << u;

     // Plot solution
     plot(u);
     interactive();

     return 0;
   }
