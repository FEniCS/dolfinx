UFL input for the Poisson equation
==================================

The first step is to define the variational problem at hand. We define
the variational problem in UFL terms in a separate form file
:download:`Poisson.ufl`.  We begin by defining the finite element::

   element = FiniteElement("Lagrange", triangle, 1)

The first argument to :py:class:`FiniteElement` is the finite element
family, the second argument specifies the domain, while the third
argument specifies the polynomial degree. Thus, in this case, our
element ``element`` consists of first-order, continuous Lagrange basis
functions on triangles (or in order words, continuous piecewise linear
polynomials on triangles).

Next, we use this element to initialize the trial and test functions
(:math:`u` and :math:`v`) and the coefficient functions (:math:`f` and
:math:`g`)::

   u = TrialFunction(element)
   v = TestFunction(element)
   f = Coefficient(element)
   g = Coefficient(element)

Finally, we define the bilinear and linear forms according to the
variational formulation of the equations::

   a = inner(grad(u), grad(v))*dx
   L = f*v*dx + g*v*ds

Before the form file can be used in the C++ program, it must be
compiled using FFC by running (on the command-line):

.. code-block:: sh

   ffc -l dolfin Poisson.ufl
