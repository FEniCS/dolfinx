UFL input for the auto adaptive Poisson problem
===============================================

UFL code::

  element = FiniteElement("CG", triangle, 1)
  u = TrialFunction(element)
  v = TestFunction(element)

  f = Coefficient(element)
  g = Coefficient(element)

  a = dot(grad(u), grad(v))*dx()
  L = f*v*dx() + g*v*ds()
  M = u*dx()

Before the form file can be used in the C++ program, it must be
compiled using FFC by running (on the command-line):

.. code-block:: sh

   ffc -l dolfin AdaptivePoisson.ufl
