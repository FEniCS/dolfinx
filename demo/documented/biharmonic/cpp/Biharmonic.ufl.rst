UFL input for the Biharmonic equation
=====================================

The finite element space is defined::

   element = FiniteElement("Lagrange", triangle, 2)

On the space ``element``, trial and test functions, and the source
term are defined::

   # Trial and test functions
   u = TrialFunction(element)
   v = TestFunction(element)
   f = Coefficient(element)


Next, the outward unit normal to cell boundaries and a measure of the
cell size are defined. The average size of cells sharing a facet will
be used (``h_avg``).  The UFL syntax ``('+')`` and ``('-')`` restricts
a function to the ``('+')`` and ``('-')`` sides of a facet,
respectively.  The penalty parameter ``alpha`` is made a
:cpp:class:`Constant` so that it can be changed in the program without
regenerating the code. ::

   # Normal component, mesh size and right-hand side
   n  = FacetNormal(triangle)
   h = 2.0*Circumradius(triangle)
   h_avg = (h('+') + h('-'))/2

   # Parameters
   alpha = Constant(triangle)

Finally the bilinear and linear forms are defined. Integrals over
internal facets are indicated by ``*dS``. ::

   # Bilinear form
   a = inner(div(grad(u)), div(grad(v)))*dx \
     - inner(avg(div(grad(u))), jump(grad(v), n))*dS \
     - inner(jump(grad(u), n), avg(div(grad(v))))*dS \
     + alpha/h_avg*inner(jump(grad(u), n), jump(grad(v),n))*dS

   # Linear form
   L = f*v*dx

Before the form file can be used in the C++ program, it must be
compiled using FFC by running (on the command-line):

.. code-block:: sh

   ffc -l dolfin Poisson.ufl
