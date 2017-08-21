UFL input for the for the Stokes equations using a mixed formulation (Taylor-Hood quadrilateral elements).
==================================

The first step is to define the variational problem at hand. We define
the variational problem in UFL terms in a separate form file
:download:`Stokes.ufl`.  We begin by defining the mixed
finite element ``TH`` which consists of continuous
piecewise quadratics (Q2) and continuous piecewise
linears (Q1) on quadrilaterals. ::

    Q2 = VectorElement("Q", quadrilateral, 2)
    Q1 = FiniteElement("Q", quadrilateral, 1)
    TH = Q2 * Q1

Next, we use this element to initialize the pairs of trial and test functions
(:math:`(u, p)` and :math:`(v, q)`) and the coefficient function (:math:`f`) ::

    (u, p) = TrialFunctions(TH)
    (v, q) = TestFunctions(TH)
    f = Coefficient(Q2)

Finally, we define the bilinear and linear forms according to the
variational formulation of the Stokes equations ::

   a = (inner(grad(u), grad(v)) - div(v)*p + div(u)*q)*dx
   L = dot(f, v)*dx

Before the form file can be used in the C++ program, it must be
compiled using FFC by running (on the command-line):

.. code-block:: sh

   ffc -l dolfin Stokes.ufl
