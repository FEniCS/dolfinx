UFL input for mixed formulation of Poisson Equation
===================================================

First we define the variational problem in UFL which we save in the
file called :download:`MixedPoisson.ufl`.

We begin by defining the finite element spaces. We define two finite
element spaces :math:`\Sigma_h = BDM` and :math:`V_h = DG` separately,
before combining these into a mixed finite element space: ::

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
test functions on this space. This can be done as follows ::

    (sigma, u) = TrialFunctions(W)
    (tau, v) = TestFunctions(W)

Further, we need to specify the source :math:`f` (a coefficient) that
will be used in the linear form of the variational problem. This
coefficient needs be defined on a finite element space, but none of
the above defined elements are quite appropriate. We therefore define
a separate finite element space for this coefficient. ::

    CG = FiniteElement("CG", triangle, 1)
    f = Coefficient(CG)

Finally, we define the bilinear and linear forms according to the equations: ::

    a = (dot(sigma, tau) + div(tau)*u + div(sigma)*v)*dx
    L = - f*v*dx
