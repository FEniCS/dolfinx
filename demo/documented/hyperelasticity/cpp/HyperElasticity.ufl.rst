UFL input for hyperleasticity
=============================

The first step is to define the variational problem at hand. We define
the variational problem in UFL terms in a separate form file
:download:`HyperElasticity.ufl`.

We are interested in solving for a discrete vector field in three
dimensions, so first we need the appropriate finite element space and
trial and test functions on this space::

    # Function spaces
    element = VectorElement("Lagrange", tetrahedron, 1)

    # Trial and test functions
    du = TrialFunction(element)     # Incremental displacement
    v  = TestFunction(element)      # Test function

Note that ``VectorElement`` creates a finite element space of vector
fields. The dimension of the vector field (the number of components)
is assumed to be the same as the spatial dimension (in this case 3),
unless otherwise specified.

Next, we will be needing functions for the boundary source ``B``, the
traction ``T`` and the displacement solution itself ``u``::

    # Functions
    u = Coefficient(element)        # Displacement from previous iteration
    B = Coefficient(element)        # Body force per unit volume
    T = Coefficient(element)        # Traction force on the boundary

Now, we can define the kinematic quantities involved in the model::

    # Kinematics
    d = len(u)
    I = Identity(d)                 # Identity tensor
    F = I + grad(u)                 # Deformation gradient
    C = F.T*F                       # Right Cauchy-Green tensor

    # Invariants of deformation tensors
    Ic = tr(C)
    J  = det(F)

Before defining the energy density and thus the total potential
energy, it only remains to specify constants for the elasticity
parameters::

    # Elasticity parameters
    mu    = Constant(tetrahedron)
    lmbda = Constant(tetrahedron)

Both the first variation of the potential energy, and the Jacobian of
the variation, can be automatically computed by a call to
``derivative``::

    # Stored strain energy density (compressible neo-Hookean model)
    psi = (mu/2)*(Ic - 3) - mu*ln(J) + (lmbda/2)*(ln(J))**2

    # Total potential energy
    Pi = psi*dx - inner(B, u)*dx - inner(T, u)*ds

    # First variation of Pi (directional derivative about u in the direction of v)
    F = derivative(Pi, u, v)

    # Compute Jacobian of F
    J = derivative(F, u, du)

Note that ``derivative`` is here used with three arguments: the form
to be differentiated, the variable (function) we are supposed to
differentiate with respect too, and the direction the derivative is
taken in.

Before the form file can be used in the C++ program, it must be
compiled using FFC by running (on the command-line):

.. code-block:: sh

    ffc -l dolfin HyperElasticity.ufl

Note the flag ``-l dolfin`` which tells FFC to generate
DOLFIN-specific wrappers that make it easy to access the generated
code from within DOLFIN.
