# The first step is to define the variational problem at hand.
#
# We are interested in solving for a discrete vector field in three
# dimensions, so first we need the appropriate finite element space and
# trial and test functions on this space:

from basix.ufl import element
from ufl import (
    Coefficient,
    FunctionSpace,
    Identity,
    Mesh,
    TestFunction,
    TrialFunction,
    derivative,
    det,
    diff,
    ds,
    dx,
    grad,
    inner,
    ln,
    tr,
    variable,
)

# Function spaces
e = element("Lagrange", "tetrahedron", 1, shape=(3,))
mesh = Mesh(e)
V = FunctionSpace(mesh, e)

# Trial and test functions
du = TrialFunction(V)  # Incremental displacement
v = TestFunction(V)  # Test function

# Note that `element` with `shape=(3,)` creates a finite element space
# of vector fields.
#
# Next, we will be needing functions for the boundary source `B`, the
# traction `T` and the displacement solution itself `u`:

# Functions
u = Coefficient(V)  # Displacement from previous iteration
B = Coefficient(element)  # Body force per unit volume
T = Coefficient(element)  # Traction force on the boundary

# Now, we can define the kinematic quantities involved in the model:

# Kinematics
d = len(u)
I = Identity(d)  # Identity tensor  # noqa: E741
F = variable(I + grad(u))  # Deformation gradient
C = F.T * F  # Right Cauchy-Green tensor

# Invariants of deformation tensors
Ic = tr(C)
J = det(F)

# Before defining the energy density and thus the total potential
# energy, it only remains to specify constants for the elasticity
# parameters:

# Elasticity parameters
E = 10.0
nu = 0.3
mu = E / (2 * (1 + nu))
lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))

# Both the first variation of the potential energy, and the Jacobian of
# the variation, can be automatically computed by a call to
# `derivative`:

# Stored strain energy density (compressible neo-Hookean model)
psi = (mu / 2) * (Ic - 3) - mu * ln(J) + (lmbda / 2) * (ln(J)) ** 2

# Total potential energy
Pi = psi * dx - inner(B, u) * dx - inner(T, u) * ds

# First variation of Pi (directional derivative about u in the direction
# of v)
F_form = derivative(Pi, u, v)

# Compute Jacobian of F
J_form = derivative(F_form, u, du)

# Compute Cauchy stress
sigma = (1 / J) * diff(psi, F) * F.T

forms = [F_form, J_form]
elements = [e]
expressions = [(sigma, [[0.25, 0.25, 0.25]])]
