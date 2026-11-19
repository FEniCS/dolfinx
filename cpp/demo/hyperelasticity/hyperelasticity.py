"""Hyperelasticity variational forms."""

# We solve for a vector field in three dimensions, so we need a
# vector-valued finite element space and trial and test functions on
# this space:

from basix.ufl import element
from ufl import (
    Coefficient,
    Constant,
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

# The argument `shape=(3,)` makes `element` vector-valued. The same
# element describes the mesh geometry.
#
# Next, we define the displacement `u`, the body force `B` and the
# traction `T`:

# Functions
u = Coefficient(V)  # Displacement from previous iteration
B = Constant(mesh, shape=(3,))  # Body force per unit volume
T = Constant(mesh, shape=(3,))  # Traction force on the boundary

# The kinematic quantities are:

# Kinematics
d = len(u)
I = Identity(d)  # Identity tensor  # noqa: E741
F = variable(I + grad(u))  # Deformation gradient
C = F.T * F  # Right Cauchy-Green tensor

# Invariants of deformation tensors
Ic = tr(C)
J = det(F)

# The Lamé parameters are computed from Young's modulus and Poisson's
# ratio:

# Elasticity parameters
E = 10.0
nu = 0.3
mu = E / (2 * (1 + nu))
lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))

# The residual (the first variation of the potential energy) and its
# Jacobian are computed by {py:func}`derivative<ufl.derivative>`. The
# Cauchy stress is computed by differentiating `psi` with respect to
# `F`, which is why `F` was declared a `variable`:

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
# Evaluate the stress at the reference cell midpoint
expressions = [(sigma, [[0.25, 0.25, 0.25]])]
