# UFL input for the Matrix-free Poisson Demo

from basix.ufl import element
from ufl import (
    Coefficient,
    Constant,
    FunctionSpace,
    Mesh,
    TestFunction,
    TrialFunction,
    action,
    dx,
    grad,
    inner,
)

coord_element = element("Lagrange", "triangle", 1, shape=(2,))
mesh = Mesh(coord_element)

# Function Space
e = element("Lagrange", "triangle", 2)
V = FunctionSpace(mesh, e)

# Trial and test functions
u = TrialFunction(V)
v = TestFunction(V)

# Define a constant RHS
f = Constant(V)

# Define the bilinear and linear forms according to the
# variational formulation of the equations::
a = inner(grad(u), grad(v)) * dx
L = inner(f, v) * dx

# Define linear form representing the action of the form "a" on
# the coefficient "ui"
ui = Coefficient(V)
M = action(a, ui)

# Define form to compute the L2 norm of the error
usol = Coefficient(V)
uexact = Coefficient(V)
E = inner(usol - uexact, usol - uexact) * dx

forms = [M, L, E]
