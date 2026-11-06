"""UFL input for the matrix-free Poisson demo."""

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

# Function space
e = element("Lagrange", "triangle", 2)
V = FunctionSpace(mesh, e)

# Trial and test functions
u = TrialFunction(V)
v = TestFunction(V)

# Constant RHS
f = Constant(mesh)

# Bilinear and linear forms
a = inner(grad(u), grad(v)) * dx
L = inner(f, v) * dx

# Linear form representing the action of `a` on the coefficient `ui`
ui = Coefficient(V)
M = action(a, ui)

# Form for the squared L2 norm of the error
usol = Coefficient(V)
uexact = Coefficient(V)
E = inner(usol - uexact, usol - uexact) * dx

forms = [M, L, E]
