# UFL input for the Matrix-free Poisson Demo
# ==================================
from basix.ufl_wrapper import create_vector_element
from ufl import (Coefficient, Constant, FiniteElement, FunctionSpace, Mesh,
                 TestFunction, TrialFunction, action, dx, grad, inner,
                 triangle)

coord_element = create_vector_element("Lagrange", "triangle", 1)
mesh = Mesh(coord_element)

# Function Space
element = FiniteElement("Lagrange", triangle, 2)
V = FunctionSpace(mesh, element)

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
