from ufl import (Coefficient, Constant, FiniteElement, FunctionSpace, Mesh,
                 TestFunction, TrialFunction, VectorElement, dx, grad, inner,
                 tetrahedron)
import basix
from basix.ufl_wrapper import create_vector_element

element = FiniteElement("Lagrange", tetrahedron, 2)
coord_element = create_vector_element("Lagrange", "tetrahedron", 1)
mesh = Mesh(coord_element)

V = FunctionSpace(mesh, element)

u = TrialFunction(V)
v = TestFunction(V)
f = Coefficient(V)
kappa = Constant(mesh)

a = kappa * inner(grad(u), grad(v)) * dx
