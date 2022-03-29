from ufl import (Coefficient, Constant, FiniteElement, FunctionSpace, Mesh,
                 TestFunction, TrialFunction, VectorElement, dx, grad, inner,
                 tetrahedron)

element = FiniteElement("Lagrange", tetrahedron, 2)
coord_element = VectorElement("Lagrange", tetrahedron, 1)
mesh = Mesh(coord_element)

V = FunctionSpace(mesh, element)

u = TrialFunction(V)
v = TestFunction(V)
f = Coefficient(V)
kappa = Constant(mesh)

a = kappa * inner(grad(u), grad(v)) * dx
