from ufl import (FunctionSpace, Mesh, VectorConstant,
                 TestFunction, TrialFunction, VectorElement, dx, grad,
                 inner, tetrahedron)


element = VectorElement("Lagrange", tetrahedron, 1)
coord_element = VectorElement("Lagrange", tetrahedron, 1)
mesh = Mesh(coord_element)

V = FunctionSpace(mesh, element)

u = TrialFunction(V)
v = TestFunction(V)

f = VectorConstant(mesh)

a = inner(grad(u), grad(v)) * dx
L = inner(f, v) * dx

forms = [a, L]
