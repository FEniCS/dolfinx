import ufl
from basix import CellType, make_quadrature
from basix.ufl import element

cell = CellType.hexahedron
mesh1 = ufl.Mesh(element("Lagrange", CellType.hexahedron, degree=1, shape=(3,)))
mesh2 = ufl.Mesh(element("Lagrange", CellType.hexahedron, degree=2, shape=(3,)))

e = element("Lagrange", CellType.hexahedron, degree=1, shape=(3,))
V1 = ufl.FunctionSpace(mesh1, e)
V2 = ufl.FunctionSpace(mesh2, e)

# Quadrature functions
quadrature_degree = 2
quadrature_rule = "default"
u1, u2 = ufl.Coefficient(V1), ufl.Coefficient(V2)
pts, _ = make_quadrature(CellType.hexahedron, quadrature_degree)
Q6_P1, Q6_P2 = ufl.grad(u1), ufl.grad(u2)
expressions = [(Q6_P1, pts), (Q6_P2, pts)]
