# This demo aims to illustrate how to assemble a matrix with a trial function
# defined on a submesh of co-dimension 0, and a test function defined on the parent mesh
from basix.ufl import element
from ufl import (
    FunctionSpace,
    Mesh,
    TestFunction,
    TrialFunction,
    dx,
)

cell = "quadrilateral"
coord_element = element("Lagrange", cell, 1, shape=(2,))
mesh = Mesh(coord_element)

# We define the function space and test function on the full mesh
e = element("Lagrange", cell, 1)
V = FunctionSpace(mesh, e)
v = TestFunction(V)

# Next we define the sub-mesh
submesh = Mesh(coord_element)
W = FunctionSpace(submesh, e)
p = TrialFunction(W)

# And finally we define a "mass matrix" on the submesh, with the test function
# of the parent mesh. The integration domain is the parent mesh, but we restrict integration
# to all cells marked with subdomain_id=3, which will indicate what cells of our mesh is part
# of the submesh
a_mixed = p * v * dx(domain=mesh, subdomain_id=3)

q = TestFunction(W)
a = p * q * dx(domain=submesh)

forms = [a_mixed, a]
