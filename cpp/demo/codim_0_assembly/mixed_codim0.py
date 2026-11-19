"""Trial function on submesh of codim 0 and test function on parent mesh."""

# The test function is defined on the parent mesh, and the trial
# function on a submesh of co-dimension 0.

from basix.ufl import element
from ufl import (
    FunctionSpace,
    Mesh,
    TestFunction,
    TrialFunction,
    dx,
    inner,
)

cell = "quadrilateral"
coord_element = element("Lagrange", cell, 1, shape=(2,))
mesh = Mesh(coord_element)

# Define the function space and test function on the parent mesh
e = element("Lagrange", cell, 1)
V = FunctionSpace(mesh, e)
v = TestFunction(V)

# Define the submesh, which has the same cell type, and the trial space
# on it
submesh = Mesh(coord_element)
W = FunctionSpace(submesh, e)
p = TrialFunction(W)

# Define a mass matrix with the trial function on the submesh and the
# test function on the parent mesh. The integration domain is the parent
# mesh, restricted to the cells with `subdomain_id=3`, which the C++
# program sets to the cells of the submesh.
a_mixed = inner(p, v) * dx(domain=mesh, subdomain_id=3)

# Mass matrix on the submesh
q = TestFunction(W)
a = inner(p, q) * dx(domain=submesh)

forms = [a_mixed, a]
