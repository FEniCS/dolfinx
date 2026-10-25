"""Poisson problem."""

# We begin by defining the finite element:

from basix.ufl import element
from ufl import (
    Coefficient,
    Constant,
    FunctionSpace,
    Mesh,
    TestFunction,
    TrialFunction,
    ds,
    dx,
    grad,
    inner,
)

e = element("Lagrange", "triangle", 1)

# The arguments to {py:func}`element<basix.ufl.element>` are the
# finite element family, the cell type and the polynomial degree. The
# element `e` is therefore the continuous piecewise linear Lagrange
# element on triangles.
#
# The mesh is described by a vector-valued linear Lagrange coordinate
# element. On the mesh we define the function space, the trial and test
# functions ($u$ and $v$), the coefficients ($f$ and $g$) and the
# constant $\kappa$:

coord_element = element("Lagrange", "triangle", 1, shape=(2,))
mesh = Mesh(coord_element)

V = FunctionSpace(mesh, e)

u = TrialFunction(V)
v = TestFunction(V)
f = Coefficient(V)
g = Coefficient(V)
kappa = Constant(mesh)

# Finally, we define the bilinear and linear forms according to the
# variational formulation of the equations:

a = kappa * inner(grad(u), grad(v)) * dx
L = inner(f, v) * dx + inner(g, v) * ds
