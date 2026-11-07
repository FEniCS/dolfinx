"""Biharmonic equation using interior penalty method."""

# We begin by defining the finite element:

from basix.ufl import element
from ufl import (
    CellDiameter,
    Coefficient,
    Constant,
    FacetNormal,
    FunctionSpace,
    Mesh,
    TestFunction,
    TrialFunction,
    avg,
    div,
    dS,
    dx,
    grad,
    inner,
    jump,
)

e = element("Lagrange", "triangle", 2)

# The arguments to {py:func}`element<basix.ufl.element>` are the
# finite element family, the cell type and the polynomial degree. The
# element `e` is therefore the continuous piecewise quadratic Lagrange
# element on triangles.
#
# On a mesh described by a linear coordinate element, we define the
# function space, the trial and test functions ($u$ and $v$) and the
# coefficient $f$:

coord_element = element("Lagrange", "triangle", 1, shape=(2,))
mesh = Mesh(coord_element)

V = FunctionSpace(mesh, e)

u = TrialFunction(V)
v = TestFunction(V)
f = Coefficient(V)

# Next, we define the outward unit normal to cell boundaries and the
# cell diameter. The average diameter of the two cells sharing a facet,
# `h_avg`, is used for $h_E$; `h("+")` and `h("-")` restrict `h` to
# either side of a facet. The penalty parameter `alpha` is a
# {py:class}`Constant<ufl.Constant>`, so it can be changed in the C++
# program without regenerating code.

n = FacetNormal(mesh)
h = CellDiameter(mesh)
h_avg = (h("+") + h("-")) / 2
alpha = Constant(mesh)

# Finally, we define the bilinear and linear forms. Integrals over
# interior facets use the measure `dS`.

# Bilinear form
a = (
    inner(div(grad(u)), div(grad(v))) * dx
    - inner(avg(div(grad(u))), jump(grad(v), n)) * dS
    - inner(jump(grad(u), n), avg(div(grad(v)))) * dS
    + alpha / h_avg * inner(jump(grad(u), n), jump(grad(v), n)) * dS
)

# Linear form
L = inner(f, v) * dx
