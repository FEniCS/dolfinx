# The first step is to define the variational problem at hand. We define
# the variational problem in UFL terms in a separate form file.
# We begin by defining the finite element::

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

# The first argument to :py:class:`FiniteElement` is the finite element
# family, the second argument specifies the domain, while the third
# argument specifies the polynomial degree. Thus, in this case, our
# element ``element`` consists of second-order, continuous Lagrange basis
# functions on triangles (or in order words, continuous piecewise linear
# polynomials on triangles).
#
# Next, we use this element to initialize the trial and test functions
# (:math:`u` and :math:`v`) and the coefficient function :math:`f`::

coord_element = element("Lagrange", "triangle", 1, shape=(2,))
mesh = Mesh(coord_element)

V = FunctionSpace(mesh, e)

u = TrialFunction(V)
v = TestFunction(V)
f = Coefficient(V)

# Next, the outward unit normal to cell boundaries and a measure of the
# cell size are defined. The average size of cells sharing a facet will
# be used (``h_avg``).  The UFL syntax ``('+')`` and ``('-')`` restricts
# a function to the ``('+')`` and ``('-')`` sides of a facet,
# respectively.  The penalty parameter ``alpha`` is made a
# :cpp:class:`Constant` so that it can be changed in the program without
# regenerating the code. ::

# Normal component, mesh size and right-hand side
n = FacetNormal(mesh)
h = CellDiameter(mesh)
h_avg = (h("+") + h("-")) / 2
alpha = Constant(mesh)

# Finally, we define the bilinear and linear forms according to the
# variational formulation of the equations. Integrals over
# internal facets are indicated by ``*dS``. ::

# Bilinear form
a = (
    inner(div(grad(u)), div(grad(v))) * dx
    - inner(avg(div(grad(u))), jump(grad(v), n)) * dS
    - inner(jump(grad(u), n), avg(div(grad(v)))) * dS
    + alpha / h_avg * inner(jump(grad(u), n), jump(grad(v), n)) * dS
)

# Linear form
L = inner(f, v) * dx
