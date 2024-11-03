# The first step is to define the variational problem at hand. We define
# the variational problem in UFL terms in a separate form file
# {download}`mixed-poisson/poisson.py`.  We begin by defining the finite
# element:

from basix.ufl import element, mixed_element
from ufl import (
    Coefficient,
    FacetNormal,
    FunctionSpace,
    Measure,
    Mesh,
    TestFunctions,
    TrialFunctions,
    div,
    inner,
)

shape = "triangle"
RT = element("RT", shape, 1)
P = element("DP", shape, 0)
ME = mixed_element([RT, P])

msh = Mesh(element("Lagrange", shape, 1, shape=(2,)))
n = FacetNormal(msh)
V = FunctionSpace(msh, ME)

(sigma, u) = TrialFunctions(V)
(tau, v) = TestFunctions(V)

V0 = FunctionSpace(msh, P)
f = Coefficient(V0)
u0 = Coefficient(V0)

dx = Measure("dx", msh)
ds = Measure("ds", msh)
a = inner(sigma, tau) * dx + inner(u, div(tau)) * dx + inner(div(sigma), v) * dx
L = -inner(f, v) * dx + inner(u0 * n, tau) * ds(1)
