# The first step is to define the variational problem at hand. We define
# the variational problem in UFL terms in a separate form file
# {download}`demo_mixed_poisson/mixed_poisson.py`.  We begin by defining the
# finite element:

from basix import CellType
from basix.cell import sub_entity_type
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

# Cell type for the mesh
msh_cell = CellType.triangle

# Weakly enforced boundary data will be represented using a function space
# defined over a submesh of the boundary. We get the submesh cell type from
# then mesh cell type.
submsh_cell = sub_entity_type(msh_cell, dim=1, index=0)

# Define finite elements for the problem
RT = element("RT", msh_cell, 1)
P = element("DP", msh_cell, 0)
ME = mixed_element([RT, P])

# Define UFL mesh and submesh
msh = Mesh(element("Lagrange", msh_cell, 1, shape=(2,)))
submsh = Mesh(element("Lagrange", submsh_cell, 1, shape=(2,)))

n = FacetNormal(msh)
V = FunctionSpace(msh, ME)

(sigma, u) = TrialFunctions(V)
(tau, v) = TestFunctions(V)

V0 = FunctionSpace(msh, P)
f = Coefficient(V0)

# We represent boundary data using a first-degree Lagrange space
# defined over a submesh of the boundary
Q = FunctionSpace(submsh, element("Lagrange", submsh_cell, 1))
u0 = Coefficient(Q)

# Specify the weak form of the problem
dx = Measure("dx", msh)
ds = Measure("ds", msh)
a = inner(sigma, tau) * dx + inner(u, div(tau)) * dx + inner(div(sigma), v) * dx
L = -inner(f, v) * dx + inner(u0 * n, tau) * ds(1)
