"""Mixed Poisson problem with boundary data on a submesh."""

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

# The Dirichlet data $u_0$, which is imposed weakly, is represented
# using a function space on a submesh of the boundary. The submesh cell
# type is the facet type of the mesh cell:

submsh_cell = sub_entity_type(msh_cell, dim=1, index=0)

# The mixed element pairs degree 1 Raviart-Thomas with degree 0
# discontinuous Lagrange:

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

# The boundary data is represented using a degree 1 Lagrange space on
# the submesh:

Q = FunctionSpace(submsh, element("Lagrange", submsh_cell, 1))
u0 = Coefficient(Q)

# The forms are defined with measures on the mesh `msh`, which is the
# integration domain. `ds(1)` restricts the boundary integral to the
# facets marked 1, which the C++ program sets to the facets of
# $\Gamma_D$:

dx = Measure("dx", msh)
ds = Measure("ds", msh)
a = inner(sigma, tau) * dx + inner(u, div(tau)) * dx + inner(div(sigma), v) * dx
L = -inner(f, v) * dx + inner(u0 * n, tau) * ds(1)
