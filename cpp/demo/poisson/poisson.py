# UFL input for the Poisson equation
# ==================================
#
# The first step is to define the variational problem at hand. We define
# the variational problem in UFL terms in a separate form file
# :download:`poisson.py`.  We begin by defining the finite element::
from basix.ufl_wrapper import create_vector_element
from ufl import (Coefficient, Constant, FiniteElement, FunctionSpace, Mesh,
                 TestFunction, TrialFunction, ds, dx, grad, inner, triangle)

element = FiniteElement("Lagrange", triangle, 1)

# The first argument to :py:class:`FiniteElement` is the finite element
# family, the second argument specifies the domain, while the third
# argument specifies the polynomial degree. Thus, in this case, our
# element ``element`` consists of first-order, continuous Lagrange basis
# functions on triangles (or in order words, continuous piecewise linear
# polynomials on triangles).
#
# Next, we use this element to initialize the trial and test functions
# (:math:`u` and :math:`v`) and the coefficient functions (:math:`f` and
# :math:`g`)::

coord_element = create_vector_element("Lagrange", "triangle", 1)
mesh = Mesh(coord_element)

V = FunctionSpace(mesh, element)

u = TrialFunction(V)
v = TestFunction(V)
f = Coefficient(V)
g = Coefficient(V)
kappa = Constant(mesh)

# Finally, we define the bilinear and linear forms according to the
# variational formulation of the equations::

a = kappa * inner(grad(u), grad(v)) * dx
L = inner(f, v) * dx + inner(g, v) * ds
