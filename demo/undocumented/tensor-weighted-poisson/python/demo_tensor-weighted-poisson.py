"""This demo program solves Poisson's equation

    - div C grad u(x, y) = f(x, y)

on the unit square with source f given by

    f(x, y) = 10*exp(-((x - 0.5)^2 + (y - 0.5)^2) / 0.02)

and boundary conditions given by

    u(x, y) = 0  for x = 0 or x = 1
du/dn(x, y) = 0  for y = 0 or y = 1

The conductivity C is a symmetric 2 x 2 matrix which
varies throughout the domain. In the left part of the
domain, the conductivity is

    C = ((1, 0.3), (0.3, 2))

and in the right part it is

    C = ((3, 0.5), (0.5, 4))

The data files where these values are stored are generated
by the program generate_data.py

This demo is dedicated to BF and Marius... ;-)
"""

__author__ = "Anders Logg (logg@simula.no)"
__date__ = "2009-12-16"
__copyright__ = "Copyright (C) 2009 Anders Logg"
__license__  = "GNU LGPL Version 2.1"

# Last changed: 2009-12-15

from dolfin import *

# Read mesh from file and create function space
mesh = Mesh("mesh.xml.gz")
V = FunctionSpace(mesh, "CG", 1)

# Define Dirichlet boundary (x = 0 or x = 1)
def boundary(x):
    return x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS

# Define boundary condition
u0 = Constant(0.0)
bc = DirichletBC(V, u0, boundary)

# Code for C++ evaluation of conductivity
conductivity_code = """

class Conductivity : public Expression
{
public:

  // Create expression with 3 components
  Conductivity() : Expression(3) {}

  // Function for evaluating expression on each cell
  //void eval(Array<double>& values, const Data& data) const
  void eval(Array<double>& values, const Array<double>& x,  const ufc::cell& cell) const
  {
    const uint D = cell.topological_dimension;
    const uint cell_index = cell.entity_indices[D][0];
    values[0] = (*c00)[cell_index];
    values[1] = (*c01)[cell_index];
    values[2] = (*c11)[cell_index];
  }

  // The data stored in mesh functions
  boost::shared_ptr<MeshFunction<double> > c00;
  boost::shared_ptr<MeshFunction<double> > c01;
  boost::shared_ptr<MeshFunction<double> > c11;

};
"""

# Define conductivity expression and matrix
c00 = MeshFunction("double", mesh, "c00.xml.gz")
c01 = MeshFunction("double", mesh, "c01.xml.gz")
c11 = MeshFunction("double", mesh, "c11.xml.gz")
c = Expression(cppcode=conductivity_code)
c.c00 = c00
c.c01 = c01
c.c11 = c11
C = as_matrix(((c[0], c[1]), (c[1], c[2])))

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Expression("10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)")
a = inner(C*grad(u), grad(v))*dx
L = f*v*dx

# Compute solution
problem = VariationalProblem(a, L, bc)
u = problem.solve()

# Save solution in VTK format
file = File("poisson.pvd")
file << u

# Plot solution
plot(u, interactive=True)
