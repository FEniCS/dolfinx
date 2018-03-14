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

# Copyright (C) 2009-2011 Anders Logg
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from dolfin import *

# Read mesh from file and create function space
mesh = Mesh("../unitsquare_32_32.xml.gz")
V = FunctionSpace(mesh, "Lagrange", 1)

# Define Dirichlet boundary (x = 0 or x = 1)
def boundary(x):
    return x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS

# Define boundary condition
u0 = Constant(0.0)
bc = DirichletBC(V, u0, boundary)

# Define conductivity components as MeshFunctions
c00 = MeshFunction("double", mesh, "../unitsquare_32_32_c00.xml.gz")
c01 = MeshFunction("double", mesh, "../unitsquare_32_32_c01.xml.gz")
c11 = MeshFunction("double", mesh, "../unitsquare_32_32_c11.xml.gz")

# Code for C++ evaluation of conductivity
conductivity_code = """

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
namespace py = pybind11;

#include <dolfin/function/Expression.h>
#include <dolfin/mesh/MeshFunction.h>

class Conductivity : public dolfin::Expression
{
public:

  // Create expression with 3 components
  Conductivity() : dolfin::Expression(3) {}

  // Function for evaluating expression on each cell
  void eval(Eigen::Ref<Eigen::VectorXd> values, Eigen::Ref<const Eigen::VectorXd> x, const ufc::cell& cell) const override
  {
    const uint cell_index = cell.index;
    values[0] = (*c00)[cell_index];
    values[1] = (*c01)[cell_index];
    values[2] = (*c11)[cell_index];
  }

  // The data stored in mesh functions
  std::shared_ptr<dolfin::MeshFunction<double>> c00;
  std::shared_ptr<dolfin::MeshFunction<double>> c01;
  std::shared_ptr<dolfin::MeshFunction<double>> c11;

};

PYBIND11_MODULE(SIGNATURE, m)
{
  py::class_<Conductivity, std::shared_ptr<Conductivity>, dolfin::Expression>
    (m, "Conductivity")
    .def(py::init<>())
    .def_readwrite("c00", &Conductivity::c00)
    .def_readwrite("c01", &Conductivity::c01)
    .def_readwrite("c11", &Conductivity::c11);
}

"""

c = CompiledExpression(compile_cpp_code(conductivity_code).Conductivity(),
                       c00=c00, c01=c01, c11=c11, degree=0)

C = as_matrix(((c[0], c[1]), (c[1], c[2])))

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Expression("10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)", degree=2)
a = inner(C*grad(u), grad(v))*dx
L = f*v*dx

# Compute solution
u = Function(V)
solve(a == L, u, bc)

# Save solution in VTK format
file = File("poisson.pvd")
file << u

# Plot solution
import matplotlib.pyplot as plt
plot(u)
plt.show()
