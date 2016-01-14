// Copyright (C) 2006-2007 Garth N. Wells
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Anders Logg, 2005-2011.
//
// First added:  2005
// Last changed: 2012-11-12
//
// This demo illustrates how to use of DOLFIN for solving a nonlinear
// PDE, in this case a nonlinear variant of Poisson's equation,
//
//     - div (1 + u^2) grad u(x, y) = f(x, y)
//
// on the unit square with source f given by
//
//     f(x, y) = x*sin(y)
//
// and boundary conditions given by
//
//     u(x, y)     = 1  for x = 0
//     du/dn(x, y) = 0  otherwise
//
// This is equivalent to solving the variational problem
//
//    F(u) = ((1 + u^2)*grad(u), grad(v)) - (f, v) = 0

#include <dolfin.h>
#include "NonlinearPoisson.h"

using namespace dolfin;

// Right-hand side
class Source : public Expression
{
public:

  Source() : Expression() {}

  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0] = x[0]*sin(x[1]);
  }

};

// Sub domain for Dirichlet boundary condition
class DirichletBoundary : public SubDomain
{
  bool inside(const Array<double>& x, bool on_boundary) const
  {
    return std::abs(x[0] - 1.0) < DOLFIN_EPS && on_boundary;
  }
};

int main()
{
  // Create mesh and define function space
  UnitSquareMesh mesh(16, 16);
  auto V = std::make_shared<NonlinearPoisson::FunctionSpace>(mesh);

  // Define boundary condition
  auto dirichlet_boundary = std::make_shared<DirichletBoundary>();
  auto g = std::make_shared<Constant>(1.0);
  DirichletBC bc(V, g, dirichlet_boundary);

  // Define source and solution functions
  auto f = std::make_shared<Source>();
  auto u = std::make_shared<Function>(V);

  // Create residual form defining (nonlinear) variational problem
  NonlinearPoisson::LinearForm F(V);
  F.u = u;
  F.f = f;

  // Create Jacobian form J = F' (for use in nonlinear solver).
  NonlinearPoisson::JacobianForm J(V, V);
  J.u = u;

  // Create solver Parameters
  Parameters params("nonlinear_variational_solver");
  Parameters newton_params("newton_solver");
  newton_params.add("relative_tolerance", 1e-6);
  params.add(newton_params);

  // Solve nonlinear variational problem
  solve(F == 0, *u, bc, J, params);

  // Save solution in VTK format
  File file("nonlinear_poisson.pvd");
  file << *u;

  // Plot solution
  plot(*u);
  interactive();

  return 0;
}
