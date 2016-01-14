// Copyright (C) 2010 Marie E. Rognes
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
// Modified by Johannes Ring, 2011
//
// First added:  2010
// Last changed: 2012-11-12
//
// This demo program illustrates how to solve Poisson's equation
//
//     - div grad u(x, y) = f(x, y)
//
// on the unit square with pure Neumann boundary conditions:
//
//     du/dn(x, y) = -sin(5*x)
//
// and source f given by
//
//     f(x, y) = 10*exp(-((x - 0.5)^2 + (y - 0.5)^2) / 0.02)
//
// Since only Neumann conditions are applied, u is only determined up to
// a constant c by the above equations. An addition constraint is thus
// required, for instance
//
//   \int u = 0
//
// This can be accomplished by introducing the constant c as an
// additional unknown (to be sought in the space of real numbers)
// and the above constraint.

#include <dolfin.h>
#include "Poisson.h"

using namespace dolfin;

// Source term (right-hand side)
class Source : public Expression
{
  void eval(Array<double>& values, const Array<double>& x) const
  {
    double dx = x[0] - 0.5;
    double dy = x[1] - 0.5;
    values[0] = 10*exp(-(dx*dx + dy*dy) / 0.02);
  }
};

// Boundary flux (Neumann boundary condition)
class Flux : public Expression
{
  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0] = -sin(5*x[0]);
  }
};

int main()
{
  // Create mesh and function space
  auto mesh = std::make_shared<UnitSquareMesh>(64, 64);
  auto V = std::make_shared<Poisson::FunctionSpace>(mesh);

  // Define variational problem
  auto a = std::make_shared<Poisson::BilinearForm>(V, V);
  auto L = std::make_shared<Poisson::LinearForm>(V);
  auto f = std::make_shared<Source>();
  auto g = std::make_shared<Flux>();
  L->f = f;
  L->g = g;
  auto w = std::make_shared<Function>(V);
  std::vector<std::shared_ptr<const DirichletBC>> bcs;
  auto problem = std::make_shared<LinearVariationalProblem>(a, L, w, bcs);

  // Compute solution
  LinearVariationalSolver solver(problem);
  solver.solve();

  // Extract subfunction
  Function u = (*w)[0];

  // Plot solution
  plot(u);
  interactive();

  return 0;
}
