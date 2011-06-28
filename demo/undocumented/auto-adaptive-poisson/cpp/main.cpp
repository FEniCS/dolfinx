// Copyright (C) 2010 Anders Logg and Marie E. Rognes
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
// First added:  2010-08-19
// Last changed: 2011-06-28

#include <dolfin.h>
#include "AdaptivePoisson.h"

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

// Normal derivative (Neumann boundary condition)
class dUdN : public Expression
{
  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0] = sin(5*x[0]);
  }
};

// Sub domain for Dirichlet boundary condition
class DirichletBoundary : public SubDomain
{
  bool inside(const Array<double>& x, bool on_boundary) const
  {
    return x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS;
  }
};

int main()
{
  // Create mesh and define function space
  UnitSquare mesh(8, 8);
  AdaptivePoisson::BilinearForm::TrialSpace V(mesh);

  // Define boundary condition
  Constant u0(0.0);
  DirichletBoundary boundary;
  DirichletBC bc(V, u0, boundary);

  // Define variational forms
  AdaptivePoisson::BilinearForm a(V, V);
  AdaptivePoisson::LinearForm L(V);
  Source f;
  dUdN g;
  L.f = f;
  L.g = g;

  // Define variational problem
  Function u(V);
  LinearVariationalProblem problem(a, L, u, bc);

  // Define goal functional (quantity of interest)
  AdaptivePoisson::GoalFunctional M(mesh);

  // Compute solution (adaptively to within accuracy)
  double tol = 1.e-5;
  AdaptiveLinearVariationalSolver solver(problem);
  solver.solve(tol, M);

  // Write a summary
  summary();

  return 0;
}
