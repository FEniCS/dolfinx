// Copyright (C) 2009 Kristian B. Oelgaard
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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2009-06-26
// Last changed: 2010-09-01
//
// This demo program solves the Biharmonic equation,
//
//     - nabla^4 u(x, y) = f(x, y)
//
// on the unit square with source f given by
//
//     f(x, y) = 4 pi^4 sin(pi*x)*sin(pi*y)
//
// and boundary conditions given by
//
//     u(x, y)     = 0
//     nabla^2 u(x, y) = 0
//
// using a discontinuous Galerkin formulation (interior penalty method).

#include <dolfin.h>
#include "Biharmonic.h"

using namespace dolfin;

// Source term
class Source : public Expression
{
public:

  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0] = 4.0*std::pow(DOLFIN_PI, 4)*std::sin(DOLFIN_PI*x[0])*std::sin(DOLFIN_PI*x[1]);
  }

};

// Sub domain for Dirichlet boundary condition
class DirichletBoundary : public SubDomain
{
  bool inside(const Array<double>& x, bool on_boundary) const
  {
    return on_boundary;
  }
};

int main()
{
  // Create mesh
  UnitSquare mesh(32, 32);

  // Create functions
  Source f;
  Constant alpha(8.0);

  // Create function space
  Biharmonic::FunctionSpace V(mesh);

  // Define boundary condition
  Constant u0(0.0);
  DirichletBoundary boundary;
  DirichletBC bc(V, u0, boundary);

  // Define forms and attach functions
  Biharmonic::BilinearForm a(V, V);
  Biharmonic::LinearForm L(V);
  a.alpha = alpha; L.f = f;

  // Create PDE
  VariationalProblem problem(a, L, bc);

  // Solve PDE
  Function u(V);
  problem.solve(u);

  // Plot solution
  plot(u);

  // Save solution in VTK format
  File file("biharmonic.pvd");
  file << u;

  return 0;
}
