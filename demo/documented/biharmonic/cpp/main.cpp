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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Anders Logg, 2011
//
// First added:  2009-06-26
// Last changed: 2012-11-12
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
//     u(x, y)         = 0
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
    values[0] = 4.0*std::pow(DOLFIN_PI, 4)*
      std::sin(DOLFIN_PI*x[0])*std::sin(DOLFIN_PI*x[1]);
  }

};

// Sub domain for Dirichlet boundary condition
class DirichletBoundary : public SubDomain
{
  bool inside(const Array<double>& x, bool on_boundary) const
  { return on_boundary; }
};

int main()
{
  // Make mesh ghosted for evaluation of DG terms
  parameters["ghost_mode"] = "shared_facet";

  // Create mesh
  auto mesh = std::make_shared<UnitSquareMesh>(32, 32);

  // Create functions
  auto f = std::make_shared<Source>();
  auto alpha = std::make_shared<Constant>(8.0);

  // Create function space
  auto V = std::make_shared<Biharmonic::FunctionSpace>(mesh);

  // Define boundary condition
  auto u0 = std::make_shared<Constant>(0.0);
  auto boundary = std::make_shared<DirichletBoundary>();
  DirichletBC bc(V, u0, boundary);

  // Define variational problem
  Biharmonic::BilinearForm a(V, V);
  Biharmonic::LinearForm L(V);
  a.alpha = alpha; L.f = f;

  // Compute solution
  Function u(V);
  solve(a == L, u, bc);

  // Save solution in VTK format
  File file("biharmonic.pvd");
  file << u;

  // Plot solution
  plot(u);
  interactive();

  return 0;
}
