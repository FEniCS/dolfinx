// Copyright (C) 2014 Jan Blechta
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
// First added:  2014-01-29
// Last changed: 2014-01-29

#include <dolfin.h>
#include "MixedPoissonDual.h"

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

// Boundary source for Neumann boundary condition
class BoundarySource : public Expression
{
  void eval(Array<double>& values, const Array<double>& x) const
  { values[0] = sin(5.0*x[0]); }
};

// Sub domain for Dirichlet boundary condition
class DirichletBoundary : public SubDomain
{
  bool inside(const Array<double>& x, bool on_boundary) const
  { return x[0] < DOLFIN_EPS || x[0] > 1.0 - DOLFIN_EPS; }
};

int main()
{
  // Create mesh
  UnitSquareMesh mesh(32, 32);

  // Construct function space
  auto W = std::make_shared<MixedPoissonDual::FunctionSpace>(mesh);
  MixedPoissonDual::BilinearForm a(W, W);
  MixedPoissonDual::LinearForm L(W);

  // Create sources and assign to L
  auto f = std::make_shared<Source>();
  auto g = std::make_shared<BoundarySource>();
  L.f = f;
  L.g = g;

  // Define boundary condition
  auto zero = std::make_shared<Constant>(0.0);
  auto boundary = std::make_shared<DirichletBoundary>();
  DirichletBC bc(W->sub(1), zero, boundary);

  // Compute solution
  Function w(W);
  solve(a == L, w, bc);

  // Extract sub functions (function views)
  Function& sigma = w[0];
  Function& u = w[1];

  // Plot solutions
  plot(u);
  plot(sigma);
  interactive();

  return 0;
}
