// Copyright (C) 2006-2007 Anders Logg
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
// Modified by Garth N. Wells, 2008.
//
// First added:  2006-02-07
// Last changed: 2012-11-12
//
// This demo program solves Poisson's equation,
//
//     - div grad u(x, y) = f(x, y)
//
// on the unit square with source f given by
//
//     f(x, y) = 500*exp(-((x - 0.5)^2 + (y - 0.5)^2)/0.02)
//
// and boundary conditions given by
//
//     u(x, y)     = 0               for x = 0,
//     du/dn(x, y) = 25 cos(5 pi y)  for x = 1,
//     du/dn(x, y) = 0               otherwise.

#include <dolfin.h>
#include "Poisson.h"

using namespace dolfin;

int main()
{
  // Source term
  class Source : public Expression
  {
  public:

    Source() : Expression() {}

    void eval(Array<double>& values, const Array<double>& x) const
    {
      double dx = x[0] - 0.5;
      double dy = x[1] - 0.5;
      values[0] = 500.0*exp(-(dx*dx + dy*dy)/0.02);
    }

  };

  // Neumann boundary condition
  class Flux : public Expression
  {
  public:

    Flux() : Expression() {}

    void eval(Array<double>& values, const Array<double>& x) const
    {
      if (x[0] > (1.0 - DOLFIN_EPS))
        values[0] = 25.0*cos(5.0*DOLFIN_PI*x[1]);
      else
        values[0] = 0.0;
    }

  };

  // Sub domain for Dirichlet boundary condition
  class DirichletBoundary : public SubDomain
  {
    bool inside(const Array<double>& x, bool on_boundary) const
    {
      return x[0] < DOLFIN_EPS;
    }
  };

  // Create mesh
  UnitSquareMesh mesh(300, 300);

  // Create functions
  auto f = std::make_shared<Source>();
  auto g = std::make_shared<Flux>();

  // Define forms and attach functions
  auto V = std::make_shared<Poisson::FunctionSpace>(mesh);
  Poisson::BilinearForm a(V, V);
  Poisson::LinearForm L(V);
  L.f = f; L.g = g;

  // Create boundary condition
  auto u0 = std::make_shared<Constant>(0.0);
  auto boundary = std::make_shared<DirichletBoundary>();
  auto bc = std::make_shared<DirichletBC>(V, u0, boundary);

  // Create function
  Function u(V);

  // Create table
  Table table("Assembly and application of bcs");

  // Matrix and vector to assemble
  Matrix A;
  Vector b;

  // Assemble A and b separately
  tic();
  assemble(A, a);
  assemble(b, L);
  bc->apply(A, b);
  table("Standard", "Assembly time") = toc();

  // Assemble A and b together
  tic();
  assemble_system(A, b, a, L, {bc});
  table("Symmetric", "Assembly time") = toc();

  // Display summary
  info(table);

  // Solve system
  LUSolver solver;
  solver.solve(A, *u.vector(), b);

  // Save solution in VTK format
  File file("poisson.pvd");
  file << u;

  // Plot solution
  plot(u);
  interactive();

  return 0;
}
