// Copyright (C) 2006-2011 Anders Logg and Kristian B. Oelgaard
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
// This demo program solves Poisson's equation,
//
//     - div grad u(x, y) = f(x, y)
//
// on the unit square with source f given by
//
//     f(x, y) = -100*exp(-((x - 0.5)^2 + (y - 0.5)^2)/0.02)
//
// and boundary conditions given by
//
//     u(x, y)     = u0 on x = 0 and x = 1
//     du/dn(x, y) = g  on y = 0 and y = 1
//
// where
//
//     u0 = x + 0.25*sin(2*pi*x)
//     g = (y - 0.5)**2
//
// using a discontinuous Galerkin formulation (interior penalty method).

#include <dolfin.h>
#include "Poisson.h"

using namespace dolfin;

int main()
{
  // Source term
  class Source : public Expression
  {
    void eval(Array<double>& values, const Array<double>& x) const
    {
      const double dx = x[0] - 0.5;
      const double dy = x[1] - 0.5;
      values[0] = -100.0*exp(-(dx*dx + dy*dy)/0.02);
    }
  };

  // Dirichlet term
  class BoundaryValue : public Expression
  {
    void eval(Array<double>& values, const Array<double>& x) const
    {
      values[0] = x[0] + 0.25*sin(2*M_PI*x[1]);
    }
  };

  // Neumann term
  class BoundaryDerivative : public Expression
  {
    void eval(Array<double>& values, const Array<double>& x) const
    {
      const double dx = x[0] - 0.5;
      values[0] = dx*dx;
    }
  };

  // Sub domain for Dirichlet boundary condition, x = 1 and x = 0
  class DirichletBoundary : public SubDomain
  {
    bool inside(const Array<double>& x, bool on_boundary) const
    { return on_boundary and near(x[0]*(1 - x[0]), 0); }
  };

  // Sub domain for Neumann boundary condition, y = 1 and y = 0
  class NeumannBoundary : public SubDomain
  {
    bool inside(const Array<double>& x, bool on_boundary) const
    { return on_boundary and near(x[1]*(1 - x[1]), 0); }
  };


  // FIXME: Make mesh ghosted
  parameters["ghost_mode"] = "shared_facet";

  // Create mesh
  UnitSquareMesh mesh(24, 24);

  // Create functions
  Source f;
  BoundaryValue u0;
  BoundaryDerivative g;

  // Create funtion space
  auto V = std::make_shared<Poisson::FunctionSpace>(mesh);

  // Mark facets of the mesh
  NeumannBoundary neumann_boundary;
  DirichletBoundary dirichlet_boundary;

  FacetFunction<std::size_t> boundaries(mesh, 0);
  neumann_boundary.mark(boundaries, 2);
  dirichlet_boundary.mark(boundaries, 1);

  // Define variational problem
  Poisson::BilinearForm a(V, V);
  Poisson::LinearForm L(V);
  L.f  = f;
  L.u0 = u0;
  L.g  = g;

  // Attach marked facets to bilinear and linear form
  a.ds = boundaries;
  L.ds = boundaries;

  // Compute solution
  Function u(V);
  solve(a == L, u);

  // Save solution in VTK format
  File file("poisson.pvd");
  file << u;

  // Plot solution
  plot(u);
  interactive();

  return 0;
}
