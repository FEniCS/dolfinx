// Copyright (C) 2007-2011 Anders Logg and Marie E. Rognes
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
// First added:  2007-04-20
// Last changed: 2012-11-12

#include <dolfin.h>
#include "MixedPoisson.h"

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

// Boundary source for flux boundary condition
class BoundarySource : public Expression
{
public:

  BoundarySource(const Mesh& mesh) : Expression(2), mesh(mesh) {}

  void eval(Array<double>& values, const Array<double>& x,
            const ufc::cell& ufc_cell) const
  {
    dolfin_assert(ufc_cell.local_facet >= 0);

    Cell cell(mesh, ufc_cell.index);
    Point n = cell.normal(ufc_cell.local_facet);

    const double g = sin(5*x[0]);
    values[0] = g*n[0];
    values[1] = g*n[1];
  }

private:

  const Mesh& mesh;

};

// Sub domain for essential boundary condition
class EssentialBoundary : public SubDomain
{
  bool inside(const Array<double>& x, bool on_boundary) const
  {
    return x[1] < DOLFIN_EPS or x[1] > 1.0 - DOLFIN_EPS;
  }
};

int main()
{
  // Create mesh
  UnitSquareMesh mesh(32, 32);

  // Construct function space
  auto W = std::make_shared<MixedPoisson::FunctionSpace>(mesh);
  MixedPoisson::BilinearForm a(W, W);
  MixedPoisson::LinearForm L(W);

  // Create source and assign to L
  auto f = std::make_shared<Source>();
  L.f = f;

  // Define boundary condition
  auto G = std::make_shared<BoundarySource>(mesh);
  auto boundary = std::make_shared<EssentialBoundary>();
  DirichletBC bc(W->sub(0), G, boundary);

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
