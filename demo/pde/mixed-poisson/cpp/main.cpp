// Copyright (C) 2007-2010 Anders Logg and Marie Rognes.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2008.
//
// First added:  2007-04-20
// Last changed: 2010-08-31

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

  BoundarySource() : Expression(2) {}

  void eval(Array<double>& values, const Data& data) const
  {
    double g = sin(5*data.x[0]);
    values[0] = g*data.normal()[0];
    values[1] = g*data.normal()[1];
  }
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
  UnitSquare mesh(32, 32);

  // Construct function space
  MixedPoisson::FunctionSpace W(mesh);
  MixedPoisson::BilinearForm a(W, W);
  MixedPoisson::LinearForm L(W);

  // Create source and assign to L
  Source f;
  L.f = f;

  // Define boundary condition
  SubSpace W0(W, 0);
  BoundarySource G;
  EssentialBoundary boundary;
  DirichletBC bc(W0, G, boundary);

  // Define variational problem
  VariationalProblem problem(a, L, bc);

  // Compute (full) solution
  Function w(W);
  problem.solve(w);

  // Extract sub functions (function views)
  Function& sigma = w[0];
  Function& u = w[1];

  // Plot solutions
  plot(u);
  plot(sigma);

  return 0;
}
