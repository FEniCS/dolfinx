// Copyright (C) 2006-2008 Anders Logg and Kristian Oelgaard.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2006-12-05
// Last changed: 2008-12-18
//
// This demo program solves Poisson's equation,
//
//     - div grad u(x, y) = f(x, y)
//
// on the unit square with source f given by
//
//     f(x, y) = 500*exp(-((x-0.5)^2 + (y-0.5)^2)/0.02)
//
// and boundary conditions given by
//
//     u(x, y)     = 0
//     du/dn(x, y) = 0
//
// using a discontinuous Galerkin formulation (interior penalty method).

#include <dolfin.h>
#include "Poisson.h"
#include "P1Projection.h"

using namespace dolfin;

int main()
{
  // Source term
  class Source : public Function
  {
    void eval(double* values, const double* x) const
    {
      double dx = x[0] - 0.5;
      double dy = x[1] - 0.5;
      values[0] = 500.0*exp(-(dx*dx + dy*dy)/0.02);
    }
  };

  // Create mesh
  UnitSquare mesh(24, 24);

  dolfin_set("linear algebra backend", "uBLAS");

  // Create functions
  Source f;
  FacetNormal n;
  AvgMeshSize h;

  // Create funtion space
  Poisson::FunctionSpace V(mesh);

  // Define forms and attach functions
  Poisson::BilinearForm a(V, V);
  Poisson::LinearForm L(V);
  a.n = n; a.h = h; L.f = f;

  // Create variational problem
  VariationalProblem problem(a, L);
  problem.parameters("symmetric") = true;

  // Solve variational problem
  Function u;
  problem.solve(u);

  // Plot solution projected
  plot(u);

  // Save solution in VTK format
  File file("poisson.pvd");
  file << u;

  return 0;
}
