// Copyright (C) 2006-2007 Anders Logg and Kristian Oelgaard.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2006-12-05
// Last changed: 2008-11-19
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
  PoissonFunctionSpace V(mesh);

  // Define forms and attach functions
  PoissonBilinearForm a(V, V);
  PoissonLinearForm L(V);
  a.n = n; a.h = h; L.f = f;

  // Create PDE
  LinearPDE pde(a, L);

  // Solve PDE
  Function u;
  pde.solve(u);

  // Project solution onto continuous basis for post-processing
  P1ProjectionFunctionSpace U(mesh);
  P1ProjectionBilinearForm a_p(U, U);
  P1ProjectionLinearForm L_p(U);
  L_p.u = u;
  LinearPDE pde_proj(a_p, L_p);
  Function u_p(U);
  pde_proj.solve(u_p);

  // Plot solution projected
  plot(u_p);

  // Save solution in VTK format
  File file("poisson.pvd");
  file << u_p;

  return 0;
}
