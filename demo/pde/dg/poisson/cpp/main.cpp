// Copyright (C) 2006-2007 Anders Logg and Kristian Oelgaard.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2006-12-05
// Last changed: 2007-08-20
//
// This demo program solves Poisson's equation
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
  public:
    
    Source(Mesh& mesh) : Function(mesh) {}

    double eval(const double* x) const
    {
      double dx = x[0] - 0.5;
      double dy = x[1] - 0.5;
      return 500.0*exp(-(dx*dx + dy*dy)/0.02);
    }

  };
 
  // Create mesh
  UnitSquare mesh(64, 64);

  // Create functions
  Source f(mesh);
  FacetNormal n(mesh);
  AvgMeshSize h(mesh);

  // Define PDE
  PoissonBilinearForm a(n, h);
  PoissonLinearForm L(f);
  LinearPDE pde(a, L, mesh);

  // Solve PDE
  Function u;
  pde.solve(u);

  // Project solution onto continuous basis for post-processing
  Function u_proj;
  P1ProjectionBilinearForm a_proj;
  P1ProjectionLinearForm L_proj(u);
  LinearPDE pde_proj(a_proj, L_proj, mesh);
  pde_proj.solve(u_proj);

  // Plot solution
  plot(u_proj);

  // Save solution to file
  File file("poisson.pvd");
  file << u_proj;

  return 0;
}
