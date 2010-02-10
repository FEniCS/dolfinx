// Copyright (C) 2007 Anders Logg and Marie Rognes.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2008.
//
// First added:  2007-04-20
// Last changed: 2010-01-27
//
// This demo program solves the mixed formulation of
// Poisson's equation,
//
//     sigma + grad(u) = 0,
//          div(sigma) = f.
//
// The corresponding weak (variational problem),
//
//     <tau, sigma> - <div(tau), u> = 0       for all tau,
//                  <w, div(sigma)> = <w, f>  for all w,
//
// is solved using BDM (Brezzi-Douglas-Marini) elements of degree q
// for (tau, sigma) and DG (discontinuous Galerkin) elements of degree
// q - 1 for (w, u).

#include <dolfin.h>
#include "MixedPoisson.h"
#include "P1Projection.h"

using namespace dolfin;

int main()
{
  // Source term
  class Source : public Expression
  {
  public:

    void eval(Array<double>& values, const Array<double>& x) const
    {
      double dx = x[0] - 0.5;
      double dy = x[1] - 0.5;
      values[0] = 500.0*exp(-(dx*dx + dy*dy)/0.02);
    }

  };

  // Create mesh and source term
  UnitSquare mesh(32, 32);
  Source f;

  // Define PDE
  MixedPoisson::FunctionSpace V(mesh);
  MixedPoisson::BilinearForm a(V, V);
  MixedPoisson::LinearForm L(V);
  L.f = f;
  VariationalProblem pde(a, L);

  // Solve PDE
  Function w(V);
  pde.solve(w);

  // Extract sub functions (function views)
  Function& sigma = w[0];
  Function& u = w[1];

  // Project sigma onto P1 continuous Lagrange for post-processing
  P1Projection::FunctionSpace Vp(mesh);
  P1Projection::BilinearForm a_p(Vp, Vp);
  P1Projection::LinearForm L_p(Vp);
  L_p.f = sigma;
  VariationalProblem pde_project(a_p, L_p);
  Function sigma_p(Vp);
  pde_project.solve(sigma_p);

  // Plot solution
  plot(sigma_p);
  plot(u);

  // Save solution in VTK format
  File f3("sigma.pvd");
  File f4("u.pvd");
  f3 << sigma_p;
  f4 << u;

  return 0;
}
