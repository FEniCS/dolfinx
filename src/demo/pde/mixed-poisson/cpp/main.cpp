// Copyright (C) 2007 Anders Logg and Marie Rognes.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-04-20
// Last changed: 2007-05-17
//
// This demo program solves the mixed formulation of
// Poisson's equation:
//
//     sigma + grad(u) = 0
//          div(sigma) = f
//
// The corresponding weak (variational problem)
//
//     <tau, sigma> - <div(tau), u> = 0       for all tau
//                  <w, div(sigma)> = <w, f>  for all w
//
// is solved using BDM (Brezzi-Douglas-Marini) elements
// of degree q (tau, sigma) and DG (discontinuous Galerkin)
// elements of degree q - 1 for (w, u).

#include <dolfin.h>
#include "MixedPoisson.h"

using namespace dolfin;

int main()
{
  // Source term
  class Source : public Function
  {
  public:
    
    Source(Mesh& mesh) : Function(mesh) {}

    real eval(const real* x) const
    {
      real dx = x[0] - 0.5;
      real dy = x[1] - 0.5;
      return 500.0*exp(-(dx*dx + dy*dy)/0.02);
    }

  };

  // Create mesh and source term
  UnitSquare mesh(16, 16);
  Source f(mesh);
  
  // Define PDE
  MixedPoissonBilinearForm a;
  MixedPoissonLinearForm L(f);
  LinearPDE pde(a, L, mesh);

  // Solve PDE
  Function sigma;
  Function u;
  pde.solve(sigma, u);

  // Plot solution
  plot(sigma);
  plot(u);

  // Save solution to file
  File f0("sigma.xml");
  File f1("u.xml");
  f0 << sigma;
  f1 << u;

  // Save solution to pvd format
  File f3("sigma.pvd");
  File f4("u.pvd");
  f3 << sigma;
  f4 << u;

  return 0;
}
