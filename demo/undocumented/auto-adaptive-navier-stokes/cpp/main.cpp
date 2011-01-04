// Copyright (C) 2010 Marie E. Rognes (meg@simula.no)
// Licensed under the GNU LGPL Version 3 or any later version

// First added:  2010-08-19
// Last changed: 2011-01-04

#include <dolfin.h>
#include "NavierStokes.h"

using namespace dolfin;

// No-slip boundary
class Noslip : public SubDomain
{
  bool inside(const Array<double>& x, bool on_boundary) const
  {
    return ((x[1] < DOLFIN_EPS || x[1] > 1.0 - DOLFIN_EPS) ||
            (on_boundary && abs(x[0] - 1.5) < 0.1 + DOLFIN_EPS));
  }
};

// Subdomain on which goal function should be defined
class Outflow : public SubDomain
{
  bool inside(const Array<double>& x, bool on_boundary) const
  {
    return x[0] > 4.0 - DOLFIN_EPS;
  }
};

// Applied pressure (Neumann boundary condition)
class Pressure : public Expression
{
  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0] = (4.0 - x[0])/4.0;
  }
};

int main() {

  // Create mesh and function space
  Mesh mesh("channel_with_flap.xml");

  NavierStokes::TestSpace W(mesh); // FIXME: typedefs

  // Unknown
  Function w(W);

  // Define boundary condition
  Constant u0(0.0, 0.0);
  Noslip noslip;
  SubSpace W0(W, 0);
  DirichletBC bc(W0, u0, noslip);

  // Define variational problem
  Constant nu(0.02);
  NavierStokes::LinearForm F(W);
  Pressure p0;
  F.p0 = p0;
  F.nu = nu;
  F.w = w;

  // Define goal functional
  NavierStokes::GoalFunctional M(mesh);

  // FIXME: The darned exterior_facet_domains must be tackled somewhere
  // Outflow outflow; M = u.ds(outflow);

  // Define variational problem
  NavierStokes::BilinearForm dF(W, W);
  dF.nu = nu;
  dF.w = w;

  // New notation for variational problem
  VariationalProblem pde(F, dF, bc);

  double tol = 0.0;
  pde.parameters("adaptive_solver")["reference"] = 0.82174229794; // FIXME

  // Solve problem with goal-oriented error control to given tolerance
  pde.solve(w, tol, M);

  summary();

  return 0;
}


/*

Python:

Refinement level   : 0
Value of functional: 0.82174229794
Tolerance          : 0
Error estimate     : 0.00928838535272
Number of cells    : 780
Number of dofs     : 3788

Timings/s

Compute solution   : 1.75277996063
Estimate error     : 3.22601509094

cpp :)

  Refinement level      : 0
  Reference             : 0
  Tolerance             : 0
  Error estimate        : 0.0092884
  Number of cells       : 780
  Number of dofs        : 3788
  Maximal number of iterations (1) exceeded! Returning anyhow.

*/
