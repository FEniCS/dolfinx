// Copyright (C) 2009 Bartosz Sawicki.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-03-30
// Last changed: 2009-10-05
//
// Eddy currents phenomena in low conducting body can be
// described using electric vector potential and curl-curl operator:
//    \nabla \times \nabla \times T = - \frac{\partial B}{\partial t}
// Electric vector potential defined as:
//    \nabla \times T = J
//
// Boundary condition
//    J_n = 0,
//    T_t=T_w=0, \frac{\partial T_n}{\partial n} = 0
// which is naturaly fulfilled for zero Dirichlet BC with Nedelec (edge)
// elements.

#include <dolfin.h>
#include "EddyCurrents.h"
#include "CurrentDensity.h"

using namespace dolfin;

int main()
{
  // Homogenous external magnetic field (dB/dt)
  class Source : public Expression
  {
  public:

    Source() : Expression(3) {}

    void eval(Array<double>& values, const Array<const double>& x) const
    {
      values[0] = 0.0;
      values[1] = 0.0;
      values[2] = 1.0;
    }

  };

  // Zero Dirichlet BC
  class Zero : public Expression
  {
  public:

    Zero() : Expression(3) {}

    void eval(Array<double>& values, const Array<const double>& x) const
    {
      values[0] = 0.0;
      values[1] = 0.0;
      values[2] = 0.0;
    }

  };

  // Everywhere on external surface
  class DirichletBoundary: public SubDomain
  {
    bool inside(const double* x, bool on_boundary) const
    {
      return on_boundary;
    }
  };

  // Create demo mesh
  UnitSphere mesh(8);

  // Define functions
  Source dbdt;
  Zero zero;

  // Define function space and boundary condition
  EddyCurrents::FunctionSpace V(mesh);
  DirichletBoundary boundary;
  DirichletBC bc(V, zero, boundary);

  // Use forms to define variational problem
  EddyCurrents::BilinearForm a(V,V);
  EddyCurrents::LinearForm L(V);
  L.dbdt = dbdt;
  VariationalProblem problem (a, L,  bc);

  // Solve problem using default solver
  Function T(V);
  problem.solve(T);

  // Define variational problem for current density (J)
  CurrentDensity::FunctionSpace V1(mesh);
  CurrentDensity::BilinearForm a1(V1,V1);
  CurrentDensity::LinearForm L1(V1);
  L1.T = T;
  VariationalProblem problem1(a1, L1);

  // Solve problem using default solver
  Function J(V1);
  problem1.solve(J);

  // Plot solution
  plot(J);

  File file("current_density.pvd");
  file << J;

  return 0;
}
