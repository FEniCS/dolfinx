// Copyright (C) 2010 Garth N. Wells
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2010-08-07
// Last changed:
//
// This demo solves the Stokes equations, using quadratic elements for
// the velocity and first degree elements for the pressure
// (Taylor-Hood elements).

#include <dolfin.h>
#include "Stokes.h"

using namespace dolfin;

int main()
{
  // Sub domain for boundary condition on left-hand boundary
  class Left : public SubDomain
  {
    bool inside(const Array<double>& x, bool on_boundary) const
    {
      return x[0] < DOLFIN_EPS && on_boundary;
    }
  };

  // Sub domain for boundary condition on right-hand boundary
  class Right : public SubDomain
  {
    bool inside(const Array<double>& x, bool on_boundary) const
    {
      return x[0] > (1.0 - DOLFIN_EPS) && on_boundary;
    }
  };

  // Sub domain for top and bottom boundary
  class TopBottom : public SubDomain
  {
    bool inside(const Array<double>& x, bool on_boundary) const
    {
      return (x[1] > (1.0 - DOLFIN_EPS) && on_boundary) || (x[1] < DOLFIN_EPS && on_boundary);
    }
  };

  // Function for no-slip boundary condition for velocity
  class Noslip : public Expression
  {
  public:

    Noslip() : Expression(2) {}

    void eval(Array<double>& values, const Array<double>& x) const
    {
      values[0] = 0.0;
      values[1] = 0.0;
    }

  };

  // Function for inflow boundary condition for velocity
  class Inflow : public Expression
  {
  public:

    Inflow() : Expression(2) {}

    void eval(Array<double>& values, const Array<double>& x) const
    {
      values[0] = -sin(x[1]*DOLFIN_PI);
      values[1] = 0.0;
    }

  };

  // Read mesh and sub domain markers
  UnitSquare mesh(32, 32);

  // Create function space and subspaces
  Stokes::FunctionSpace W(mesh);
  SubSpace W0(W, 0);
  SubSpace W1(W, 1);

  // Create functions for boundary conditions
  Noslip noslip;
  Inflow inflow;
  Constant zero(0);

  // No-slip boundary condition for velocity
  TopBottom top_bottom;
  DirichletBC bc0(W0, noslip, top_bottom);

  // Inflow boundary condition for velocity
  Left left;
  DirichletBC bc1(W0, inflow, left);

  // Boundary condition for pressure at outflow
  Right right;
  DirichletBC bc2(W1, zero, right);

  // Collect boundary conditions
  std::vector<const BoundaryCondition*> bcs;
  bcs.push_back(&bc0); bcs.push_back(&bc1); bcs.push_back(&bc2);

  // Set up PDE
  Constant f(0.0, 0.0);
  Stokes::BilinearForm a(W, W);
  Stokes::LinearForm L(W);
  L.f = f;
  VariationalProblem problem(a, L, bcs);

  // Solve PDE
  Function w(W);
  problem.parameters["linear_solver"] = "direct";
  problem.solve(w);
  Function u = w[0];
  Function p = w[1];

  // Plot solution
  plot(u);
  plot(p);

  // Save solution in VTK format
  File ufile_pvd("velocity.pvd");
  ufile_pvd << u;
  File pfile_pvd("pressure.pvd");
  pfile_pvd << p;
}
