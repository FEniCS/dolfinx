// Copyright (C) 2006-2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2006-02-09
// Last changed: 2007-04-24

#include <dolfin.h>
#include "Stokes.h"

using namespace dolfin;

int main()
{
  // No-slip boundary condition for velocity
  class Noslip : public Function
  {
  public:

    Noslip(Mesh& mesh) : Function(mesh) {}

    void eval(real* values, const real* x)
    {
      values[0] = 0.0;
      values[1] = 0.0;
    }

  };

  // Sub domain for no-slip boundary condition (top and bottom)
  class NoslipDomain : public SubDomain
  {
    bool inside(const real* x, bool on_boundary)
    {
      return x[0] > DOLFIN_EPS && x[0] < 1.0 && on_boundary;
    }
  };

  // Inflow boundary condition for velocity
  class Inflow : public Function
  {
  public:

    Inflow(Mesh& mesh) : Function(mesh) {}

    void eval(real* values, const real* x)
    {
      values[0] = -1.0;
      values[1] = 0.0;
    }

  };

  // Sub domain for inflow boundary condition (right)
  class InflowDomain : public SubDomain
  {
    bool inside(const real* x, bool on_boundary)
    {
      return x[0] > 1.0 - DOLFIN_EPS;
    }
  };

  // Pressure ground level boundary condition
  class GroundLevel : public Function
  {
  public:

    GroundLevel(Mesh& mesh) : Function(mesh) {}

    real eval(const real* x)
    {
      return 0.0;
    }
  };

  // Sub domain for pressure ground level (x = y = 0)
  class GroundLevelDomain : public SubDomain
  {
    bool inside(const real* x, bool on_boundary)
    {
      return x[0] < DOLFIN_EPS && x[1] < DOLFIN_EPS;
    }
  };

  // Read mesh
  Mesh mesh("../../../../../data/meshes/dolfin-2.xml.gz");
  
  // Set up boundary conditions
  Noslip g0(mesh); NoslipDomain G0; BoundaryCondition bc0(g0, mesh, G0);
  Inflow g1(mesh); InflowDomain G1; BoundaryCondition bc1(g1, mesh, G1);
  GroundLevel g2(mesh); GroundLevelDomain G2; BoundaryCondition bc2(g2, mesh, G2);
  Array <BoundaryCondition*> bcs;
  bcs.push_back(&bc0);
  bcs.push_back(&bc1);
  bcs.push_back(&bc2);

  // Set up PDE
  Function f(mesh, 0.0);
  StokesBilinearForm a;
  StokesLinearForm L(f);
  PDE pde(a, L, mesh, bcs);

  // Solve PDE
  Function w;
  pde.set("PDE linear solver", "direct");
  pde.solve(w);

  // Save solution
  File file("solution.xml");
  file << w;

  /*
  // Save solution to file
  File ufile("velocity.pvd");
  File pfile("pressure.pvd");
  ufile << U;
  pfile << P;
  */
}
