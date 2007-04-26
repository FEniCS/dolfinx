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

  // Read mesh and sub domain markers
  Mesh mesh("../../../../../data/meshes/dolfin-2.xml.gz");
  MeshFunction<unsigned int> sub_domains(mesh, "subdomains.xml.gz");

  // Create functions for boundary conditions
  Noslip noslip(mesh);
  Inflow inflow(mesh);
  Function zero(mesh, 0.0);
  
  // Define sub systems for boundary conditions
  SubSystem velocity(0);
  SubSystem pressure(1);

  // No-slip boundary condition for velocity
  BoundaryCondition bc0(noslip, sub_domains, 0, velocity);

  // Inflow boundary condition for velocity
  BoundaryCondition bc1(inflow, sub_domains, 1, velocity);

  // Boundary condition for pressure at inflow
  BoundaryCondition bc2(zero, sub_domains, 1, pressure);

  // Collect boundary conditions
  Array <BoundaryCondition*> bcs(&bc0, &bc1, &bc2);

  // Set up PDE
  Function f(mesh, 0.0);
  StokesBilinearForm a;
  StokesLinearForm L(f);

  //PDE pde(a, L, mesh, bcs);
  PDE pde(a, L, mesh);

  // Solve PDE
  Function w;
  pde.set("PDE linear solver", "direct");
  pde.solve(w);

  // Save solution
  //File file("solution.xml");
  //file << w;

  /*
  // Save solution to file
  File ufile("velocity.pvd");
  File pfile("pressure.pvd");
  ufile << U;
  pfile << P;
  */
}
