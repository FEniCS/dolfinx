// Copyright (C) 2006-2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2006-02-09
// Last changed: 2007-07-11
//
// This demo solves the Stokes equations, using quadratic
// elements for the velocity and first degree elements for
// the pressure (Taylor-Hood elements). The sub domains
// for the different boundary conditions used in this
// simulation are computed by the demo program in
// src/demo/mesh/subdomains.

#include <dolfin.h>
#include "Stokes.h"

using namespace dolfin;

int main()
{
  // Function for no-slip boundary condition for velocity
  class Noslip : public Function
  {
  public:

    Noslip(Mesh& mesh) : Function(mesh) {}

    void eval(real* values, const real* x) const
    {
      values[0] = 0.0;
      values[1] = 0.0;
    }

  };

  // Function for inflow boundary condition for velocity
  class Inflow : public Function
  {
  public:

    Inflow(Mesh& mesh) : Function(mesh) {}

    void eval(real* values, const real* x) const
    {
      values[0] = -1.0;
      values[1] = 0.0;
    }

  };

  // Read mesh and sub domain markers
  Mesh mesh("../../../../../../data/meshes/dolfin-2.xml.gz");
  MeshFunction<unsigned int> sub_domains(mesh, "../subdomains.xml.gz");

  // Create functions for boundary conditions
  Noslip noslip(mesh);
  Inflow inflow(mesh);
  Function zero(mesh, 0.0);
  
  // Define sub systems for boundary conditions
  SubSystem velocity(0);
  SubSystem pressure(1);

  // No-slip boundary condition for velocity
  DirichletBC bc0(noslip, sub_domains, 0, velocity);

  // Inflow boundary condition for velocity
  DirichletBC bc1(inflow, sub_domains, 1, velocity);

  // Boundary condition for pressure at outflow
  DirichletBC bc2(zero, sub_domains, 2, pressure);

  // Collect boundary conditions
  Array <BoundaryCondition*> bcs(&bc0, &bc1, &bc2);

  // Set up PDE
  Function f(mesh, 0.0);
  StokesBilinearForm a;
  StokesLinearForm L(f);
  LinearPDE pde(a, L, mesh, bcs);

  // Solve PDE
  Function u;
  Function p;
  pde.set("PDE linear solver", "direct");
  pde.solve(u, p);

  // Plot solution
  plot(u);
  plot(p);

  // Save solution
  File ufile("velocity.xml");
  ufile << u;
  File pfile("pressure.xml");
  pfile << p;

  // Save solution in VTK format
  File ufile_pvd("velocity.pvd");
  ufile_pvd << u;
  File pfile_pvd("pressure.pvd");
  pfile_pvd << p;
}
