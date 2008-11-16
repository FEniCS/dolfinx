// Copyright (C) 2006-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2006-02-09
// Last changed: 2008-11-16
//
// This demo solves the Stokes equations, using quadratic elements for
// the velocity and first degree elements for the pressure
// (Taylor-Hood elements). The sub domains for the different boundary
// conditions used in this simulation are computed by the demo program
// in src/demo/mesh/subdomains.

#include <dolfin.h>
#include "Stokes.h"

using namespace dolfin;

int main()
{
  // Function for no-slip boundary condition for velocity
  class Zero : public Function
  {
    void eval(double* values, const Data& data) const
    {
      values[0] = 0.0;
      values[1] = 0.0;
    }
  };

  // Function for no-slip boundary condition for velocity
  class Noslip : public Function
  {
    void eval(double* values, const Data& data) const
    {
      values[0] = 0.0;
      values[1] = 0.0;
    }
  };

  // Function for inflow boundary condition for velocity
  class Inflow : public Function
  {
    void eval(double* values, const Data& data) const
    {
      double y  = data.x[1];
      values[0] = -sin(y*DOLFIN_PI);
      values[1] = 0.0;
    }
  };

  // Read mesh and sub domain markers
  Mesh mesh("../../../../../data/meshes/dolfin-2.xml.gz");
  MeshFunction<unsigned int> sub_domains(mesh, "../subdomains.xml.gz");

  // Create function space
  StokesFunctionSpace V(mesh);

  // Create velocity subspace
  SubSpace Vu(V, 0);

  // Create pressure subspace
  SubSpace Vp(V, 1);

  // Create functions for boundary conditions
  Noslip noslip;
  Inflow inflow;
  Constant zero(0.0);
  
  // No-slip boundary condition for velocity
  DirichletBC bc0(noslip, Vu, sub_domains, 0);

  // Inflow boundary condition for velocity
  DirichletBC bc1(inflow, Vu, sub_domains, 1);

  // Boundary condition for pressure at outflow
  DirichletBC bc2(zero, Vp, sub_domains, 2);

  // Collect boundary conditions
  Array<BoundaryCondition*> bcs(&bc0, &bc1, &bc2);

  // Set up PDE
  Zero f;
  StokesBilinearForm a(V, V);
  StokesLinearForm L(V);
  L.f = f;
  LinearPDE pde(a, L, bcs);

  // Solve PDE
  Function w;
  pde.set("PDE linear solver", "direct");
  pde.solve(w);
  Function u = w[0];
  Function p = w[1];

  // Plot solution
  plot(u);
  plot(p);

  // Save solution in XML format
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
