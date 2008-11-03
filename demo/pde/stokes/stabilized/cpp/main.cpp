// Copyright (C) 2006-2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2006-02-09
// Last changed: 2007-07-11
//
// This demo solves the Stokes equations, using stabilized
// first order elements for the velocity and pressure. The
// sub domains for the different boundary conditions used
// in this simulation are computed by the demo program in
// src/demo/mesh/subdomains.

#include <dolfin.h>
#include "Stokes.h"

using namespace dolfin;

int main()
{
  // Function for no-slip boundary condition for velocity
  class ScalarZero : public Function
  {
  public:

    ScalarZero(const FunctionSpace& V) : Function(V) {}

    double eval(const double* x) const
    {
      return 0.0;
    }
  };

  // Function for no-slip boundary condition for velocity
  class Zero : public Function
  {
  public:

    void eval(double* values, const double* x) const
    {
      values[0] = 0.0;
      values[1] = 0.0;
    }
  };

  // Function for no-slip boundary condition for velocity
  class Noslip : public Function
  {
  public:

    Noslip(const FunctionSpace& V) : Function(V) {}

    void eval(double* values, const double* x) const
    {
      values[0] = 0.0;
      values[1] = 0.0;
    }
  };

  // Function for inflow boundary condition for velocity
  class Inflow : public Function
  {
  public:

    Inflow(const FunctionSpace& V) : Function(V) {}

    void eval(double* values, const double* x) const
    {
      values[0] = -sin(x[1]*DOLFIN_PI);
      values[1] = 0.0;
    }

  };

  // Read mesh and sub domain markers
  Mesh mesh("../../../../../data/meshes/dolfin-2.xml.gz");
  mesh.order();
  MeshFunction<unsigned int> sub_domains(mesh, "../subdomains.xml.gz");

  // Create function spaces
  StokesFunctionSpace V(mesh);
  std::auto_ptr<const FunctionSpace> Vu(V.extract_sub_space(0));
  std::auto_ptr<const FunctionSpace> Vp(V.extract_sub_space(1));

  // Create functions for boundary conditions
  Noslip noslip(*Vu);
  Inflow inflow(*Vu);
  ScalarZero zero(*Vp);
  //Constant zero(0.0);
  
  // No-slip boundary condition for velocity
  DirichletBC bc0(noslip, *Vu, sub_domains, 0);

  // Inflow boundary condition for velocity
  DirichletBC bc1(inflow, *Vu, sub_domains, 1);

  // Boundary condition for pressure at outflow
  DirichletBC bc2(zero, *Vp, sub_domains, 2);

  // Collect boundary conditions
  Array<DirichletBC*> bcs(&bc0, &bc1, &bc2);

  // Set up PDE
  MeshSize h;
  //Function f(2, 0.0);
  Zero f;
  StokesBilinearForm a(V, V);
  a.h = h;
  StokesLinearForm L(V);
  L.f = f;
  L.h = h;
  LinearPDE pde(a, L, mesh, bcs);

  // Solve PDE
  Function U(V);
  pde.set("PDE linear solver", "direct");
  pde.solve(U);

  Function u = U[0];
  Function p = U[1];

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

  File x_file("x.xml");
  x_file << u.vector();


}
