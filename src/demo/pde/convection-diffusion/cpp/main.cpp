// Copyright (C) 2006-2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2006-02-09
// Last changed: 2007-07-11
//
// This demo solves the time-dependent convection-diffusion equation by
// a least-squares stabilized cG(1)cG(1) method. The velocity field used
// in the simulation is the output from the Stokes (Taylor-Hood) demo.
// The sub domains for the different boundary conditions are computed
// by the demo program in src/demo/subdomains.

#include <dolfin.h>
#include "ConvectionDiffusion.h"

using namespace dolfin;

int main()
{
  // Read mesh and sub domain markers
  Mesh mesh("../../../../../data/meshes/dolfin-2.xml.gz");
  MeshFunction<unsigned int> sub_domains(mesh, "../subdomains.xml.gz");

  // Read velocity field
  Function velocity("../velocity.xml.gz");

  // Source term and initial condition
  Function f(mesh, 0.0);
  Function u0(mesh, 0.0);

  // Set up forms
  ConvectionDiffusionBilinearForm a(velocity);
  ConvectionDiffusionLinearForm L(u0, velocity, f);

  // Set up boundary condition
  Function g(mesh, 1.0);
  DirichletBC bc(g, sub_domains, 1);

  // Linear system
  Matrix A;
  Vector x, b;

  // Creat dof maps
  DofMapSet dof_map_set(a, mesh);

  // Solution vector
  Function u1(mesh, dof_map_set[1], x, a);

  // Assemble matrix
  assemble(A, a, mesh, dof_map_set);
  
  // Parameters for time-stepping
  real T = 2.0;
  real k = 0.05;
  real t = k;
  
  // Output file
  File file("temperature.pvd");

  // Time-stepping
  Progress p("Time-stepping");
  while ( t < T )
  {
    // Assemble vector and apply boundary conditions
    assemble(b, L, mesh, dof_map_set);
    bc.apply(A, b, dof_map_set[0], a);
    
    // Solve the linear system
    solve(A, x, b);
    
    // Save the solution to file
    file << u1;

    // Move to next interval
    p = t / T;
    t += k;
    u0 = u1;
  }

  // Plot solution
  plot(u1);

}
