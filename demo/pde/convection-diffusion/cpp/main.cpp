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

int main(int argc, char *argv[])
{
  // Read mesh and sub domain markers
  Mesh mesh("../../../../data/meshes/dolfin-2.xml.gz");
  mesh.order();
  MeshFunction<unsigned int> sub_domains(mesh, "../subdomains.xml.gz");


  // Read velocity field
  Function velocity("../velocity.xml.gz");

  // Create function space
  ConvectionDiffusionFunctionSpace V(mesh);

  // Source term and initial condition
  Constant f(0.0);
  Function u0(V);
  u0.vector().zero();

  // Set up forms
  ConvectionDiffusionBilinearForm a(V, V);
  a.b = velocity;
  ConvectionDiffusionLinearForm L(V);
  L.u0 = u0; L.b = velocity; L.f = f;  

  // Set up boundary condition
  Constant g(1.0);
  DirichletBC bc(g, V, sub_domains, 1);

  // Solution vector
  Function u1(V);

  // Linear system
  Matrix A;
  Vector b;
  GenericVector& x = u1.vector();

  // LU
  LUSolver lu;

  // Assemble matrix
  Assembler::assemble(A, a);
  Assembler::assemble(b, L);
  bc.apply(A, b);

  // Parameters for time-stepping
  double T = 2.0;
  double k = 0.05;
  double t = k;
  
  // Output file
  File file("temperature.pvd");

  // Time-stepping
  Progress p("Time-stepping");
  while ( t < T )
  {
    // Assemble vector and apply boundary conditions
    Assembler::assemble(b, L);
    bc.apply(A, b);
    
    // Solve the linear system
    lu.solve(A, x, b);
    
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
