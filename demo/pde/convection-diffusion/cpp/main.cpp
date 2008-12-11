// Copyright (C) 2006-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2006-02-09
// Last changed: 2008-11-17
//
// This demo solves the time-dependent convection-diffusion equation
// by a least-squares stabilized cG(1)cG(1) method. The velocity field
// used in the simulation is the output from the Stokes (Taylor-Hood)
// demo.  The sub domains for the different boundary conditions are
// computed by the demo program in src/demo/subdomains.

#include <dolfin.h>
#include "ConvectionDiffusion.h"

using namespace dolfin;

int main(int argc, char *argv[])
{
  // Read velocity field
  Function velocity("../velocity.xml.gz");

  // Read sub domain markers
  dolfin_debug("");
  const Mesh& mesh(velocity.function_space().mesh());
  dolfin_debug("");
  MeshFunction<unsigned int> sub_domains(mesh, "../subdomains.xml.gz");
  dolfin_debug("");

  // Create function space
  ConvectionDiffusionFunctionSpace V(mesh);
  dolfin_debug("");

  // Source term and initial condition
  Constant f(0.0);
  Function u(V);
  u.vector().zero();

  // Set up forms
  ConvectionDiffusionBilinearForm a(V, V);
  a.b = velocity;
  ConvectionDiffusionLinearForm L(V);
  L.u0 = u; L.b = velocity; L.f = f;

  // Set up boundary condition
  Constant g(1.0);
  DirichletBC bc(g, V, sub_domains, 1);

  // Solution
  Function u1(V);

  // Linear system
  Matrix A;
  Vector b;

  // LU
  LUSolver lu;

  // Assemble matrix
  assemble(A, a);
  bc.apply(A);

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
    assemble(b, L);
    bc.apply(b);
    
    // Solve the linear system
    lu.solve(A, u.vector(), b);
    
    // Save solution in VTK format
    file << u;

    // Move to next interval
    p = t / T;
    t += k;
  }

  // Plot solution
  plot(u);
}
