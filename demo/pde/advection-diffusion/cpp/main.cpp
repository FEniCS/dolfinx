// Copyright (C) 2006-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2006-02-09
// Last changed: 2008-12-12
//
// This demo solves the time-dependent advection-diffusion equation
// by a least-squares stabilized cG(1)cG(1) method. The velocity field
// used in the simulation is the output from the Stokes (Taylor-Hood)
// demo.  The sub domains for the different boundary conditions are
// computed by the demo program in src/demo/subdomains.

#include <dolfin.h>
#include "AdvectionDiffusion.h"
#include "Velocity.h"

using namespace dolfin;

int main(int argc, char *argv[])
{
  parameters["linear_algebra_backend"] = "uBLAS";

  // Read mesh
  Mesh mesh("../mesh.xml.gz");
  mesh.init();
  cout << "Vertices " << mesh.num_vertices() << endl;
  cout << "facets   " << mesh.num_facets() << endl;
  cout << "Vertices " << mesh.num_vertices() << endl;
  cout << "u_dimfacets   " << (mesh.num_vertices()+mesh.num_facets())*2 << endl;

  // Create velocity FunctionSpace
  Velocity::FunctionSpace V_u(mesh);

  // Create velocity function
  Function velocity(V_u, "../velocity.xml.gz");

  // Read sub domain markers
  MeshFunction<unsigned int> sub_domains(mesh, "../subdomains.xml.gz");

  // Create function space
  AdvectionDiffusion::FunctionSpace V(mesh);

  // Source term and initial condition
  Constant f(0.0);
  Function u(V);
  u.vector().zero();

  // Set up forms
  AdvectionDiffusion::BilinearForm a(V, V);
  a.b = velocity;
  AdvectionDiffusion::LinearForm L(V);
  L.u0 = u; L.b = velocity; L.f = f;

  // Set up boundary condition
  Constant g(1.0);
  DirichletBC bc(V, g, sub_domains, 1);

  // Solution
  Function u1(V);

  // Linear system
  Matrix A;
  Vector b;

  // Assemble matrix
  assemble(A, a);
  bc.apply(A);

  // LU
  UmfpackLUSolver lu(A);
  lu.factorize();

  // Parameters for time-stepping
  double T = 2.0;
  const double k = 0.05;
  double t = k;

  // Output file
  File file("temperature.pvd");

  // Time-stepping
  Progress p("Time-stepping");
  while (t < T)
  {
    // Assemble vector and apply boundary conditions
    assemble(b, L);
    bc.apply(b);

    // Solve the linear system (re-use the already factorized matrix A)
    lu.solve_factorized(u.vector(), b);

    // Save solution in VTK format
    file << std::make_pair(&u, t);

    // Move to next interval
    p = t / T;
    t += k;
  }

  // Plot solution
  plot(u);
}
