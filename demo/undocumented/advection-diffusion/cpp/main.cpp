// Copyright (C) 2006-2008 Anders Logg
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2006-02-09
// Last changed: 2011-01-17
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
  // Read mesh
  Mesh mesh("../mesh.xml.gz");
  mesh.init();

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

  // LU solver
  LUSolver lu(A);
  lu.parameters["reuse_factorization"] = true;

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
    lu.solve(u.vector(), b);

    // Save solution in VTK format
    file << std::make_pair(&u, t);

    // Move to next interval
    p = t / T;
    t += k;
  }
  std::cout << "Test vector norm: " << u.vector().norm("l2") << std::endl;

  // Plot solution
  plot(u);
}
