// Copyright (C) 2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-09-15
// Last changed: 2010-05-03
//
// Simple Poisson solver

#include <cstdlib>
#include <dolfin.h>
#include "Poisson.h"

using namespace dolfin;

int main(int argc, char* argv[])
{
  // Parse command-line arguments
  if (argc != 2)
  {
    info("Usage: solve-poisson n");
    return 1;
  }
  int n = atoi(argv[1]);

  // Create mesh and function space
  UnitCube mesh(n, n, n);
  Poisson::FunctionSpace V(mesh);

  // Define boundary condition
  Constant u0(0.0);
  DomainBoundary boundary;
  DirichletBC bc(V, u0, boundary);

  // Define variational problem
  Poisson::BilinearForm a(V, V);
  Poisson::LinearForm L(V);
  Constant f(1.0);
  L.f = f;
  VariationalProblem problem(a, L, bc);
  info(problem.parameters, true);
  problem.parameters["linear_solver"] = "cg";
  problem.parameters["preconditioner"] = "amg_hypre";
  Function u(V);

  // Solve problem
  MPI::barrier();
  double t = time();
  problem.solve(u);
  MPI::barrier();
  t = time() - t;

  // Report timing
  if (MPI::process_number() == 0)
    info("TIME: %.5g", t);

  return 0;
}
