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
  Function u(V);

  // Create linear solver
  KrylovSolver solver("gmres", "amg_hypre");

  // Assemble matrix and vector, and apply Dirichlet boundary conditions
  Matrix A;
  Vector b;
  dolfin::MPI::barrier();
  double t = time();
  assemble(A, a);
  assemble(b, L);
  bc.apply(A, b);
  dolfin::MPI::barrier();
  t = time() - t;

  // Solve problem
  dolfin::MPI::barrier();
  t = time();
  solver.solve(A, u.vector(), b);
  dolfin::MPI::barrier();
  t = time() - t;

  // Report timing
  if (dolfin::MPI::process_number() == 0)
    info("TIME: %.5g", t);

  return 0;
}
