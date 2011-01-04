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

  #ifdef HAS_SLEPC

  //parameters["mesh_partitioner"] = "SCOTCH";
  //parameters["linear_algebra_backend"] = "Epetra";
  parameters["linear_algebra_backend"] = "PETSc";

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

  // Create preconditioner and linear solver
  //TrilinosPreconditioner pc("amg_ml");
  //EpetraKrylovSolver solver("gmres", pc);
  PETScPreconditioner pc("amg_hypre");
  PETScKrylovSolver solver("gmres", pc);

  // Assemble matrix and vector, and apply Dirichlet boundary conditions
  Matrix A;
  Vector b;
  assemble(A, a);
  assemble(b, L);
  bc.apply(A, b);

  // Solve linear system
  dolfin::MPI::barrier();
  double t = time();
  solver.solve(A, u.vector(), b);
  dolfin::MPI::barrier();
  t = time() - t;
  if (dolfin::MPI::process_number() == 0)
    info("TIME (first time): %.5g", t);

  // Solve linear system (preconditioner assuming same non-zero pattern)
  solver.parameters("preconditioner")["same_nonzero_pattern"] = true;
  u.vector().zero();
  dolfin::MPI::barrier();
  t = time();
  solver.solve(A, u.vector(), b);
  dolfin::MPI::barrier();
  t = time() - t;
  if (dolfin::MPI::process_number() == 0)
    info("TIME (same nonzero pattern): %.5g", t);

  // Solve linear system (re-use preconditioner)
  solver.parameters("preconditioner")["reuse"] = true;
  u.vector().zero();
  dolfin::MPI::barrier();
  t = time();
  solver.solve(A, u.vector(), b);
  dolfin::MPI::barrier();
  t = time() - t;
  if (dolfin::MPI::process_number() == 0)
    info("TIME (re-use preconditioner): %.5g", t);

  #else
  error("This benchmark requires PETSc.");
  #endif

  return 0;
}
