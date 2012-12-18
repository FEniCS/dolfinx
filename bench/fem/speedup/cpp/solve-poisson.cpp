// Copyright (C) 2009 Anders Logg
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
// First added:  2009-09-15
// Last changed: 2012-12-12
//
// Simple Poisson solver

#include <cstdlib>
#include <dolfin.h>
#include "Poisson.h"

using namespace dolfin;

int main(int argc, char* argv[])
{

  #ifdef HAS_PETSC

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
  UnitCubeMesh mesh(n, n, n);
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
  Function u(V);

  // Create preconditioner and linear solver
  //TrilinosPreconditioner pc("amg_ml");
  //EpetraKrylovSolver solver("gmres", pc);
  //PETScPreconditioner pc("amg_hypre");
  //PETScPreconditioner pc("amg_ml");
  //PETScKrylovSolver solver("gmres", pc);

  PETScLUSolver solver;

  // Assemble matrix and vector, and apply Dirichlet boundary conditions
  Matrix A;
  Vector b;
  assemble(A, a);
  assemble(b, L);
  bc.apply(A, b);

  // Solve linear system
  dolfin::MPI::barrier();
  double t = time();
  solver.solve(A, *u.vector(), b);
  dolfin::MPI::barrier();
  t = time() - t;
  if (dolfin::MPI::process_number() == 0)
    info("TIME (first time): %.5g", t);

  // Solve linear system (preconditioner assuming same non-zero pattern)
  solver.parameters("preconditioner")["same_nonzero_pattern"] = true;
  u.vector()->zero();
  dolfin::MPI::barrier();
  t = time();
  solver.solve(A, *u.vector(), b);
  dolfin::MPI::barrier();
  t = time() - t;
  if (dolfin::MPI::process_number() == 0)
    info("TIME (same nonzero pattern): %.5g", t);

  // Solve linear system (re-use preconditioner)
  solver.parameters("preconditioner")["reuse"] = true;
  u.vector()->zero();
  dolfin::MPI::barrier();
  t = time();
  solver.solve(A, *u.vector(), b);
  dolfin::MPI::barrier();
  t = time() - t;
  if (dolfin::MPI::process_number() == 0)
    info("TIME (re-use preconditioner): %.5g", t);

  #else
  error("This benchmark requires PETSc.");
  #endif

  return 0;
}
