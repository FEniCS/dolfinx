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
  parameters["linear_algebra_backend"] = "PETSc";

  // Parse command-line arguments
  if (argc != 2)
  {
    info("Usage: solve-poisson n");
    return 1;
  }
  int n = atoi(argv[1]);

  // Create mesh and function space
  auto mesh = std::make_shared<UnitCubeMesh>(n, n, n);
  auto V = std::make_shared<const Poisson::FunctionSpace>(mesh);

  // MPI communicator
  const MPI_Comm comm = mesh->mpi_comm();

  // Define boundary condition
  auto u0 = std::make_shared<const Constant>(0.0);
  auto boundary = std::make_shared<const DomainBoundary>();
  DirichletBC bc(V, u0, boundary);

  // Define variational problem
  Poisson::BilinearForm a(V, V);
  Poisson::LinearForm L(V);
  auto f= std::make_shared<Constant>(1.0);
  L.f = f;
  Function u(V);

  // Create preconditioner and linear solver
  //TrilinosPreconditioner pc("amg_ml");
  //PETScPreconditioner pc("amg_hypre");
  //PETScPreconditioner pc("amg_ml");
  //PETScKrylovSolver solver("gmres", pc);

  // Pick solver; UMFPACK runs out of memory even on 24GB RAM machine
  std::string method = "lu";
  if (has_lu_solver_method("mumps"))
    method = "mumps";
  else if (has_lu_solver_method("superlu_dist"))
    method = "superlu_dist";
  PETScLUSolver solver(method);

  // Assemble matrix and vector, and apply Dirichlet boundary conditions
  Matrix A;
  Vector b;
  assemble(A, a);
  assemble(b, L);
  bc.apply(A, b);

  // Solve linear system
  dolfin::MPI::barrier(comm);
  double t = time();
  solver.solve(A, *u.vector(), b);
  dolfin::MPI::barrier(comm);
  t = time() - t;
  if (dolfin::MPI::rank(comm) == 0)
    info("TIME (first time): %.5g", t);

  // Solve linear system (preconditioner assuming same non-zero pattern)
  if (solver.parameters.has_key("preconditioner"))
      solver.parameters("preconditioner")["same_nonzero_pattern"] = true;
  u.vector()->zero();
  dolfin::MPI::barrier(comm);
  t = time();
  solver.solve(A, *u.vector(), b);
  dolfin::MPI::barrier(comm);
  t = time() - t;
  if (dolfin::MPI::rank(comm) == 0)
    info("TIME (same nonzero pattern): %.5g", t);

  // Solve linear system (re-use preconditioner)
  if (solver.parameters.has_key("preconditioner"))
    solver.parameters("preconditioner")["reuse"] = true;
  u.vector()->zero();
  dolfin::MPI::barrier(comm);
  t = time();
  solver.solve(A, *u.vector(), b);
  dolfin::MPI::barrier(comm);
  t = time() - t;
  if (dolfin::MPI::rank(comm) == 0)
    info("TIME (re-use preconditioner): %.5g", t);

  #else
  error("This benchmark requires PETSc.");
  #endif

  return 0;
}
