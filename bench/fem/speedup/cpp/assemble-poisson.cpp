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
// Simple Poisson assembler

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
  auto mesh = std::make_shared<UnitCubeMesh>(n, n, n);
  auto V = std::make_shared<Poisson::FunctionSpace>(mesh);

  const MPI_Comm comm = mesh->mpi_comm();

  // Assemble matrix
  Poisson::BilinearForm a(V, V);
  Matrix A;
  dolfin::MPI::barrier(comm);
  double t = time();
  Assembler assembler;
  assembler.assemble(A, a);
  dolfin::MPI::barrier(comm);
  t = time() - t;

  // Report timing
  if (dolfin::MPI::rank(comm) == 0)
    info("TIME (first assembly): %.5g", t);

  // Re-assemble matrix
  dolfin::MPI::barrier(comm);
  t = time();
  assembler.assemble(A, a);
  dolfin::MPI::barrier(comm);
  t = time() - t;

  // Report timing
  if (dolfin::MPI::rank(comm) == 0)
    info("TIME (second assembly): %.5g", t);

  return 0;
}
