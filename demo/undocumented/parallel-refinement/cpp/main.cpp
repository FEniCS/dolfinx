// Copyright (C) 2013 Chris Richardson
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
// First added:  2013-04-26
// Last changed: 2013-04-26

#include <dolfin.h>

using namespace dolfin;

int main()
{
  // Create mesh
  auto mesh = std::make_shared<UnitSquareMesh>(20, 20);

  // Create MeshFunction to hold cell process rank
  CellFunction<std::size_t>
    processes0(mesh, dolfin::MPI::rank(mesh->mpi_comm()));

  // Output cell distribution to VTK file
  File file("processes.pvd");
  file << processes0;

  // Mark all cells on process 0 for refinement
  const CellFunction<bool>
    marker(mesh, (dolfin::MPI::rank(mesh->mpi_comm()) == 0));

  // Refine mesh, but keep all new cells on parent process
  auto mesh0 = std::make_shared<Mesh>(refine(*mesh, marker, false));

  // Create MeshFunction to hold cell process rank
  const CellFunction<std::size_t>
    processes1(mesh0, dolfin::MPI::rank(mesh->mpi_comm()));
  file << processes1;

  // Refine mesh, but this time repartition the mesh after refinement
  auto mesh1 = std::make_shared<Mesh>(refine(*mesh, marker, false));

  // Create MeshFunction to hold cell process rank
  CellFunction<std::size_t>
    processes2(mesh1, dolfin::MPI::rank(mesh->mpi_comm()));
  file << processes2;

  return 0;
}
