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
  UnitSquareMesh mesh(20, 20);

  // Create MeshFunction to hold cell process rank
  CellFunction<std::size_t> processes0(mesh, MPI::process_number());

  // Output cell distribution to VTK file
  File file("processes.pvd");
  file << processes0;

  // Mark all cells on process 0 for refinement
  const CellFunction<bool> marker(mesh, (MPI::process_number() == 0));

  // Refine mesh, but keep all new cells on parent process
  Mesh mesh0 = refine(mesh, marker, false);

  // Create MeshFunction to hold cell process rank
  const CellFunction<std::size_t> processes1(mesh0, MPI::process_number());
  file << processes1;

  // Try to find a repartitioning partitioner,
  parameters["partitioning_approach"] = "REPARTITION";
  if (has_parmetis())
    parameters["mesh_partitioner"] = "ParMETIS";
  else if (has_trilinos())
    parameters["mesh_partitioner"] = "Zoltan_PHG";
  else
    parameters["mesh_partitioner"] = "SCOTCH";

  // Refine mesh, but this time repartition the mesh after refinement
  Mesh mesh1 = refine(mesh, marker, true);

  // Create MeshFunction to hold cell process rank
  CellFunction<std::size_t> processes2(mesh1, MPI::process_number());
  file << processes2;

  return 0;
}
