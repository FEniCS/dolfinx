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
  UnitSquareMesh mesh(20,20);
  // Output cell distribution amongst processes
  File file("processes.pvd");

  MeshFunction<std::size_t> processes(mesh, mesh.topology().dim(), MPI::process_number());
  file << processes;
  
  // Mark all cells on process 0 for refinement
  MeshFunction<bool> marker(mesh, mesh.topology().dim(), (MPI::process_number() == 0));
  
  // Do refinement, but keep all new cells on parent process
  parameters["mesh_partitioner"] = "None";
  Mesh mesh2;
  mesh2 = refine(mesh, marker);
  
  processes = MeshFunction<std::size_t>(mesh2, mesh2.topology().dim(), MPI::process_number());
  file << processes;

  // try to find a repartitioning partitioner, and do the previous refinement again
  parameters["partitioning_approach"] = "REPARTITION";
  if(has_parmetis())
  {
    parameters["mesh_partitioner"] = "ParMETIS";
  }
  else if(has_trilinos())
  {
    parameters["mesh_partitioner"] = "Zoltan_PHG";
  }
  else
    parameters["mesh_partitioner"] = "SCOTCH";
    
  mesh2 = refine(mesh, marker);
  processes = MeshFunction<std::size_t>(mesh2, mesh2.topology().dim(), MPI::process_number());
  file << processes;
  
  return 0;
}
