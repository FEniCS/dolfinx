// Copyright (C) 2015 Garth N. Wells
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

#include <vector>
#include <dolfin/fem/GenericDofMap.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/la/GenericMatrix.h>
#include <dolfin/la/PETScMatrix.h>
#include <dolfin/mesh/Edge.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/Vertex.h>
#include "DiscreteGradient.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
std::shared_ptr<GenericMatrix>
DiscreteOperators::build_gradient(const FunctionSpace& V0,
                                  const FunctionSpace& V1)
{
  // Get mesh
  dolfin_assert(V0.mesh());
  const Mesh& mesh = *(V0.mesh());

  // Check that mesh is the same for both function spaces
  dolfin_assert(V1.mesh());
  if (&mesh != V1.mesh().get())
  {
    dolfin_error("DiscreteGradient.cpp",
                 "compute discrete gradient operator",
                 "function spaces do not share the same mesh");
  }

  // Check that V0 is a (lowest-order) edge basis
  mesh.init(1);
  if (V0.dim() != mesh.size_global(1))
  {
    dolfin_error("DiscreteGradient.cpp",
                 "compute discrete gradient operator",
                 "function spaces is not a lowest-order edge space");
  }

  // Check that V1 is a linear nodal basis
  std::cout << "Test dim: " << V1.dim() << std::endl;
  if (V1.dim() != mesh.size_global(0))
  {
    dolfin_error("DiscreteGradient.cpp",
                 "compute discrete gradient operator",
                 "function space is not a linear nodal function space");
  }


  // Build tensor layout
  //const std::vector<std::size_t> dims = {mesh.size_global(D - 1),
  //                                       mesh.size_global(0)};
  //TensorLayout layout(mesh.mpi_comm(), dims, 0, 1, . . ., true);

  // Build ,aps from entities to local dof indices
  const std::vector<dolfin::la_index> edge_to_dof = V0.dofmap()->dofs(mesh, 1);
  const std::vector<dolfin::la_index> vertex_to_dof
    = V1.dofmap()->dofs(mesh, 0);

  // Build maps from local dof numbering to global
  std::vector<std::size_t> local_to_global_map0;
  std::vector<std::size_t> local_to_global_map1;
  V0.dofmap()->tabulate_local_to_global_dofs(local_to_global_map0);
  V1.dofmap()->tabulate_local_to_global_dofs(local_to_global_map1);

  // FIXME: This should not be PETSc-specific

  // Create matrix
  //auto G = std::make_shared<PETScMatrix>();
  //Mat _G = G->mat();
  Mat _G;

  // Get local ranges
  std::size_t num_local_rows = V0.dofmap()->local_dimension("owned");
  std::size_t num_local_cols = V1.dofmap()->local_dimension("owned");

  // Initialize matrix
  MatCreate(mesh.mpi_comm(), &_G);
  MatSetSizes(_G, num_local_rows, num_local_cols, PETSC_DECIDE, PETSC_DECIDE);
  MatSetType(_G, MATAIJ);
  MatSeqAIJSetPreallocation(_G, 2, NULL);
  MatMPIAIJSetPreallocation(_G, 2, NULL, 2, NULL);
  MatSetUp(_G);

  // Initialize edge -> vertex connections
  mesh.init(1, 0);

  // Build discrete gradient operator
  for (EdgeIterator edge(mesh); !edge.end(); ++edge)
  {
    dolfin::la_index row;
    dolfin::la_index cols[2];
    double values[2];

    row = local_to_global_map0[edge_to_dof[edge->index()]];

    Vertex v0(mesh, edge->entities(0)[0]);
    Vertex v1(mesh, edge->entities(0)[1]);

    cols[0] = local_to_global_map1[vertex_to_dof[v0.index()]];
    cols[1] = local_to_global_map1[vertex_to_dof[v1.index()]];
    if (v1.global_index() < v0.global_index())
    {
      values[0] =  1.0;
      values[1] = -1.0;
    }
    else
    {
      values[0] = -1.0;
      values[1] =  1.0;
    }
    MatSetValues(_G, 1, &row, 2, cols, values, INSERT_VALUES);
  }
  MatAssemblyBegin(_G, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(_G, MAT_FINAL_ASSEMBLY);

  return std::make_shared<PETScMatrix>(_G);
}
//-----------------------------------------------------------------------------
