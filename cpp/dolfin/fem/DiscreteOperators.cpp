// Copyright (C) 2015 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "DiscreteOperators.h"
#include <array>
#include <dolfin/common/ArrayView.h>
#include <dolfin/fem/GenericDofMap.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/la/PETScMatrix.h>
#include <dolfin/la/SparsityPattern.h>
#include <dolfin/mesh/Edge.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshIterator.h>
#include <dolfin/mesh/Vertex.h>
#include <vector>

using namespace dolfin;

//-----------------------------------------------------------------------------
std::shared_ptr<PETScMatrix>
DiscreteOperators::build_gradient(const FunctionSpace& V0,
                                  const FunctionSpace& V1)
{
  // TODO: This function would be significantly simplified if it was
  // easier to build matrix sparsity patterns.

  // Get mesh
  dolfin_assert(V0.mesh());
  const Mesh& mesh = *(V0.mesh());

  // Check that mesh is the same for both function spaces
  dolfin_assert(V1.mesh());
  if (&mesh != V1.mesh().get())
  {
    dolfin_error("DiscreteGradient.cpp", "compute discrete gradient operator",
                 "function spaces do not share the same mesh");
  }

  // Check that V0 is a (lowest-order) edge basis
  mesh.init(1);
  if (V0.dim() != mesh.num_entities_global(1))
  {
    dolfin_error("DiscreteGradient.cpp", "compute discrete gradient operator",
                 "function spaces is not a lowest-order edge space");
  }

  // Check that V1 is a linear nodal basis
  if (V1.dim() != mesh.num_entities_global(0))
  {
    dolfin_error("DiscreteGradient.cpp", "compute discrete gradient operator",
                 "function space is not a linear nodal function space");
  }

  // Build maps from entities to local dof indices
  const std::vector<dolfin::la_index_t> edge_to_dof
      = V0.dofmap()->dofs(mesh, 1);
  const std::vector<dolfin::la_index_t> vertex_to_dof
      = V1.dofmap()->dofs(mesh, 0);

  // Build maps from local dof numbering to global
  std::vector<std::size_t> local_to_global_map0;
  std::vector<std::size_t> local_to_global_map1;
  V0.dofmap()->tabulate_local_to_global_dofs(local_to_global_map0);
  V1.dofmap()->tabulate_local_to_global_dofs(local_to_global_map1);

  // Declare matrix
  auto A = std::make_shared<PETScMatrix>(mesh.mpi_comm());

  // Initialize edge -> vertex connections
  mesh.init(1, 0);

  // Copy index maps from dofmaps
  std::array<std::shared_ptr<const IndexMap>, 2> index_maps
      = {{V0.dofmap()->index_map(), V1.dofmap()->index_map()}};
  std::vector<std::array<std::int64_t, 2>> local_range
      = {V0.dofmap()->ownership_range(), V1.dofmap()->ownership_range()};

  // Initialise sparsity pattern
  SparsityPattern pattern(mesh.mpi_comm(), index_maps, 0);

  // Build sparsity pattern
  std::vector<dolfin::la_index_t> rows;
  std::vector<dolfin::la_index_t> cols;
  for (auto& edge : MeshRange<Edge>(mesh))
  {
    // Row index (global indices)
    const std::int64_t row = local_to_global_map0[edge_to_dof[edge.index()]];
    rows.push_back(row);

    if (row >= local_range[0][0] and row < local_range[0][1])
    {
      // Column indices (global indices)
      const Vertex v0(mesh, edge.entities(0)[0]);
      const Vertex v1(mesh, edge.entities(0)[1]);
      std::size_t col0 = local_to_global_map1[vertex_to_dof[v0.index()]];
      std::size_t col1 = local_to_global_map1[vertex_to_dof[v1.index()]];
      cols.push_back(col0);
      cols.push_back(col1);
    }
  }

  const std::array<common::ArrayView<const dolfin::la_index_t>, 2> entries
      = {{common::ArrayView<const dolfin::la_index_t>(rows.size(), rows.data()),
          common::ArrayView<const dolfin::la_index_t>(cols.size(), cols.data())}};
  pattern.insert_global(entries);
  pattern.apply();

  // Initialise matrix
  A->init(pattern);

  // Build discrete gradient operator/matrix
  for (auto& edge : MeshRange<Edge>(mesh))
  {
    dolfin::la_index_t row;
    dolfin::la_index_t cols[2];
    double values[2];

    row = local_to_global_map0[edge_to_dof[edge.index()]];

    Vertex v0(mesh, edge.entities(0)[0]);
    Vertex v1(mesh, edge.entities(0)[1]);

    cols[0] = local_to_global_map1[vertex_to_dof[v0.index()]];
    cols[1] = local_to_global_map1[vertex_to_dof[v1.index()]];
    if (v1.global_index() < v0.global_index())
    {
      values[0] = 1.0;
      values[1] = -1.0;
    }
    else
    {
      values[0] = -1.0;
      values[1] = 1.0;
    }

    // Set values in matrix
    A->set(values, 1, &row, 2, cols);
  }

  // Finalise matrix
  A->apply(PETScMatrix::AssemblyType::FINAL);

  return A;
}
//-----------------------------------------------------------------------------
