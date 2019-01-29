// Copyright (C) 2015 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "DiscreteOperators.h"
#include <array>
#include <dolfin/common/IndexMap.h>
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
using namespace dolfin::fem;

//-----------------------------------------------------------------------------
la::PETScMatrix
DiscreteOperators::build_gradient(const function::FunctionSpace& V0,
                                  const function::FunctionSpace& V1)
{
  // TODO: This function would be significantly simplified if it was
  // easier to build matrix sparsity patterns.

  // Get mesh
  assert(V0.mesh());
  const mesh::Mesh& mesh = *(V0.mesh());

  // Check that mesh is the same for both function spaces
  assert(V1.mesh());
  if (&mesh != V1.mesh().get())
  {
    throw std::runtime_error(
        "Ccompute discrete gradient operator. Function spaces "
        "do not share the same mesh");
  }

  // Check that V0 is a (lowest-order) edge basis
  mesh.init(1);
  if (V0.dim() != mesh.num_entities_global(1))
  {
    throw std::runtime_error(
        "Cannot compute discrete gradient operator. Function "
        "spaces is not a lowest-order edge space");
  }

  // Check that V1 is a linear nodal basis
  if (V1.dim() != mesh.num_entities_global(0))
  {
    throw std::runtime_error(
        "Cannot compute discrete gradient operator. Function "
        "space is not a linear nodal function space");
  }

  // Build maps from entities to local dof indices
  const Eigen::Array<PetscInt, Eigen::Dynamic, 1> edge_to_dof
      = V0.dofmap()->dofs(mesh, 1);
  const Eigen::Array<PetscInt, Eigen::Dynamic, 1> vertex_to_dof
      = V1.dofmap()->dofs(mesh, 0);

  // Build maps from local dof numbering to global
  Eigen::Array<std::size_t, Eigen::Dynamic, 1> local_to_global_map0
      = V0.dofmap()->tabulate_local_to_global_dofs();
  Eigen::Array<std::size_t, Eigen::Dynamic, 1> local_to_global_map1
      = V1.dofmap()->tabulate_local_to_global_dofs();

  // Initialize edge -> vertex connections
  mesh.init(1, 0);

  // Copy index maps from dofmaps
  std::array<std::shared_ptr<const common::IndexMap>, 2> index_maps
      = {{V0.dofmap()->index_map(), V1.dofmap()->index_map()}};
  std::vector<std::array<std::int64_t, 2>> local_range
      = {V0.dofmap()->ownership_range(), V1.dofmap()->ownership_range()};

  // Initialise sparsity pattern
  la::SparsityPattern pattern(mesh.mpi_comm(), index_maps);

  // Build sparsity pattern
  std::vector<PetscInt> rows;
  std::vector<PetscInt> cols;
  for (auto& edge : mesh::MeshRange<mesh::Edge>(mesh))
  {
    // Row index (global indices)
    const std::int64_t row = local_to_global_map0[edge_to_dof[edge.index()]];
    rows.push_back(row);

    if (row >= local_range[0][0] and row < local_range[0][1])
    {
      // Column indices (global indices)
      const mesh::Vertex v0(mesh, edge.entities(0)[0]);
      const mesh::Vertex v1(mesh, edge.entities(0)[1]);
      std::size_t col0 = local_to_global_map1[vertex_to_dof[v0.index()]];
      std::size_t col1 = local_to_global_map1[vertex_to_dof[v1.index()]];
      cols.push_back(col0);
      cols.push_back(col1);
    }
  }

  Eigen::Map<const EigenArrayXpetscint> _rows(rows.data(), rows.size());
  Eigen::Map<const EigenArrayXpetscint> _cols(cols.data(), cols.size());
  pattern.insert_global(_rows, _cols);
  pattern.assemble();

  // Create matrix
  la::PETScMatrix A(mesh.mpi_comm(), pattern);

  // Build discrete gradient operator/matrix
  for (auto& edge : mesh::MeshRange<mesh::Edge>(mesh))
  {
    PetscInt row;
    PetscInt cols[2];
    PetscScalar values[2];

    row = local_to_global_map0[edge_to_dof[edge.index()]];

    mesh::Vertex v0(mesh, edge.entities(0)[0]);
    mesh::Vertex v1(mesh, edge.entities(0)[1]);

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
    A.set(values, 1, &row, 2, cols);
  }

  // Finalise matrix
  A.apply(la::PETScMatrix::AssemblyType::FINAL);

  return A;
}
//-----------------------------------------------------------------------------
