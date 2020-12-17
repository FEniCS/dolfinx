// Copyright (C) 2015-2020 Garth N. Wells, Jørgen S. Dokken
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "discreteoperators.h"
#include <array>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/la/PETScMatrix.h>
#include <dolfinx/la/SparsityPattern.h>
#include <dolfinx/mesh/Mesh.h>
#include <vector>

using namespace dolfinx;

//-----------------------------------------------------------------------------
la::PETScMatrix fem::build_discrete_gradient(const fem::FunctionSpace& V0,
                                             const fem::FunctionSpace& V1)
{
  // Get mesh
  std::shared_ptr<const mesh::Mesh> mesh = V0.mesh();
  assert(mesh);

  // Check that mesh is the same for both function spaces
  assert(V1.mesh());
  if (mesh != V1.mesh())
  {
    throw std::runtime_error("Compute discrete gradient operator. Function "
                             "spaces do not share the same mesh");
  }

  // Check that V0 is a (lowest-order) edge basis
  mesh->topology_mutable().create_entities(1);
  std::int64_t num_edges_global = mesh->topology().index_map(1)->size_global();
  if (V0.dim() != num_edges_global)
  {
    throw std::runtime_error(
        "Cannot compute discrete gradient operator. Function "
        "spaces is not a lowest-order edge space");
  }

  // Check that V1 is a linear nodal basis
  const std::int64_t num_vertices_global
      = mesh->topology().index_map(0)->size_global();
  if (V1.dim() != num_vertices_global)
  {
    throw std::runtime_error(
        "Cannot compute discrete gradient operator. Function "
        "space is not a linear nodal function space");
  }

  // Build maps from entities to local dof indices
  std::shared_ptr<const dolfinx::fem::ElementDofLayout> layout0
      = V0.dofmap()->element_dof_layout;
  std::shared_ptr<const dolfinx::fem::ElementDofLayout> layout1
      = V1.dofmap()->element_dof_layout;

  // Copy index maps from dofmaps
  std::array<std::shared_ptr<const common::IndexMap>, 2> index_maps
      = {{V0.dofmap()->index_map, V1.dofmap()->index_map}};
  std::array<int, 2> block_sizes
      = {V0.dofmap()->index_map_bs(), V1.dofmap()->index_map_bs()};
  std::vector<std::array<std::int64_t, 2>> local_range
      = {index_maps[0]->local_range(), index_maps[1]->local_range()};
  assert(block_sizes[0] == block_sizes[1]);

  // Initialise sparsity pattern
  la::SparsityPattern pattern(mesh->mpi_comm(), index_maps, block_sizes);

  // Initialize required connectivities
  const int tdim = mesh->topology().dim();
  mesh->topology_mutable().create_connectivity(1, 0);
  auto e_to_v = mesh->topology().connectivity(1, 0);
  mesh->topology_mutable().create_connectivity(tdim, 1);
  auto c_to_e = mesh->topology().connectivity(tdim, 1);
  mesh->topology_mutable().create_connectivity(1, tdim);
  auto e_to_c = mesh->topology().connectivity(1, tdim);
  mesh->topology_mutable().create_connectivity(tdim, 0);
  auto c_to_v = mesh->topology().connectivity(tdim, 0);

  // Build sparsity pattern
  std::vector<std::int32_t> rows;
  std::vector<std::int32_t> cols;
  const std::int32_t num_edges = mesh->topology().index_map(1)->size_local()
                                 + mesh->topology().index_map(1)->num_ghosts();
  const std::shared_ptr<const fem::DofMap> dofmap0 = V0.dofmap();
  assert(dofmap0);
  for (std::int32_t e = 0; e < num_edges; ++e)
  {
    // Find local index of edge in one of the cells it is part of
    auto cells = e_to_c->links(e);
    assert(cells.size() > 0);
    const std::int32_t cell = cells[0];
    auto edges = c_to_e->links(cell);
    const auto* it = std::find(edges.data(), edges.data() + edges.rows(), e);
    assert(it != (edges.data() + edges.rows()));
    const int local_edge = std::distance(edges.data(), it);

    // Find the dofs located on the edge
    auto dofs0 = dofmap0->cell_dofs(cell);
    auto local_dofs = layout0->entity_dofs(1, local_edge);
    assert(local_dofs.size() == 1);
    rows.push_back(dofs0[local_dofs[0]]);
    auto vertices = e_to_v->links(e);
    assert(vertices.size() == 2);
    auto cell_vertices = c_to_v->links(cell);

    // Find local index of each of the vertices and map to local dof
    for (std::int32_t i = 0; i < 2; ++i)
    {
      const auto* it
          = std::find(cell_vertices.data(),
                      cell_vertices.data() + cell_vertices.rows(), vertices[i]);
      assert(it != (cell_vertices.data() + cell_vertices.rows()));
      const int local_vertex = std::distance(cell_vertices.data(), it);
      auto local_v_dofs = layout1->entity_dofs(0, local_vertex);
      assert(local_v_dofs.size() == 1);
      auto dofs1 = V1.dofmap()->cell_dofs(cell);
      cols.push_back(dofs1[local_v_dofs[0]]);
    }
  }

  Eigen::Map<const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>> _rows(
      rows.data(), rows.size());
  Eigen::Map<const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>> _cols(
      cols.data(), cols.size());
  pattern.insert(_rows, _cols);
  pattern.assemble();

  // Create matrix
  la::PETScMatrix A(mesh->mpi_comm(), pattern);
  Mat _A = A.mat();

  // Build discrete gradient operator/matrix
  const std::vector<std::int64_t>& global_indices
      = mesh->topology().index_map(0)->global_indices();
  std::array<PetscScalar, 2> Ae;
  for (std::int32_t e = 0; e < num_edges; ++e)
  {
    std::vector<std::int32_t> cols;

    // Find local index of edge in one of the cells it is part of
    auto cells = e_to_c->links(e);
    assert(cells.size() > 0);
    const std::int32_t cell = cells[0];
    auto edges = c_to_e->links(cell);
    const auto* it = std::find(edges.data(), edges.data() + edges.rows(), e);
    assert(it != (edges.data() + edges.rows()));
    const int local_edge = std::distance(edges.data(), it);

    // Find the dofs located on the edge
    auto dofs0 = dofmap0->cell_dofs(cell);

    // FIXME: avoid this expensive call
    const Eigen::Array<int, Eigen::Dynamic, 1> local_dofs
        = layout0->entity_dofs(1, local_edge);
    assert(local_dofs.size() == 1);
    // FIXME: avoid this expensive call
    std::int32_t row = dofs0[local_dofs[0]];

    auto vertices = e_to_v->links(e);
    assert(vertices.size() == 2);
    auto cell_vertices = c_to_v->links(cell);

    // Find local index of each of the vertices and map to local dof
    for (std::int32_t i = 0; i < 2; ++i)
    {
      const auto* it
          = std::find(cell_vertices.data(),
                      cell_vertices.data() + cell_vertices.rows(), vertices[i]);
      assert(it != (cell_vertices.data() + cell_vertices.rows()));
      const int local_vertex = std::distance(cell_vertices.data(), it);

      // FIXME: avoid this expensive call
      const Eigen::Array<int, Eigen::Dynamic, 1> local_v_dofs
          = layout1->entity_dofs(0, local_vertex);
      assert(local_v_dofs.size() == 1);
      auto dofs1 = V1.dofmap()->cell_dofs(cell);
      cols.push_back(dofs1[local_v_dofs[0]]);
    }

    if (global_indices[vertices[1]] < global_indices[vertices[0]])
    {
      Ae[0] = 1;
      Ae[1] = -1;
    }
    else
    {
      Ae[0] = -1;
      Ae[1] = 1;
    }

    MatSetValuesLocal(_A, 1, &row, 2, cols.data(), Ae.data(), INSERT_VALUES);
  }

  // Finalise matrix
  A.apply(la::PETScMatrix::AssemblyType::FINAL);

  return A;
}
//-----------------------------------------------------------------------------
