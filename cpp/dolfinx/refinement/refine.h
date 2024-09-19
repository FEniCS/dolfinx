// Copyright (C) 2010-2024 Garth N. Wells and Paul T. Kühner
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "dolfinx/graph/AdjacencyList.h"
#include "dolfinx/mesh/Mesh.h"
#include "dolfinx/mesh/Topology.h"
#include "dolfinx/mesh/cell_types.h"
#include "dolfinx/mesh/utils.h"
#include "interval.h"
#include "plaza.h"
#include <algorithm>
#include <concepts>
#include <optional>
#include <utility>

namespace dolfinx::refinement
{

/// @brief Partitioner that maintains the regional destribution for a refined
/// mesh. Meaning, process local data is just maintained and no redistribution
/// happens.
///
/// @param[in] comm MPI Communicator
/// @param[in] nparts Number of partitions (input has no effect)
/// @param[in] cell_types Cell types in the mesh
/// @param[in] cells Lists of cells of each cell type.
/// @return Destination ranks for each cell on this process
graph::AdjacencyList<std::int32_t> maintain_coarse_partitioner(
    MPI_Comm comm, int nparts, const std::vector<mesh::CellType>& cell_types,
    const std::vector<std::span<const std::int64_t>>& cells);

/// @brief Refine with markers, optionally redistributing, and
/// optionally calculating the parent-child relationships.
///
/// @param[in] mesh Input mesh to be refined.
/// @param[in] edges Optional indices of the edges that should be split
/// by this refinement, if optional is not set, a uniform refinement is
/// performed, same behavior as passing a list of all indices.
/// @param[in] partitioner partitioner to be used for the refined mesh
/// @param[in] option Control the computation of parent facets, parent
/// cells. If an option is unselected, an empty list is returned.
/// @return New Mesh and optional parent cell index, parent facet
/// indices.
template <std::floating_point T>
std::tuple<mesh::Mesh<T>, std::optional<std::vector<std::int32_t>>,
           std::optional<std::vector<std::int8_t>>>
refine(const mesh::Mesh<T>& mesh,
       std::optional<std::span<const std::int32_t>> edges,
       mesh::CellPartitionFunction partitioner = maintain_coarse_partitioner,
       Option option = Option::none)
{
  auto topology = mesh.topology();
  assert(topology);
  if (!mesh::is_simplex(topology->cell_type()))
    throw std::runtime_error("Refinement only defined for simplices");

  bool oned = topology->cell_type() == mesh::CellType::interval;
  auto [cell_adj, new_vertex_coords, xshape, parent_cell, parent_facet]
      = oned ? interval::compute_refinement_data(mesh, edges, option)
             : plaza::compute_refinement_data(mesh, edges, option);

  mesh::Mesh<T> refined_mesh
      = mesh::create_mesh(mesh.comm(), mesh.comm(), std::move(cell_adj).array(),
                          mesh.geometry().cmap(), mesh.comm(),
                          std::move(new_vertex_coords), xshape, partitioner);

  // Report the number of refined cells
  const int D = topology->dim();
  const std::int64_t n0 = topology->index_map(D)->size_global();
  const std::int64_t n1 = refined_mesh.topology()->index_map(D)->size_global();
  spdlog::info(
      "Number of cells increased from {} to {} ({}% increase).", n0, n1,
      100.0 * (static_cast<double>(n1) / static_cast<double>(n0) - 1.0));

  return {std::move(refined_mesh), std::move(parent_cell),
          std::move(parent_facet)};
}

} // namespace dolfinx::refinement
