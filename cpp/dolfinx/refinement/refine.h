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
/// @brief Refine with markers, optionally redistributing, and
/// optionally calculating the parent-child relationships.
///
/// @note Using the default partitioner for a refined mesh, the refined
/// mesh will include ghosts cells (connected by facet) even if the
/// parent mesh is not ghosted.
///
/// @warning Passing `nullptr` for `partitioner`, refined cells will be
/// on the same process as the parent cell but the refined mesh will
/// **not** have ghosts cells. The possibility to not re-partition the
/// refined mesh and include ghost cells in the refined mesh will be
/// added in a future release.
///
/// @param[in] mesh Input mesh to be refined.
/// @param[in] edges Indices of the edges that should be split by this
/// refinement. If not provides, a uniform refinement is performed.
/// @param[in] partitioner Partitioner to be used for the refined mesh.
/// If not callable, refined cells will be on the same process as the
/// parent cell.
/// @param[in] option Control the computation of parent facets, parent
/// cells. If an option is unselected, an empty list is returned.
/// @return New mesh and optional parent cell index, parent facet
/// indices.
template <std::floating_point T>
std::tuple<mesh::Mesh<T>, std::optional<std::vector<std::int32_t>>,
           std::optional<std::vector<std::int8_t>>>
refine(const mesh::Mesh<T>& mesh,
       std::optional<std::span<const std::int32_t>> edges,
       const mesh::CellPartitionFunction& partitioner
       = mesh::create_cell_partitioner(mesh::GhostMode::none),
       Option option = Option::none)
{
  auto topology = mesh.topology();
  assert(topology);
  if (!mesh::is_simplex(topology->cell_type()))
    throw std::runtime_error("Refinement only defined for simplices");

  auto [cell_adj, new_vertex_coords, xshape, parent_cell, parent_facet]
      = (topology->cell_type() == mesh::CellType::interval)
            ? interval::compute_refinement_data(mesh, edges, option)
            : plaza::compute_refinement_data(mesh, edges, option);

  mesh::Mesh<T> refined_mesh = mesh::create_mesh(
      mesh.comm(), mesh.comm(), cell_adj.array(), mesh.geometry().cmap(),
      mesh.comm(), new_vertex_coords, xshape, partitioner);

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
