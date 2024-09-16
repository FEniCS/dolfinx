// Copyright (C) 2010-2023 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

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
namespace impl
{
/// @brief create the refined mesh by optionally redistributing it
template <std::floating_point T>
mesh::Mesh<T>
create_refined_mesh(const mesh::Mesh<T>& mesh,
                    const graph::AdjacencyList<std::int64_t>& cell_adj,
                    const std::vector<T>& new_vertex_coords,
                    std::array<std::size_t, 2> xshape, bool redistribute,
                    mesh::GhostMode ghost_mode)
{
  if (dolfinx::MPI::size(mesh.comm()) == 1)
  {
    // No parallel construction necessary, i.e. redistribute also has no
    // effect
    return mesh::create_mesh(mesh.comm(), cell_adj.array(),
                             mesh.geometry().cmap(), new_vertex_coords, xshape,
                             ghost_mode);
  }
  else
  {
    // Let partition handle the parallel construction of the mesh
    return partition<T>(mesh, cell_adj, new_vertex_coords, xshape, redistribute,
                        ghost_mode);
  }
}
} // namespace impl

/// @brief Refine with markers, optionally redistributing, and
/// optionally calculating the parent-child relationships.
///
/// @param[in] mesh Input mesh to be refined
/// @param[in] edges Optional indices of the edges that should be split
/// by this refinement, if optional is not set, a uniform refinement is
/// performed, same behavior as passing a list of all indices.
/// @param[in] redistribute Flag to call the Mesh Partitioner to
/// redistribute after refinement.
/// @param[in] ghost_mode Ghost mode of the refined mesh.
/// @param[in] option Control the computation of parent facets, parent
/// cells. If an option is unselected, an empty list is returned.
/// @return New Mesh and optional parent cell index, parent facet
/// indices.
template <std::floating_point T>
std::tuple<mesh::Mesh<T>, std::optional<std::vector<std::int32_t>>,
           std::optional<std::vector<std::int8_t>>>
refine(const mesh::Mesh<T>& mesh,
       std::optional<std::span<const std::int32_t>> edges, bool redistribute,
       mesh::GhostMode ghost_mode = mesh::GhostMode::shared_facet,
       Option option = Option::none)
{
  auto topology = mesh.topology();
  assert(topology);

  mesh::CellType cell_t = topology->cell_type();
  if (!mesh::is_simplex(cell_t))
    throw std::runtime_error("Refinement only defined for simplices");
  bool oned = topology->cell_type() == mesh::CellType::interval;
  auto [cell_adj, new_vertex_coords, xshape, parent_cell, parent_facet]
      = oned ? interval::compute_refinement_data(mesh, edges, option)
             : plaza::compute_refinement_data(mesh, edges, option);

  mesh::Mesh<T> refined_mesh = impl::create_refined_mesh(
      mesh, std::move(cell_adj), std::move(new_vertex_coords), xshape,
      redistribute, ghost_mode);

  // Report the number of refined cellse
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
