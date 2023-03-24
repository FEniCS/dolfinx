// Copyright (C) 2014-2018 Chris Richardson
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <cstdint>
#include <dolfinx/graph/AdjacencyList.h>
#include <span>
#include <tuple>
#include <utility>
#include <vector>

#pragma once

namespace dolfinx::mesh
{
template <typename T>
class Mesh;
} // namespace dolfinx::mesh

/// @brief Plaza mesh refinement.
///
/// Functions for the refinement method described in Plaza and Carey
/// "Local refinement of simplicial grids based on the skeleton",
/// Applied Numerical Mathematics 32 (2000), 195-218.
namespace dolfinx::refinement::plaza
{
/// @brief Options for data to compute during mesh refinement.
enum class Option : int
{
  none = 0, /*!< No extra data */
  parent_cell
  = 1,      /*!< Compute list with the parent cell index for each new cell  */
  parent_facet
  = 2, /*!< Compute list of the cell-local facet indices in the parent cell of
          each facet in each new cell (or -1 if no match) */
  parent_cell_and_facet = 3 /*!< Both cell and facet parent data */
};

/// @brief Uniform refine, optionally redistributing and optionally
/// calculating the parent-child relationships`.
///
/// @param[in] mesh Input mesh to be refined
/// @param[in] redistribute Flag to call the mesh partitioner to
/// redistribute after refinement
/// @param[in] option Control the computation of parent facets, parent
/// cells. If an option is unselected, an empty list is returned.
/// @return Refined mesh and optional parent cell index, parent facet
/// indices
std::tuple<mesh::Mesh<double>, std::vector<std::int32_t>,
           std::vector<std::int8_t>>
refine(const mesh::Mesh<double>& mesh, bool redistribute, Option option);

/// @brief Refine with markers, optionally redistributing, and
/// optionally calculating the parent-child relationships.
///
/// @param[in] mesh Input mesh to be refined
/// @param[in] edges Indices of the edges that should be split by this
/// refinement
/// @param[in] redistribute Flag to call the Mesh Partitioner to
/// redistribute after refinement
/// @param[in] option Control the computation of parent facets, parent
/// cells. If an option is unselected, an empty list is returned.
/// @return New Mesh and optional parent cell index, parent facet indices
std::tuple<mesh::Mesh<double>, std::vector<std::int32_t>,
           std::vector<std::int8_t>>
refine(const mesh::Mesh<double>& mesh, std::span<const std::int32_t> edges,
       bool redistribute, Option option);

/// @brief Refine mesh returning new mesh data.
///
/// @param[in] mesh Input mesh to be refined
/// @param[in] option Control computation of parent facets and parent
/// cells. If an option is unselected, an empty list is returned.
/// @return New mesh data: cell topology, vertex coordinates, vertex
/// coordinates shape, and optional parent cell index, and parent facet
/// indices.
std::tuple<graph::AdjacencyList<std::int64_t>, std::vector<double>,
           std::array<std::size_t, 2>, std::vector<std::int32_t>,
           std::vector<std::int8_t>>
compute_refinement_data(const mesh::Mesh<double>& mesh, Option option);

/// Refine with markers returning new mesh data.
///
/// @param[in] mesh Input mesh to be refined
/// @param[in] edges Indices of the edges that should be split by this
/// refinement
/// @param[in] option Control the computation of parent facets, parent
/// cells. If an option is unselected, an empty list is returned.
/// @return New mesh data: cell topology, vertex coordinates and parent
/// cell index, and stored parent facet indices (if requested).
std::tuple<graph::AdjacencyList<std::int64_t>, std::vector<double>,
           std::array<std::size_t, 2>, std::vector<std::int32_t>,
           std::vector<std::int8_t>>
compute_refinement_data(const mesh::Mesh<double>& mesh,
                        std::span<const std::int32_t> edges, Option option);
} // namespace dolfinx::refinement::plaza
