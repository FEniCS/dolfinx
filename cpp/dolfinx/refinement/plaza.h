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
class Mesh;
template <typename T>
class MeshTags;
} // namespace dolfinx::mesh

namespace dolfinx::refinement
{

/// Function in this namespace implement the refinement method described
/// in Plaza and Carey "Local refinement of simplicial grids based on
/// the skeleton" (Applied Numerical Mathematics 32 (2000) 195-218).
namespace plaza
{

/// Selection of options when refining a Mesh. `parent_cell` will output a list
/// containing the local parent cell index for each new cell, `parent_facet`
/// will output a list of the cell-local facet indices in the parent cell of
/// each facet in each new cell (or -1 if no match). `parent_cell_and_facet`
/// will output both datasets.
enum class RefinementOptions : int
{
  none = 0,
  parent_cell = 1,
  parent_facet = 2,
  parent_cell_and_facet = 3
};

/// Uniform refine, optionally redistributing and optionally
/// calculating the parent-child relationships, selected by `RefinementOptions`.
///
/// @param[in] mesh Input mesh to be refined
/// @param[in] redistribute Flag to call the mesh partitioner to
/// redistribute after refinement
/// @param[in] options RefinementOptions enum to choose the computation of
/// parent facets, parent cells. If an option is unselected, an empty list is
/// returned.
/// @return New Mesh and optional parent cell index, parent facet indices
std::tuple<mesh::Mesh, std::vector<std::int32_t>, std::vector<std::int8_t>>
refine(const mesh::Mesh& mesh, bool redistribute, RefinementOptions options);

/// Refine with markers, optionally redistributing, and optionally
/// calculating the parent-child relationships, selected by `RefinementOptions`.
///
/// @param[in] mesh Input mesh to be refined
/// @param[in] edges Indices of the edges that should be split by this
/// refinement
/// @param[in] redistribute Flag to call the Mesh Partitioner to
/// redistribute after refinement
/// @param[in] options RefinementOptions enum to choose the computation of
/// parent facets, parent cells. If an option is unselected, an empty list is
/// returned.
/// @return New Mesh and optional parent cell index, parent facet indices
std::tuple<mesh::Mesh, std::vector<std::int32_t>, std::vector<std::int8_t>>
refine(const mesh::Mesh& mesh, std::span<const std::int32_t> edges,
       bool redistribute, RefinementOptions options);

/// Refine mesh returning new mesh data.
///
/// @param[in] mesh Input mesh to be refined
/// @param[in] options RefinementOptions enum to choose the computation of
/// parent facets, parent cells. If an option is unselected, an empty list is
/// returned.
/// @return New mesh data: cell topology, vertex coordinates, vertex
/// coordinates shape,  and optional parent cell index, and parent facet
/// indices.
std::tuple<graph::AdjacencyList<std::int64_t>, std::vector<double>,
           std::array<std::size_t, 2>, std::vector<std::int32_t>,
           std::vector<std::int8_t>>
compute_refinement_data(const mesh::Mesh& mesh, RefinementOptions options);

/// Refine with markers returning new mesh data.
///
/// @param[in] mesh Input mesh to be refined
/// @param[in] edges Indices of the edges that should be split by this
/// refinement
/// @param[in] options RefinementOptions enum to choose the computation of
/// parent facets, parent cells. If an option is unselected, an empty list is
/// returned.
/// @return New mesh data: cell topology, vertex coordinates and parent
/// cell index, and stored parent facet indices (if requested).
std::tuple<graph::AdjacencyList<std::int64_t>, std::vector<double>,
           std::array<std::size_t, 2>, std::vector<std::int32_t>,
           std::vector<std::int8_t>>
compute_refinement_data(const mesh::Mesh& mesh,
                        std::span<const std::int32_t> edges,
                        RefinementOptions options);

} // namespace plaza
} // namespace dolfinx::refinement
