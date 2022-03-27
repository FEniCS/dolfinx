// Copyright (C) 2014-2018 Chris Richardson
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <cstdint>
#include <dolfinx/graph/AdjacencyList.h>
#include <utility>
#include <vector>
#include <xtl/xspan.hpp>

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

/// Uniform refine, optionally redistributing and optionally
/// calculating the parent-child relation for facets (in 2D)
///
/// @param[in] mesh Input mesh to be refined
/// @param[in] redistribute Flag to call the mesh partitioner to
/// redistribute after refinement
/// @param[in] compute_facets Flag to save facet data for further use
/// @return New mesh
std::tuple<std::vector<std::int32_t>, std::vector<std::int8_t>, mesh::Mesh>
refine(const mesh::Mesh& mesh, bool redistribute, bool compute_facets);

/// Refine with markers, optionally redistributing.
///
/// @param[in] mesh Input mesh to be refined
/// @param[in] edges Indices of the edges that should be split by this
/// refinement
/// @param[in] redistribute Flag to call the Mesh Partitioner to
/// redistribute after refinement
/// @param[in] compute_facets Flag to save facet data for further use
/// @return optional refeinemtn data and New Mesh
std::tuple<std::vector<std::int32_t>, std::vector<std::int8_t>, mesh::Mesh>
refine(const mesh::Mesh& mesh, const xtl::span<const std::int32_t>& edges,
       bool redistribute, bool compute_facets);

/// Refine mesh returning new mesh data.
///
/// @param[in] mesh Input mesh to be refined
/// @param[in] compute_facets If true, returns list of facets for each new cell,
/// as local facet index of parent cell, or -1 if no corresponding facet.
/// @return New mesh data: cell topology, vertex coordinates and parent
/// cell index, and stored parent facet indices (if requested).
std::tuple<graph::AdjacencyList<std::int64_t>, xt::xtensor<double, 2>,
           std::vector<std::int32_t>, std::vector<std::int8_t>>
compute_refinement_data(const mesh::Mesh& mesh, bool compute_facets);

/// Refine with markers returning new mesh data.
///
/// @param[in] mesh Input mesh to be refined
/// @param[in] edges Indices of the edges that should be split by this
/// refinement
/// @param[in] compute_facets  If true, returns list of facets for each new
/// cell, as local facet index of parent cell, or -1 if no corresponding facet.
/// @return New mesh data: cell topology, vertex coordinates and parent
/// cell index, and stored parent facet indices (if requested).
std::tuple<graph::AdjacencyList<std::int64_t>, xt::xtensor<double, 2>,
           std::vector<std::int32_t>, std::vector<std::int8_t>>
compute_refinement_data(const mesh::Mesh& mesh,
                        const xtl::span<const std::int32_t>& edges,
                        bool compute_facets);

} // namespace plaza
} // namespace dolfinx::refinement
