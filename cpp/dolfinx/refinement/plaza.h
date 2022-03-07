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
/// @return New mesh
mesh::Mesh refine(const mesh::Mesh& mesh, bool redistribute);

/// Refine with markers, optionally redistributing.
///
/// @param[in] mesh Input mesh to be refined
/// @param[in] edges Indices of the edges that should be split by this
/// refinement
/// @param[in] redistribute Flag to call the Mesh Partitioner to
/// redistribute after refinement
/// @return New Mesh
mesh::Mesh refine(const mesh::Mesh& mesh,
                  const xtl::span<const std::int32_t>& edges,
                  bool redistribute);

/// Refine mesh returning new mesh data.
///
/// @param[in] mesh Input mesh to be refined
/// @return New mesh data: cell topology, vertex coordinates and parent
/// cell index
std::tuple<graph::AdjacencyList<std::int64_t>, xt::xtensor<double, 2>,
           std::vector<std::int32_t>>
compute_refinement_data(const mesh::Mesh& mesh);

/// Refine with markers returning new mesh data.
///
/// @param[in] mesh Input mesh to be refined
/// @param[in] edges Indices of the edges that should be split by this
/// refinement
/// @return New mesh data: cell topology, vertex coordinates and parent
/// cell index
std::tuple<graph::AdjacencyList<std::int64_t>, xt::xtensor<double, 2>,
           std::vector<std::int32_t>>
compute_refinement_data(const mesh::Mesh& mesh,
                        const xtl::span<const std::int32_t>& edges);

} // namespace plaza
} // namespace dolfinx::refinement
