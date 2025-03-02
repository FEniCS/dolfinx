// Copyright (C) 2012-2024 Chris N. Richardson, Jørgen S. Dokken
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <array>
#include <basix/mdspan.hpp>
#include <dolfinx/common/types.h>
#include <span>
#include <utility>
#include <vector>

namespace dolfinx
{

namespace fem
{
class ElementDofLayout;
} // namespace fem

namespace mesh
{
class Topology;
} // namespace mesh

namespace io
{

/// @brief Get owned entities and associated data from input entities
/// defined by global 'node' indices.
///
/// The input entities and data can be supplied on any rank and this
/// function will manage the communication.
///
/// @param[in] topology A mesh topology.
/// @param[in] nodes_g Global 'input' indices for the mesh, as returned
/// by Geometry::input_global_indices.
/// @param[in] num_nodes_g Global number of geometry nodes, as returned
/// by `Geometry::index_map()->size_global()`.
/// @param[in] cmap_dof_layout Coordinate element dof layout, computed
/// using `Geometry::cmap().create_dof_layout()`.
/// @param[in] xdofmap Dofmap for the mesh geometry (Geometry::dofmap).
/// @param[in] entity_dim Topological dimension of entities to extract.
/// @param[in] entities Mesh entities defined using global input indices
/// ('nodes'), typically from an input mesh file, e.g. [gi0, gi1, gi2]
/// for a triangle. Let [v0, v1, v2] be the vertex indices of some
/// triangle (using local indexing). Each vertex has a 'node' (geometry
/// dof) index, and each node has a persistent input global index, so
/// the triangle [gi0, gi1, gi2] could be identified with [v0, v1, v2].
/// The data is flattened and the shape is `(num_entities,
/// nodes_per_entity)`.
/// @param[in] data Data associated with each entity in `entities`.
/// @return (entity-vertex connectivity of owned entities, associated
/// data (values) with each entity).
///
/// @note This function involves parallel distribution and must be
/// called collectively. Global input indices for entities which are not
/// owned by current rank could be passed to this function. E.g., rank0
/// provides an entity with global input indices [gi0, gi1, gi2], but
/// this identifies a triangle that is owned by rank1. It will be
/// distributed and rank1 will receive the (local) cell-vertex
/// connectivity for this triangle.
template <typename T>
std::pair<std::vector<std::int32_t>, std::vector<T>> distribute_entity_data(
    const mesh::Topology& topology, std::span<const std::int64_t> nodes_g,
    std::int64_t num_nodes_g, const fem::ElementDofLayout& cmap_dof_layout,
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        const std::int32_t,
        MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>
        xdofmap,
    int entity_dim,
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        const std::int64_t,
        MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>
        entities,
    std::span<const T> data);

} // namespace io
} // namespace dolfinx
