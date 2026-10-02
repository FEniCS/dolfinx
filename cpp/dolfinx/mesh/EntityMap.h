// Copyright (C) 2025-2026 Jørgen S. Dokken and Joseph P. Dean
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "Topology.h"
#include <algorithm>
#include <basix/mdspan.hpp>
#include <cassert>
#include <concepts>
#include <cstdint>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/types.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <format>
#include <functional>
#include <iterator>
#include <memory>
#include <optional>
#include <ranges>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace dolfinx::mesh
{
/// @brief A bidirectional map relating entities in one topology to
/// another.
class EntityMap
{
public:
  /// @brief Constructor of a bidirectional map relating entities of
  /// dimension `dim` in `topology` and `sub_topology`.
  ///
  /// @tparam U
  /// @param topology A mesh topology.
  /// @param sub_topology Topology of another mesh. This must be a
  /// "sub-topology" of `topology`, i.e. every entity in `sub_topology`
  /// must also exist in `topology`.
  /// @param dim Topological dimension of the entities.
  /// @param sub_topology_to_topology List of entities in `topology`
  /// where `sub_topology_to_topology[i]` is the index in `topology`
  /// corresponding to entity `i` in `sub_topology`.
  /// @pre `sub_topology_to_topology` entries must be distinct.
  template <typename U>
    requires std::is_convertible_v<std::remove_cvref_t<U>,
                                   std::vector<std::int32_t>>
  EntityMap(std::shared_ptr<const Topology> topology,
            std::shared_ptr<const Topology> sub_topology, int dim,
            U&& sub_topology_to_topology)
      : _dim(dim), _topology(topology),
        _sub_topology_to_topology(std::forward<U>(sub_topology_to_topology)),
        _sub_topology(sub_topology)
  {
    if (!topology)
      throw std::invalid_argument("topology must not be null.");
    if (!sub_topology)
      throw std::invalid_argument("sub_topology must not be null.");
    if (dim < 0 or dim > topology->dim() or dim > sub_topology->dim())
    {
      throw std::invalid_argument(
          "dim out of range for topology/sub_topology.");
    }

    auto e_imap = sub_topology->index_map(_dim);
    std::size_t num_ents = e_imap->size_local() + e_imap->num_ghosts();
    if (num_ents != _sub_topology_to_topology.size())
    {
      throw std::invalid_argument(
          "Size mismatch between `sub_topology_to_topology` and index map.");
    }
  }

  /// Copy constructor
  EntityMap(const EntityMap& map) = default;

  /// Move constructor
  EntityMap(EntityMap&& map) = default;

  /// Destructor
  ~EntityMap() = default;

  // Copy assignment (deleted)
  EntityMap& operator=(const EntityMap& map) = delete;

  /// Move assignment
  EntityMap& operator=(EntityMap&& map) = default;

  /// @brief Get the topological dimension of the entities related by
  /// this `EntityMap`.
  /// @return The topological dimension.
  int dim() const;

  /// @brief Get the (parent) topology.
  /// @return The parent topology.
  std::shared_ptr<const Topology> topology() const;

  /// @brief Get the sub-topology.
  /// @return The sub-topology.
  std::shared_ptr<const Topology> sub_topology() const;

  /// @brief Map entities between the sub-topology and the parent
  /// topology.
  ///
  /// If `inverse` is false, this function maps a list of
  /// `this->dim()`-dimensional entities from `this->sub_topology()` to
  /// the corresponding entities in `this->topology()`. If `inverse` is
  /// true, it performs the inverse mapping: from `this->topology()` to
  /// `this->sub_topology()`. Entities that do not exist in the
  /// sub-topology are marked as -1.
  ///
  /// @note If `inverse` is `true`, this function recomputes the inverse
  /// map on every call (it is not cached), which may be expensive if
  /// called repeatedly.
  ///
  /// @param entities List of entity indices in the source topology.
  /// @param inverse If false, maps from `this->sub_topology()` to
  /// `this->topology()`. If true, maps from `this->topology()` to
  /// `this->sub_topology()`.
  /// @return A list of mapped entity indices. Entities that do not
  /// exist in the target topology are marked as -1.
  std::vector<std::int32_t> sub_topology_to_topology(CellRange auto&& entities,
                                                     bool inverse) const
  {
    std::size_t num_entities = std::ranges::size(entities);
    if (!inverse)
    {
      // In this case, we want to map from entity indices in
      // `_sub_topology` to corresponding entities in `_topology`. Hence,
      // for each index in `entities`, we get the corresponding index in
      // `_topology` using `_sub_topology_to_topology`
      auto mapped
          = std::forward<decltype(entities)>(entities)
            | std::views::transform([this](std::int32_t i)
                                    { return _sub_topology_to_topology[i]; });
      std::vector<std::int32_t> mapped_v;
      mapped_v.reserve(num_entities);
      std::ranges::copy(mapped, std::back_inserter(mapped_v));
      return mapped_v;
    }
    else
    {
      // In this case, we are mapping from entity indices in `_topology`
      // to entity indices in `_sub_topology`. Hence, we first need to
      // construct the "inverse" of `_sub_topology_to_topology`
      std::unordered_map<std::int32_t, std::int32_t> topology_to_sub_topology;
      topology_to_sub_topology.reserve(_sub_topology_to_topology.size());
      for (std::size_t i = 0; i < _sub_topology_to_topology.size(); ++i)
      {
        topology_to_sub_topology.insert(
            {_sub_topology_to_topology[i], static_cast<std::int32_t>(i)});
      }

      // For each entity index in `entities` (which are indices in
      // `_topology`), get the corresponding entity in `_sub_topology`.
      // Since `_sub_topology` consists of a subset of entities in
      // `_topology`, there are entities in topology that may not exist in
      // `_sub_topology`. If this is the case, mark those entities with
      // -1.

      auto mapped = std::forward<decltype(entities)>(entities)
                    | std::views::transform(
                        [&topology_to_sub_topology](std::int32_t i)
                        {
                          // Map the entity if it exists. If it doesn't, mark
                          // with -1.
                          auto it = topology_to_sub_topology.find(i);
                          return (it != topology_to_sub_topology.end())
                                     ? it->second
                                     : -1;
                        });
      std::vector<std::int32_t> mapped_v;
      mapped_v.reserve(num_entities);
      std::ranges::copy(mapped, std::back_inserter(mapped_v));
      return mapped_v;
    }
  }

private:
  // Dimension of the entities
  int _dim;

  // A topology
  std::shared_ptr<const Topology> _topology;

  // A list of `_dim`-dimensional entities in _topology, where
  // `_sub_topology_to_topology[i]` is the index in `_topology` of the
  // `i`th entity in `_sub_topology`
  std::vector<std::int32_t> _sub_topology_to_topology;

  // A second topology, consisting of a subset of entities in
  // `_topology`
  std::shared_ptr<const Topology> _sub_topology;
};

/// @brief Find the entity map relating two topologies.
/// @param[in] entity_maps Maps to search.
/// @param[in] topology0 One of the topologies.
/// @param[in] topology1 The other topology.
/// @return The map whose topology and sub-topology are `topology0` and
/// `topology1`, in either order.
const EntityMap& find_entity_map(
    std::span<const std::reference_wrapper<const EntityMap>> entity_maps,
    const Topology& topology0, const Topology& topology1);

/// @brief Map integration entities of a topology to the cells of a
/// related topology.
/// @param[in] topology_c Topology to extract cell indices on.
/// @param[in] topology Topology that `entities` belong to.
/// @param[in] entities Integration entities. Either a rank-1 list of
/// cells of `topology`, or a rank-2 list of (cell, local entity index)
/// pairs.
/// @param[in] entity_map Map between `topology` and `topology_c`.
/// Required when `topology_c` is not `topology`.
/// @return Cell of `topology_c` for each of `entities`, or -1 if there
/// is none.
/// @note For (cell, local entity index) pairs and a lower-dimensional
/// `topology_c`, the connectivity of `topology` from its cells to
/// entities of dimension `topology_c.dim()` must have been computed.
template <typename E>
  requires(std::remove_cvref_t<E>::rank() == 1
           or std::remove_cvref_t<E>::rank() == 2)
          and std::same_as<typename std::remove_cvref_t<E>::value_type,
                           std::int32_t>
std::vector<std::int32_t> extract_cells_from_entities(
    const Topology& topology_c, const Topology& topology, E entities,
    std::optional<std::reference_wrapper<const EntityMap>> entity_map)
{
  auto span_to_vector = [](auto entities)
  {
    assert(entities.rank() == 1);

    std::vector<std::int32_t> vec;
    vec.reserve(entities.extent(0));
    for (std::size_t i = 0; i < entities.extent(0); ++i)
      vec.push_back(entities[i]);
    return vec;
  };

  if (&topology_c == &topology)
  {
    // If same topology no mapping is needed
    if constexpr (entities.rank() == 1)
      return span_to_vector(entities);
    else
      // If (cell, local_index) pairs are given, extract the cells
      return span_to_vector(md::submdspan(entities, md::full_extent, 0));
  }

  if (!entity_map)
  {
    throw std::invalid_argument(
        "An entity map is required when topology_c is not topology.");
  }
  const EntityMap& emap = entity_map->get();
  const int tdim = topology.dim();
  const int codim = tdim - topology_c.dim();
  const bool inverse = emap.sub_topology().get() == &topology_c;
  if constexpr (entities.rank() == 1)
  {
    // Cells map directly to cells only between equal dimensions
    if (codim != 0)
    {
      throw std::invalid_argument(
          std::format("Cannot map cells of a topology of dimension {} to "
                      "cells of a topology of dimension {}.",
                      tdim, topology_c.dim()));
    }
    return emap.sub_topology_to_topology(span_to_vector(entities), inverse);
  }
  else
  {
    if (codim == 0)
    {
      // If codim is zero we extract the cells and map them
      auto cells = md::submdspan(entities, md::full_extent, 0);
      return emap.sub_topology_to_topology(span_to_vector(cells), inverse);
    }
    else
    {
      // Any other codim needs to map (cell, local index) to entities and
      // then to cells of `topology_c`
      if (!inverse)
      {
        throw std::invalid_argument(
            "Unsupported mapping. Can only map from submesh to parent mesh.");
      }
      assert(codim > 0);
      std::shared_ptr<const graph::AdjacencyList<std::int32_t>> c_to_e
          = topology.connectivity(tdim, tdim - codim);
      if (!c_to_e)
      {
        throw std::runtime_error(
            std::format("Topology connectivity from dimension {} to {} not "
                        "found.",
                        tdim, tdim - codim));
      }
      // Map (cell, local_index) to entity of `topology`
      std::vector<std::int32_t> sub_entities;
      sub_entities.reserve(entities.extent(0));
      for (std::size_t e = 0; e < entities.extent(0); ++e)
        sub_entities.push_back(c_to_e->links(entities(e, 0))[entities(e, 1)]);

      // Map entity of `topology` to cell of `topology_c`
      return emap.sub_topology_to_topology(sub_entities, inverse);
    }
  }
}
} // namespace dolfinx::mesh
