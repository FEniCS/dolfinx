// Copyright (C) 2025 Jørgen S. Dokken and Joseph P. Dean
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once
#include "Topology.h"
#include <concepts>
#include <dolfinx/common/IndexMap.h>
#include <iostream>
#include <ranges>
#include <span>
#include <unordered_map>
#include <vector>

namespace dolfinx::mesh
{
/// @brief A map between entities of two meshes
class EntityMap
{
public:
  /// @brief Constructor of a map from a set of entities in one mesh to a set of
  /// entities in another
  ///
  /// @tparam U
  /// @param topology A mesh topology
  /// @param sub_topology Topology of another mesh. This must be a
  /// "sub-topology" of `topology` i.e. every entity in `sub_topology` must also
  /// exist in `topology`.
  /// @param sub_topology_to_topology A list of entities in `topology`.
  /// `sub_topology_to_topology[i]` is the index in `topology` corresponding to
  /// cell `i` in `sub_topology`.
  template <typename U>
    requires std::is_convertible_v<std::remove_cvref_t<U>,
                                   std::vector<std::int32_t>>
  EntityMap(std::shared_ptr<const Topology> topology,
            std::shared_ptr<const Topology> sub_topology,
            U&& sub_topology_to_topology)
      : _dim(sub_topology->dim()), _topology(topology),
        _sub_topology_to_topology(std::forward<U>(sub_topology_to_topology)),
        _sub_topology(sub_topology)
  {
    auto e_map = sub_topology->index_map(_dim);
    if (!e_map)
    {
      throw std::runtime_error(
          "No index map for entities, call `Topology::create_entities("
          + std::to_string(_dim) + ")");
    }
    std::size_t num_ents = e_map->size_local() + e_map->num_ghosts();
    if (num_ents != _sub_topology_to_topology.size())
    {
      throw std::runtime_error(
          "Size mismatch between `sub_topology_to_topology` and index map.");
    }
  }

  /// Copy constructor
  EntityMap(const EntityMap& map) = default;

  /// Move constructor
  EntityMap(EntityMap&& map) = default;

  // Destructor
  ~EntityMap() = default;

  /// @brief Determine if this `EntityMap` contains a given topology
  /// @param topology A topology
  /// @return Returns true if the topology is present, and false otherwise.
  bool contains(const Topology& topology) const
  {
    return &topology == _topology.get() or &topology == _sub_topology.get();
  }

  /// @brief Map entities indices in one topology in this `EntityMap` to indices
  /// in the other topology in this `EntityMap`. When mapping to entity indices
  /// in the sub-topology, any entities that don't exist are marked with -1.
  /// @param entities Entities in one of the topologies in this `EntityMap`
  /// @param topology The topology to map to
  /// @return The mapped entity indices
  std::vector<std::int32_t> map_entities(std::span<const std::int32_t> entities,
                                         const Topology& topology) const
  {
    if (&topology == _topology.get())
    {
      // In this case, we want to map from cell indices in `_sub_topology` to
      // corresponding entities in `_topology`. Hence, for each index in
      // `entities`, we get the corresponding index in `_topology` using
      // `_sub_topology_to_topology`
      auto mapped
          = entities
            | std::views::transform([this](std::int32_t i)
                                    { return _sub_topology_to_topology[i]; });
      return std::vector<std::int32_t>(mapped.begin(), mapped.end());
    }
    else if (&topology == _sub_topology.get())
    {
      // In this case, we are mapping from entity indices in `_topology` to cell
      // indices in `_sub_topology`. Hence, we first need to construct the
      // "inverse" of `_sub_topology_to_topology`
      std::unordered_map<std::int32_t, std::int32_t> topology_to_sub_topology;
      topology_to_sub_topology.reserve(_sub_topology_to_topology.size());
      for (std::size_t i = 0; i < _sub_topology_to_topology.size(); ++i)
      {
        topology_to_sub_topology[_sub_topology_to_topology[i]]
            = static_cast<std::int32_t>(i);
      }

      // For each entity index in `entities` (which are indices in `_topology`),
      // get the corresponding cell in
      // `_sub_topology`. Since `_sub_topology` consists of a subset of entities
      // in `_topology`, there are entities in topology that may not exist in
      // `_sub_topology`. If this is the case, mark those entities with -1.
      auto mapped = entities
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
      return std::vector<std::int32_t>(mapped.begin(), mapped.end());
    }
    else
      throw std::runtime_error("Topology not in the map.");
  }

  /// @brief Get a list representing the map from entities in one topology to
  /// to the other
  /// @param topology The topology to map to
  /// @return A list whose `i`th entry is the entity index in `topology` of
  /// entity `i` in the other topology if it exists, or -1 if it does not.
  std::vector<std::int32_t> map(std::shared_ptr<const Topology> topology) const
  {
    if (topology == _topology)
    {
      // The map from `_sub_topology` to `topology` is simply
      // `_sub_topology_to_topology`
      return _sub_topology_to_topology;
    }
    else if (topology == _sub_topology)
    {
      auto imap = _topology->index_map(_dim);
      assert(imap);
      std::vector<std::int32_t> topology_to_sub_topology(imap->size_local()
                                                         + imap->num_ghosts());

      // Create the "inverse" of `_sub_topology_to_topology`
      for (std::size_t i = 0; i < _sub_topology_to_topology.size(); ++i)
      {
        topology_to_sub_topology[_sub_topology_to_topology[i]]
            = static_cast<std::int32_t>(i);
      }

      return topology_to_sub_topology;
    }
    else
      throw std::runtime_error("Topology not in the map.");
  }

  /// @brief Get the topological dimension of entities the map is created for.
  /// @return The dimension
  std::size_t dim() const { return _dim; }

private:
  std::size_t _dim;                          ///< Dimension of the entities
  std::shared_ptr<const Topology> _topology; ///< A topology
  std::vector<std::int32_t>
      _sub_topology_to_topology; ///< A list of entities in _topology, where
                                 ///< `_sub_topology_to_topology[i]` is the
                                 ///< index in topology of the `i`th entity in
                                 ///< `_sub_topology`

  std::shared_ptr<const Topology>
      _sub_topology; ///< A second topology, consisting of a subset of entities
                     ///< in `_topology`
};

} // namespace dolfinx::mesh