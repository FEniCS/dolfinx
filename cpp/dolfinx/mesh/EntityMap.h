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
/// @brief A bidirectional map relating entities in one topology to another
class EntityMap
{
public:
  /// @brief Constructor of a map from a set of entities in one topology to a
  /// set of entities in another.
  ///
  /// @tparam U
  /// @param topology A mesh topology
  /// @param sub_topology Topology of another mesh. This must be a
  /// "sub-topology" of `topology` i.e. every entity in `sub_topology` must also
  /// exist in `topology`.
  /// @param dim The dimension of the entities
  /// @param sub_topology_to_topology A list of entities in `topology`.
  /// `sub_topology_to_topology[i]` is the index in `topology` corresponding to
  /// entity `i` in `sub_topology`.
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
  bool contains(const Topology& topology) const;

  /// @brief Map entity indices in one topology in this `EntityMap` to indices
  /// in the other topology in this `EntityMap`. When mapping to entity indices
  /// in the sub-topology, any entities that don't exist are marked with -1.
  /// @param entities Entities in one of the topologies in this `EntityMap`
  /// @param topology The topology to map to
  /// @return The mapped entity indices
  std::vector<std::int32_t> map_entities(std::span<const std::int32_t> entities,
                                         const Topology& topology) const;

  /// @brief Get a list representing the map from entities indices in one
  /// topology of this `EntityMap` to entity indices in the other topology.
  /// @param topology The topology to map to
  /// @return A list whose `i`th entry is the entity index in `topology` of
  /// entity `i` in the other topology in this `EntityMap`. If the entity does
  /// not exist in `topology`, then it is marked with -1.
  std::vector<std::int32_t> map(const Topology& topology) const;

  /// @brief Get the topological dimension of entities the map is created for.
  /// @return The dimension
  std::size_t dim() const;

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