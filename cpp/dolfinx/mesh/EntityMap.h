// Copyright (C) 2025 Jørgen S. Dokken and Joseph P. Dean
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once
#include "Topology.h"
#include <concepts>
#include <dolfinx/common/IndexMap.h>
#include <span>
#include <vector>

namespace dolfinx::mesh
{
/// @brief A bidirectional map relating entities in one topology to another
class EntityMap
{
public:
  /// @brief Constructor of a bidirectional map relating entities of dimension
  /// `dim` in `topology` and `sub_topology`.
  ///
  /// @tparam U
  /// @param topology A mesh topology
  /// @param sub_topology Topology of another mesh. This must be a
  /// "sub-topology" of `topology` i.e. every entity in `sub_topology` must also
  /// exist in `topology`.
  /// @param dim The dimension of the entities
  /// @param sub_topology_to_topology A list of entities in `topology` where
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

  /// @brief Given a list of entities in a source topology (either of the
  /// topologies in this `EntityMap`), this function returns their corresponding
  /// entity indices in the given target topology `target_topology`.
  ///
  /// If the target topology is the sub-topology, any entities that don't exist
  /// in the sub-topology are marked -1.
  ///
  /// @note This function computes a map every call (the map is not stored). For
  /// multiple calls, this can be expensive and `this->map()` should be used
  /// instead.
  ///
  /// @param entities A list of entity indices in the source topology
  /// @param target_topology The target topology to map the indices to
  /// @return The corresponding list of entities in the target topology.
  /// Entities that don't exist in the target topology are marked -1.
  std::vector<std::int32_t> map_entities(std::span<const std::int32_t> entities,
                                         const Topology& target_topology) const;

  /// @brief Get a list representing the map from entities indices in a source
  /// topology (either of the topologies in this `EntityMap`), to a given
  /// target topology `target_topology`.
  ///
  /// If the target topology is the sub-topology, any entities that don't exist
  /// in the sub-topology are marked -1.
  ///
  /// @param target_topology The target topology to map to
  /// @return A list whose `i`th entry is the entity index in `target_topology`
  /// of entity `i` in the source topology. If the entity does not exist in
  /// `target_topology`, then it is marked with -1.
  std::vector<std::int32_t> map(const Topology& target_topology) const;

  /// @brief Get the topological dimension of the entities related by this
  /// `EntityMap`.
  /// @return The topological dimension
  std::size_t dim() const;

private:
  std::size_t _dim;                          ///< Dimension of the entities
  std::shared_ptr<const Topology> _topology; ///< A topology
  std::vector<std::int32_t>
      _sub_topology_to_topology; ///< A list of `_dim`-dimensional entities in
                                 ///< _topology, where
                                 ///< `_sub_topology_to_topology[i]` is the
                                 ///< index in `_topology` of the `i`th entity
                                 ///< in
                                 ///< `_sub_topology`

  std::shared_ptr<const Topology>
      _sub_topology; ///< A second topology, consisting of a subset of entities
                     ///< in `_topology`
};

} // namespace dolfinx::mesh