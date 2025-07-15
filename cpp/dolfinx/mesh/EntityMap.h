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

  /// @brief Get the topological dimension of the entities related by this
  /// `EntityMap`.
  /// @return The topological dimension
  std::size_t dim() const;

  /// @brief Get the (parent) topology
  /// @return The topology
  std::shared_ptr<const Topology> topology() const;

  /// @brief Get the sub-topology
  /// @return The sub-topology
  std::shared_ptr<const Topology> sub_topology() const;

  /// @brief Map entities between the sub-topology and the parent topology.
  ///
  /// If `inverse` is false, this function maps a list of
  /// `this->dim()`-dimensional entities from `this->sub_topology()` to the
  /// corresponding entities in `this->topology()`. If `inverse` is true, it
  /// performs the inverse mapping: from `this->topology()` to
  /// `this->sub_topology()`. Entities that do not exist in the sub-topology are
  /// marked as -1.
  ///
  /// @note This function recomputes the map on every call (it is not
  /// cached), which may be expensive if called repeatedly.
  ///
  /// @param entities A list of entity indices in the source topology.
  /// @param inverse If false, maps from `this->sub_topology()` to
  /// `this->topology()`. If true, maps from `this->topology()` to
  /// `this->sub_topology()`.
  /// @return A list of mapped entity indices. Entities that don't exist in the
  /// target topology are marked as -1.
  std::vector<std::int32_t>
  sub_topology_to_topology(std::span<const std::int32_t> entities,
                           bool inverse) const;

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