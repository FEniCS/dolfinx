// Copyright (C) 2025 Jørgen S. Dokken
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
  /// @brief Constructor of a map between a set of entities belonging to two
  /// meshes.
  ///
  /// Entity `i` in mesh1 is assumed to map to `entities0[i]`.
  ///
  /// @tparam U
  /// @param topology0 The first topology in the mapping relation
  /// @param topology1 The second topology in the mapping relation
  /// @param dim  Topological dimension of the mapped entities
  /// @param entities0 The entities belonging to the first mesh
  template <typename U>
    requires std::is_convertible_v<std::remove_cvref_t<U>,
                                   std::vector<std::int32_t>>
  EntityMap(std::shared_ptr<const Topology> topology0,
            std::shared_ptr<const Topology> topology1, int dim, U&& entities0)
      : _dim(dim), _topology0(topology0),
        _entities0(std::forward<U>(entities0)), _topology1(topology1)
  {
    auto e_map = topology1->index_map(dim);
    if (!e_map)
    {
      throw std::runtime_error(
          "No index map for entities, call `Topology::create_entities("
          + std::to_string(dim) + ")");
    }
    std::size_t num_ents
        = static_cast<std::size_t>(e_map->size_local() + e_map->num_ghosts());
    if (num_ents != _entities0.size())
      throw std::runtime_error("Size mismatch between entities and index map.");
  }

  /// Copy constructor
  EntityMap(const EntityMap& map) = default;

  /// Move constructor
  EntityMap(EntityMap&& map) = default;

  // Destructor
  ~EntityMap() = default;

  /// TODO
  bool contains(std::shared_ptr<const Topology> topology) const
  {
    return topology == _topology0 or topology == _topology1;
  }

  /// @brief TODO
  /// @param entities
  /// @param topology
  /// @return
  std::vector<std::int32_t>
  map_entities(std::span<const std::int32_t> entities,
               std::shared_ptr<const Topology> topology) const
  {
    if (topology == _topology0)
    {
      auto mapped
          = entities
            | std::views::transform([this](int i) { return _entities0[i]; });
      return std::vector<std::int32_t>(mapped.begin(), mapped.end());
    }
    else if (topology == _topology1)
    {
      std::unordered_map<std::int32_t, std::int32_t> parent_to_sub;
      parent_to_sub.reserve(_entities0.size());
      for (std::size_t sub_idx = 0; sub_idx < _entities0.size(); ++sub_idx)
      {
        parent_to_sub[_entities0[sub_idx]] = static_cast<std::int32_t>(sub_idx);
      }

      auto mapped = entities
                    | std::views::transform(
                        [&parent_to_sub](int parent_idx)
                        {
                          auto it = parent_to_sub.find(parent_idx);
                          return (it != parent_to_sub.end()) ? it->second : -1;
                        });
      return std::vector<std::int32_t>(mapped.begin(), mapped.end());
    }
    else
      throw std::runtime_error("Topology not in the map.");
  }

  /// TODO
  std::size_t dim() const { return _dim; }

private:
  std::size_t _dim;                           ///< Dimension of the entities
  std::shared_ptr<const Topology> _topology0; ///< The first mesh
  std::vector<std::int32_t>
      _entities0; ///<  Entities belonging to the first mesh

  std::shared_ptr<const Topology> _topology1; ///< The second mesh
};

} // namespace dolfinx::mesh