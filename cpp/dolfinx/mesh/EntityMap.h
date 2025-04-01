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
#include <span>
namespace dolfinx::mesh
{
/// @brief A map between entities of two meshes
class EntityMap
{
public:
  /// @brief Constructor of a map between a set of entities belonging to two
  /// meshes.
  /// @param topology0 The first topology in the mapping relation
  /// @param mesh1 The second topology in the mapping relation
  /// @param dim Topological dimension of the mapped entities
  /// @param entities0 The entities belonging to the first mesh
  /// @param entities1 The entities belonging to the second mesh
  template <typename U>
    requires std::is_convertible_v<std::remove_cvref_t<U>,
                                   std::vector<std::int32_t>>
  EntityMap(std::shared_ptr<const Topology> topology0,
            std::shared_ptr<const Topology> topology1, int dim, U&& entities0,
            U&& entities1)
      : _dim(dim), _topology0(topology0),
        _entities0(std::forward<U>(entities0)), _topology1(topology1),
        _entities1(std::forward<U>(entities1))
  {
    if (_entities0.size() != _entities1.size())
    {
      throw std::runtime_error("Entities must have the same size.");
    }
  }

  /// @brief Constructor of a map between a set of entities belonging to two
  /// meshes.
  ///
  /// Entity `i` in mesh1 is assumed to map to `entities0[i]`.
  ///
  /// @param mesh0 The first mesh in the mapping relation
  /// @param mesh1 The second mesh in the mapping relation
  /// @param dim Topological dimension of the mapped entities
  /// @param entities0 The entities belonging to the first mesh
  template <typename U>
    requires std::is_convertible_v<std::remove_cvref_t<U>,
                                   std::vector<std::int32_t>>
  EntityMap(std::shared_ptr<const Topology> topology0,
            std::shared_ptr<const Topology> topology1, int dim, U&& entities0)
      : _dim(dim), _topology0(topology0), _topology1(topology1),
        _entities0(std::forward<U>(entities0))
  {
    auto e_map = topology1->index_map(dim);
    if (!e_map)
    {
      throw std::runtime_error(
          "No index map for entities, call `Topology::create_entities("
          + std::to_string(dim) + ")");
    }
    std::cout << e_map->size_local() + e_map->num_ghosts()
              << " == " << _entities0.size() << "\n";
    if (e_map->size_local() + e_map->num_ghosts() != _entities0.size())
      throw std::runtime_error("Size mismatch between entities and index map.");
    _entities1.resize(entities0.size());
    std::iota(_entities1.begin(), _entities1.end(), 0);
  }

  /// Copy constructor
  EntityMap(const EntityMap& map) = default;

  /// Move constructor
  EntityMap(EntityMap&& map) = default;

  // Destructor
  ~EntityMap() = default;

  bool contains(std::shared_ptr<const Topology> topology) const
  {
    return topology == _topology0 || topology == _topology1;
  }

  std::uint8_t topology_index(std::shared_ptr<const Topology> topology) const
  {
    if (topology == _topology0)
      return 0;
    else if (topology == _topology1)
      return 1;
    else
      throw std::runtime_error("Mesh not in the map.");
  }

  std::span<const std::int32_t>
  get_entities(std::shared_ptr<const Topology> topology) const
  {
    if (topology == _topology0)
      return std::span<const std::int32_t>(_entities0);
    else if (topology == _topology1)
      return std::span<const std::int32_t>(_entities1);
    else
      throw std::runtime_error("Mesh not in the map.");
  }

  std::size_t dim() const { return _dim; }

private:
  std::size_t _dim;                           ///< Dimension of the entities
  std::shared_ptr<const Topology> _topology0; ///< The first mesh
  std::vector<std::int32_t>
      _entities0; ///<  Entities belonging to the first mesh

  std::shared_ptr<const Topology> _topology1; ///< The second mesh
  std::vector<std::int32_t>
      _entities1; ///<  Entities belonging to the second mesh
};

} // namespace dolfinx::mesh