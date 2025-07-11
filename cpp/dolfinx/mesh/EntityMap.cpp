// Copyright (C) 2025 Jørgen S. Dokken and Joseph P. Dean
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "EntityMap.h"
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
//-----------------------------------------------------------------------------
bool EntityMap::contains(const Topology& topology) const
{
  return &topology == _topology.get() or &topology == _sub_topology.get();
}
//-----------------------------------------------------------------------------
std::vector<std::int32_t>
EntityMap::map_entities(std::span<const std::int32_t> entities,
                        const Topology& target_topology) const
{
  if (&target_topology == _topology.get())
  {
    // In this case, we want to map from entity indices in `_sub_topology` to
    // corresponding entities in `_topology`. Hence, for each index in
    // `entities`, we get the corresponding index in `_topology` using
    // `_sub_topology_to_topology`
    auto mapped
        = entities
          | std::views::transform([this](std::int32_t i)
                                  { return _sub_topology_to_topology[i]; });
    return std::vector<std::int32_t>(mapped.begin(), mapped.end());
  }
  else if (&target_topology == _sub_topology.get())
  {
    // In this case, we are mapping from entity indices in `_topology` to
    // entity indices in `_sub_topology`. Hence, we first need to construct
    // the "inverse" of `_sub_topology_to_topology`
    std::unordered_map<std::int32_t, std::int32_t> topology_to_sub_topology;
    topology_to_sub_topology.reserve(_sub_topology_to_topology.size());
    for (std::size_t i = 0; i < _sub_topology_to_topology.size(); ++i)
    {
      topology_to_sub_topology.insert(
          {_sub_topology_to_topology[i], static_cast<std::int32_t>(i)});
    }

    // For each entity index in `entities` (which are indices in `_topology`),
    // get the corresponding entity in
    // `_sub_topology`. Since `_sub_topology` consists of a subset of entities
    // in `_topology`, there are entities in topology that may not exist in
    // `_sub_topology`. If this is the case, mark those entities with -1.
    auto mapped
        = entities
          | std::views::transform(
              [&topology_to_sub_topology](std::int32_t i)
              {
                // Map the entity if it exists. If it doesn't, mark
                // with -1.
                auto it = topology_to_sub_topology.find(i);
                return (it != topology_to_sub_topology.end()) ? it->second : -1;
              });
    return std::vector<std::int32_t>(mapped.begin(), mapped.end());
  }
  else
    throw std::runtime_error("Topology not in the map.");
}
//-----------------------------------------------------------------------------
std::vector<std::int32_t> EntityMap::map(const Topology& topology) const
{
  if (&topology == _topology.get())
  {
    // The map from `_sub_topology` to `topology` is simply
    // `_sub_topology_to_topology`
    return _sub_topology_to_topology;
  }
  else if (&topology == _sub_topology.get())
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
//-----------------------------------------------------------------------------
std::size_t EntityMap::dim() const { return _dim; }
//-----------------------------------------------------------------------------
} // namespace dolfinx::mesh
