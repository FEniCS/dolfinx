// Copyright (C) 2025 Jørgen S. Dokken and Joseph P. Dean
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "EntityMap.h"
#include "Topology.h"
#include <ranges>
#include <span>
#include <unordered_map>
#include <vector>

namespace dolfinx::mesh
{
//-----------------------------------------------------------------------------
std::size_t EntityMap::dim() const { return _dim; }
//-----------------------------------------------------------------------------
std::shared_ptr<const Topology> EntityMap::topology() const
{
  return _topology;
}
//-----------------------------------------------------------------------------
std::shared_ptr<const Topology> EntityMap::sub_topology() const
{
  return _sub_topology;
}
//-----------------------------------------------------------------------------
std::vector<std::int32_t>
EntityMap::sub_topology_to_topology(std::span<const std::int32_t> entities,
                                    bool inverse) const
{
  if (!inverse)
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
  else
  {
    // In this case, we are mapping from entity indices in `_topology` to
    // entity indices in `_sub_topology`. Hence, we first need to construct
    // the "inverse" of `_sub_topology_to_topology`
    std::unordered_map<std::int32_t, std::int32_t> topology_to_sub_topology;
    topology_to_sub_topology.reserve(_sub_topology_to_topology.size());
    for (std::size_t i = 0; i < _sub_topology_to_topology.size(); ++i)
    {
      topology_to_sub_topology.insert({_sub_topology_to_topology[i], i});
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
}
//-----------------------------------------------------------------------------
} // namespace dolfinx::mesh
