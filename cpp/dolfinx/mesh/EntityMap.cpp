// Copyright (C) 2025-2026 Jørgen S. Dokken and Joseph P. Dean
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "EntityMap.h"
#include "Topology.h"
#include <algorithm>
#include <format>
#include <functional>
#include <span>
#include <stdexcept>

namespace dolfinx::mesh
{
//-----------------------------------------------------------------------------
int EntityMap::dim() const { return _dim; }
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
const EntityMap& find_entity_map(
    std::span<const std::reference_wrapper<const EntityMap>> entity_maps,
    const Topology& topology0, const Topology& topology1)
{
  auto it
      = std::ranges::find_if(entity_maps,
                             [&topology0, &topology1](const EntityMap& em)
                             {
                               const Topology* t = em.topology().get();
                               const Topology* st = em.sub_topology().get();
                               return (t == &topology0 and st == &topology1)
                                      or (t == &topology1 and st == &topology0);
                             });
  if (it == entity_maps.end())
  {
    throw std::invalid_argument(
        std::format("No entity map relating topologies of dimension {} and "
                    "{} in entity_maps.",
                    topology0.dim(), topology1.dim()));
  }
  return *it;
}
//-----------------------------------------------------------------------------
void check_entity_cell_mapping(const Topology& topology_c,
                               const Topology& topology, int dim)
{
  if (topology.dim() > topology_c.dim() and dim != topology_c.dim())
  {
    throw std::invalid_argument(std::format(
        "Cannot map entities of dimension {} to cells of a topology of "
        "dimension {}. Data on another mesh must live on the entities "
        "being integrated or evaluated over.",
        dim, topology_c.dim()));
  }
}
//-----------------------------------------------------------------------------
} // namespace dolfinx::mesh
