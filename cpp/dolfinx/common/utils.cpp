// Copyright (C) 2009 Anders Logg
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "utils.h"
#include <sstream>
#include <xtensor/xadapt.hpp>
#include <xtensor/xio.hpp>
//-----------------------------------------------------------------------------
std::string dolfinx::common::indent(std::string block)
{
  std::string indentation("  ");
  std::stringstream s;

  s << indentation;
  for (std::size_t i = 0; i < block.size(); ++i)
  {
    s << block[i];
    if (block[i] == '\n' && i < block.size() - 1)
      s << indentation;
  }

  return s.str();
}
//-----------------------------------------------------------------------------
std::shared_ptr<const dolfinx::common::IndexMap>
dolfinx::common::compress_index_map(
    std::shared_ptr<const dolfinx::common::IndexMap> map,
    const xtl::span<const std::int32_t>& indices)
{
  const std::int32_t local_size = map->size_local();
  std::vector<std::int32_t> local_entities;
  const std::vector<std::int64_t>& ghosts = map->ghosts();
  const std::vector<std::int32_t> is_ghost
      // Split indices into local and ghost indices
      for (std::size_t i = 0; i < indices.size(); ++i)
  {
    if (indices[i] < local_size)
      local_entities.push_back(indices[i]);
    else
      ghost_indices.push_back(indices[i] - local_size);
  }
  // Commmunicate ghost indices
  MPI_Comm forward_comm
      = map->comm(dolfinx::common::IndexMap::Direction::forward);
  std::vector<std::int64_t> ghost_entities(ghost_indices.size());
  std::vector<std::int32_t> ghost_owners(ghost_indices.size());
  const std::vector<std::int64_t>& ghosts = map->ghosts();
  map->ghost_owner_rank() for (std::size_t i = 0; i < ghost_indices.size(); ++i)
  {
    ghost_entities[i] = ghosts[ghost_indices[i]];
    ghost_owners[i] = ghost_owners[gh]
  }

  return std::make_shared<const dolfinx::common::IndexMap>(
      map->comm(dolfinx::common::IndexMap::Direction::forward),
      map->size_local());
}
//-----------------------------------------------------------------------------
