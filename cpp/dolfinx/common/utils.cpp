// Copyright (C) 2009 Anders Logg
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "utils.h"
#include <numeric>
#include <sstream>
#include <unordered_map>
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
  const std::vector<std::int64_t>& ghosts = map->ghosts();

  std::vector<int> ghost_owners = map->ghost_owner_rank();

  // Commmunicate ghost indices
  MPI_Comm reverse_comm
      = map->comm(dolfinx::common::IndexMap::Direction::reverse);
  MPI_Comm forward_comm
      = map->comm(dolfinx::common::IndexMap::Direction::forward);
  std::tuple<std::vector<std::int32_t>, std::vector<std::int32_t>> neighbours
      = dolfinx::MPI::neighbors(forward_comm);
  std::vector<std::int32_t>& source_ranks = std::get<0>(neighbours);

  // Create inverse map of src ranks
  std::unordered_map<std::int32_t, std::int32_t> rank_glob_to_loc;
  for (std::size_t i = 0; i < source_ranks.size(); ++i)
    rank_glob_to_loc.insert({source_ranks[i], i});

  // Split indices into local and ghost indices
  const std::int32_t local_size = map->size_local();
  std::vector<std::int32_t> local_entities;
  std::vector<std::int32_t> ghost_indices;
  std::vector<std::int32_t> num_ghosts(source_ranks.size());
  for (std::size_t i = 0; i < indices.size(); ++i)
  {
    if (indices[i] < local_size)
      local_entities.push_back(indices[i]);
    else
    {
      const std::int32_t ghost_index = indices[i] - local_size;
      ghost_indices.push_back(ghost_index);
      num_ghosts[rank_glob_to_loc[ghost_owners[ghost_index]]]++;
    }
  }
  // Compute insertion position in adjacency array
  std::vector<std::int32_t> ghost_adj_offset(source_ranks.size() + 1, 0);
  std::partial_sum(num_ghosts.begin(), num_ghosts.end(),
                   ghost_adj_offset.begin() + 1);
  std::vector<std::int32_t> ghost_adj(ghost_adj_offset.back());

  std::vector<std::int32_t> inserted_ghosts(source_ranks.size(), 0);
  for (std::size_t i = 0; i < ghost_indices.size(); ++i)
  {
    const std::int32_t ghost_index = ghost_indices[i];
    const std::int32_t rank = rank_glob_to_loc[ghost_owners[ghost_index]];
    const std::int32_t local_pos
        = ghost_adj_offset[rank] + inserted_ghosts[rank];
    ghost_adj[local_pos] = ghosts[ghost_index];
    inserted_ghosts[rank]++;
  }
  dolfinx::graph::AdjacencyList<std::int32_t> send_ghosts(ghost_adj,

                                                          ghost_adj_offset);
  //----DEBUG---
  std::stringstream cc;
  std::int32_t glob_rank = dolfinx::MPI::rank(MPI_COMM_WORLD);
  cc << glob_rank << ":\n";
  //---END DEBUG---

  dolfinx::graph::AdjacencyList<std::int32_t> recv_ghosts
      = dolfinx::MPI::neighbor_all_to_all(reverse_comm, send_ghosts);

  cc << "Sent ghosts\n";
  for (std::size_t i = 0; i < ghost_indices.size(); i++)
  {
    cc << ghosts[ghost_indices[i]] << " ";
  }
  cc << "\n";

  cc << "Received ghosts\n";
  for (std::int32_t i = 0; i < recv_ghosts.num_nodes(); i++)
  {
    auto links = recv_ghosts.links(i);
    for (auto li : links)
    {
      cc << li << " ";
    }
    cc << "\n";
  }

  //----DEBUG---
  std::cout << cc.str() << "\n";
  //----END DEBUG---

  return std::make_shared<const dolfinx::common::IndexMap>(
      map->comm(dolfinx::common::IndexMap::Direction::forward),
      map->size_local());
}
//-----------------------------------------------------------------------------
