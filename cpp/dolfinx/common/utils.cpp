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
  std::vector<std::int32_t> local_indices;
  std::vector<std::int32_t> ghost_indices;
  std::vector<std::int32_t> num_ghosts(source_ranks.size());
  for (std::size_t i = 0; i < indices.size(); ++i)
  {
    if (indices[i] < local_size)
      local_indices.push_back(indices[i]);
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
  std::vector<std::int64_t> ghost_adj(ghost_adj_offset.back());

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
  dolfinx::graph::AdjacencyList<std::int64_t> send_ghosts(ghost_adj,

                                                          ghost_adj_offset);

  dolfinx::graph::AdjacencyList<std::int64_t> recv_ghosts
      = dolfinx::MPI::neighbor_all_to_all(reverse_comm, send_ghosts);

  // Convert local input indices to global indices
  std::vector<std::int64_t> global_indices(local_indices.size());
  map->local_to_global(local_indices, global_indices);

  // Create a sorted vector of indices owned by the process by merging
  // input indices with recieved ghost indices, removing duplicates
  std::vector<std::int64_t> sorted_ghosts = recv_ghosts.array();
  std::sort(sorted_ghosts.begin(), sorted_ghosts.end());
  std::sort(global_indices.begin(), global_indices.end());
  std::vector<std::int64_t> org_global_indices;
  org_global_indices.reserve(local_indices.size() + sorted_ghosts.size());
  std::merge(global_indices.begin(), global_indices.end(),
             sorted_ghosts.begin(), sorted_ghosts.end(),
             std::back_inserter(org_global_indices));
  org_global_indices.erase(
      std::unique(org_global_indices.begin(), org_global_indices.end()),
      org_global_indices.end());

  // Compute global indices for new index map
  std::size_t offset = 0;
  std::size_t new_num_local = org_global_indices.size();
  MPI_Exscan(&new_num_local, &offset, 1, dolfinx::MPI::mpi_type<std::size_t>(),
             MPI_SUM, MPI_COMM_WORLD);

  //----DEBUG---
  std::stringstream cc;
  std::int32_t glob_rank = dolfinx::MPI::rank(MPI_COMM_WORLD);
  cc << glob_rank << ":\n";
  //---END DEBUG---

  // For each received ghost, find new local index
  const std::vector<std::int64_t>& ghost_array = recv_ghosts.array();
  std::vector<std::int64_t> new_ghosts(sorted_ghosts.size());
  for (std::size_t i = 0; i < sorted_ghosts.size(); ++i)
  {
    auto idx_it = std::lower_bound(org_global_indices.begin(),
                                   org_global_indices.end(), ghost_array[i]);
    std::int32_t index = std::distance(org_global_indices.begin(), idx_it);
    new_ghosts[i] = index + offset;
  }

  cc << "Sorted unique global indices: " << xt::adapt(org_global_indices)
     << "\n";
  dolfinx::graph::AdjacencyList<std::int64_t> send_new_ghosts(
      new_ghosts, recv_ghosts.offsets());
  dolfinx::graph::AdjacencyList<std::int64_t> new_ghosts_adj
      = dolfinx::MPI::neighbor_all_to_all(forward_comm, send_new_ghosts);
  cc << "Old ghosts" << xt::adapt(ghost_adj) << "\n";
  cc << "New ghosts" << xt::adapt(new_ghosts_adj.array()) << "\n";
  //----DEBUG---

  std::cout << cc.str() << "\n";
  //----END DEBUG---

  return std::make_shared<const dolfinx::common::IndexMap>(
      map->comm(dolfinx::common::IndexMap::Direction::forward),
      map->size_local());
}
//-----------------------------------------------------------------------------
