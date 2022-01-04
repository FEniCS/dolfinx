// Copyright (C) 2021 Joseph P. Dean
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "utils.h"
#include <dolfinx/common/MPI.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

std::vector<int32_t> dolfinx::common::get_owned_indices(
    const xtl::span<const std::int32_t>& indices,
    std::shared_ptr<const dolfinx::common::IndexMap> index_map)
{
  // Split indices into those owned by this process and those that
  // are ghosts. `ghost_indices` contains the position of the ghost
  // in index_map->ghosts()
  std::vector<std::int32_t> owned;
  std::vector<std::int32_t> ghost_indices;
  for (std::size_t i = 0; i < indices.size(); ++i)
  {
    if (indices[i] < index_map->size_local())
    {
      owned.push_back(indices[i]);
    }
    else
    {
      const std::int32_t ghost_index = indices[i] - index_map->size_local();
      ghost_indices.push_back(ghost_index);
    }
  }

  // Create an AdjacencyList whose nodes are the processes in the
  // neighborhood and the links for a given process are the ghosts (global
  // numbering) in `indices` owned by that process.
  MPI_Comm reverse_comm
      = index_map->comm(dolfinx::common::IndexMap::Direction::reverse);
  std::vector<std::int32_t> dest_ranks = dolfinx::MPI::neighbors(reverse_comm)[1];
  const std::vector<std::int32_t> ghost_owner_rank = index_map->ghost_owner_rank();
  const std::vector<std::int64_t>& ghosts = index_map->ghosts();
  std::vector<std::int64_t> ghosts_to_send;
  std::vector<std::int32_t> data_per_proc(dest_ranks.size(), 0);
  // Loop through all destination ranks in the neighborhood
  for (std::size_t dest_rank_index = 0; dest_rank_index < dest_ranks.size();
       ++dest_rank_index)
  {
    // Loop through all ghost indices on this rank
    for (std::int32_t ghost_index : ghost_indices)
    {
      // Check if the ghost is owned by the destination rank. If so,
      // add that ghost so it is sent to the correct process.
      if (ghost_owner_rank[ghost_index] == dest_ranks[dest_rank_index])
      {
        ghosts_to_send.push_back(ghosts[ghost_index]);
        data_per_proc[dest_rank_index]++;
      }
    }
  }
  // Create a list of partial sums of the number of ghosts per process
  // and create the AdjacencyList
  std::vector<int> send_disp(dest_ranks.size() + 1, 0);
  std::partial_sum(data_per_proc.begin(), data_per_proc.end(),
                   std::next(send_disp.begin(), 1));
  const dolfinx::graph::AdjacencyList<std::int64_t> data_out(
      std::move(ghosts_to_send), std::move(send_disp));

  // Communicate ghosts on this process in `indices` back to their owners
  const dolfinx::graph::AdjacencyList<std::int64_t> data_in
      = dolfinx::MPI::neighbor_all_to_all(reverse_comm, data_out);

  // Get the local index from the global indices received from other
  // processes and add to `owned`
  std::vector<std::int64_t> global_indices = index_map->global_indices();
  for (std::int64_t global_index : data_in.array())
  {
    auto it
        = std::find(global_indices.begin(), global_indices.end(), global_index);
    assert(it != global_indices.end());
    owned.push_back(std::distance(global_indices.begin(), it));
  }

  // Sort `owned` and remove non-unique entries (we could have received
  // the same ghost from multiple other processes)
  std::sort(owned.begin(), owned.end());
  owned.erase(std::unique(owned.begin(), owned.end()), owned.end());

  return owned;
}
