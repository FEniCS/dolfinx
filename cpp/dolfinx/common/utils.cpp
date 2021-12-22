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
  // TODO Specify size of vectors

  // Split indices into those owned by this process and those that
  // are ghosts. Note that `ghost_indices` contains the position
  // of the ghost in index_map->ghosts()
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

  // Communicate ghosts on this process back to their owners
  MPI_Comm reverse_comm
      = index_map->comm(dolfinx::common::IndexMap::Direction::reverse);
  auto dest_ranks = dolfinx::MPI::neighbors(reverse_comm)[1];

  auto ghost_owner_rank = index_map->ghost_owner_rank();
  auto ghosts = index_map->ghosts();
  std::vector<std::int64_t> ghosts_to_send;
  std::vector<std::int32_t> data_per_proc(dest_ranks.size(), 0);
  for (std::size_t dest_rank_index = 0; dest_rank_index < dest_ranks.size();
       ++dest_rank_index)
  {
    for (auto gi : ghost_indices)
    {
      if (ghost_owner_rank[gi] == dest_ranks[dest_rank_index])
      {
        ghosts_to_send.push_back(ghosts[gi]);
        data_per_proc[dest_rank_index]++;
      }
    }
  }

  std::vector<int> send_disp(dest_ranks.size() + 1, 0);
  std::partial_sum(data_per_proc.begin(), data_per_proc.end(),
                   std::next(send_disp.begin(), 1));

  const dolfinx::graph::AdjacencyList<std::int64_t> data_out(
      std::move(ghosts_to_send), std::move(send_disp));
  const dolfinx::graph::AdjacencyList<std::int64_t> data_in
      = dolfinx::MPI::neighbor_all_to_all(reverse_comm, data_out);

  // Get the local index of the global indices received from other processes,
  // and add to `owned`
  auto global_indices = index_map->global_indices();
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
