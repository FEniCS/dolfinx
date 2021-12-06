#include "utils.h"
#include <iostream>
#include <dolfinx/common/MPI.h>
#include <vector>
#include <dolfinx/graph/AdjacencyList.h>
#include <string>
#include <iostream>
#include <sstream>

std::vector<int32_t> dolfinx::common::get_owned_indices(
    MPI_Comm comm, const std::vector<std::int32_t>& indices,
    std::shared_ptr<const dolfinx::common::IndexMap> index_map)
{
  std::stringstream ss;

  int rank = dolfinx::MPI::rank(comm);
  ss << "\n\nmesh::sub\nrank = " << rank << "\n";

  // Get the vertices in submesh_vertices that are owned by this process
  std::vector<std::int32_t> owned_indices; // Local numbering
  for (auto i : indices)
  {
    if (i < index_map->size_local())
    {
      owned_indices.push_back(i);
    }
  }

  ss << "owned_indices = ";
  for (auto i : owned_indices)
  {
    ss << i << " ";
  }
  ss << "\n";

  MPI_Comm reverse_comm
      = index_map->comm(dolfinx::common::IndexMap::Direction::reverse);
  auto dest_ranks = dolfinx::MPI::neighbors(reverse_comm)[1];

  // Index in index_map.ghosts() of ghost vertices
  // TODO Combine with above loop
  std::vector<std::int32_t> ghost_indices;
  for (std::size_t i = 0; i < indices.size(); ++i)
  {
    if (indices[i] >= index_map->size_local())
    {
      const std::int32_t ghost_index = indices[i] - index_map->size_local();
      ghost_indices.push_back(ghost_index);
    }
  }

  ss << "ghost_indices = ";
  for (auto gi : ghost_indices)
  {
    ss << gi << " ";
  }
  ss << "\n";

  ss << "dest_ranks = ";
  for (auto rank : dest_ranks)
  {
    ss << rank << " ";
  }
  ss << "\n";

  std::vector<std::int32_t> data_per_proc(dest_ranks.size(), 0);
  auto ghost_owner_rank = index_map->ghost_owner_rank();
  for (std::size_t dest_rank_index = 0; dest_rank_index < dest_ranks.size();
       ++dest_rank_index)
  {
    for (auto gi : ghost_indices)
    {
      if (ghost_owner_rank[gi] == dest_ranks[dest_rank_index])
      {
        data_per_proc[dest_rank_index]++;
      }
    }
  }

  ss << "data_per_proc = ";
  for (auto dpp : data_per_proc)
  {
    ss << dpp << " ";
  }
  ss << "\n";

  std::vector<int> send_disp(dest_ranks.size() + 1, 0);
  std::partial_sum(data_per_proc.begin(), data_per_proc.end(),
                   std::next(send_disp.begin(), 1));

  ss << "send_disp = ";
  for (auto disp : send_disp)
  {
    ss << disp << " ";
  }
  ss << "\n";

  // TODO Combine with above loop
  std::vector<std::int64_t> ghosts_to_send;
  auto ghosts = index_map->ghosts();
  for (std::size_t dest_rank_index = 0; dest_rank_index < dest_ranks.size();
       ++dest_rank_index)
  {
    for (auto gi : ghost_indices)
    {
      if (ghost_owner_rank[gi] == dest_ranks[dest_rank_index])
      {
        ghosts_to_send.push_back(ghosts[gi]);
      }
    }
  }

  ss << "ghosts_to_send = ";
  for (auto g : ghosts_to_send)
  {
    ss << g << " ";
  }
  ss << "\n";

  const dolfinx::graph::AdjacencyList<std::int64_t> data_out(std::move(ghosts_to_send),
                                                             std::move(send_disp));
  const dolfinx::graph::AdjacencyList<std::int64_t> data_in
      = dolfinx::MPI::neighbor_all_to_all(reverse_comm, data_out);

  ss << "data_in = ";
  for (auto d : data_in.array())
  {
    ss << d << " ";
  }
  ss << "\n";

  // Append ghost vertices from other processes owned by this process to
  // submesh_owned_vertices. First need to get the local index
  auto global_indices = index_map->global_indices();
  for (std::int64_t global_index : data_in.array())
  {
    auto it = std::find(global_indices.begin(), global_indices.end(),
                        global_index);
    assert(it != global_indices.end());
    std::int32_t local_index = std::distance(global_indices.begin(), it);
    owned_indices.push_back(local_index);
  }
  // Sort owned_indices and make unique (could have received same ghost
  // vertex from multiple ranks)
  std::sort(owned_indices.begin(), owned_indices.end());
  owned_indices.erase(std::unique(owned_indices.begin(), owned_indices.end()),
                      owned_indices.end());

  ss << "submesh_owned_vertices = ";
  for (auto sv : owned_indices)
  {
    ss << sv << " ";
  }
  ss << "\n";

  return owned_indices;
}
