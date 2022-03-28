// Copyright (C) 2006-2022 Anders Logg and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "Topology.h"
#include "cell_types.h"
#include "permutationcomputation.h"
#include "topologycomputation.h"
#include "utils.h"
#include <algorithm>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/log.h>
#include <dolfinx/common/sort.h>
#include <dolfinx/common/utils.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/graph/partition.h>
#include <numeric>
#include <random>

using namespace dolfinx;
using namespace dolfinx::mesh;

namespace
{

/// @brief Find an entry in an a sorted container of `std::pair` objects
/// by the first element of the pair.
///
/// @note This function uses `std::lower_bound`, hence the input
/// container should be sorted. To check that an enty is found, the
/// return iterators should be checked against the `value.`
///
/// @param[in] x Sorted container of `std::pair` objects
/// @param[in] value The value to find in the container (first entry of
/// the pair)
/// @return Iterator to the position in `x` where `value` is found
template <typename T>
auto find_idx(T& x, typename T::value_type::first_type value)
{
  using P = typename T::value_type;
  auto it
      = std::lower_bound(x.begin(), x.end(), P(value, 0),
                         [](auto& a, auto& b) { return a.first < b.first; });
  return it;
}

//-----------------------------------------------------------------------------

/// @todo Improve documentation
/// @brief Determine owner and sharing ranks sharing an index.
///
/// @note Collective
///
/// Indices are sent to a 'post office' rank, which uses a
/// (deterministic) random number generator to determine which ranks is
/// the 'owner'. This information is sent back to the ranks who sent the
/// index to the post office,
///
/// @param[in] comm MPI communicator
/// @param[in] indices Global indices to determine a an owning MPI ranks
/// for.
/// @return Map from global index to sharing ranks for each index in
/// indices. The owner rank is the first as the first in the of ranks.
graph::AdjacencyList<int>
determine_sharing_ranks(MPI_Comm comm,
                        const xtl::span<const std::int64_t>& indices)
{
  common::Timer timer("Topology: determine shared index ownership");

  const int size = dolfinx::MPI::size(comm);

  // FIXME: use sensible name
  std::int64_t global_range = 0;
  {
    std::int64_t max_index
        = indices.empty() ? 0
                          : *std::max_element(indices.begin(), indices.end());
    MPI_Allreduce(&max_index, &global_range, 1, MPI_INT64_T, MPI_MAX, comm);
    global_range += 1;
  }

  // Build {dest, pos} list, and sort
  std::vector<std::array<int, 2>> dest_to_index;
  {
    dest_to_index.reserve(indices.size());
    for (auto idx : indices)
    {
      int dest = dolfinx::MPI::index_owner(size, idx, global_range);
      dest_to_index.push_back({dest, static_cast<int>(dest_to_index.size())});
    }
    std::sort(dest_to_index.begin(), dest_to_index.end());
  }

  // Build list of neighbour dest ranks and count number of indices to
  // send to each post office
  std::vector<int> dest;
  std::vector<std::int32_t> num_items_per_dest0;
  {
    auto it = dest_to_index.begin();
    while (it != dest_to_index.end())
    {
      // const int neigh_rank = dest.size();

      // Store global rank and find iterator to next global rank
      dest.push_back((*it)[0]);
      auto it1
          = std::find_if(it, dest_to_index.end(),
                         [r = dest.back()](auto& idx) { return idx[0] != r; });

      // Store number of items for current rank
      num_items_per_dest0.push_back(std::distance(it, it1));

      // Advance iterator
      it = it1;
    }
  }

  // Determine src ranks. Sort ranks so that ownership determination is
  // deterministic for a given number of ranks.
  std::vector<int> src = dolfinx::MPI::compute_graph_edges_nbx(comm, dest);
  std::sort(src.begin(), src.end());

  // Create neighbourhood communicator for sending data to post offices
  MPI_Comm neigh_comm0;
  MPI_Dist_graph_create_adjacent(comm, src.size(), src.data(), MPI_UNWEIGHTED,
                                 dest.size(), dest.data(), MPI_UNWEIGHTED,
                                 MPI_INFO_NULL, false, &neigh_comm0);

  // Compute send displacements
  std::vector<std::int32_t> send_disp0(num_items_per_dest0.size() + 1, 0);
  std::partial_sum(num_items_per_dest0.begin(), num_items_per_dest0.end(),
                   std::next(send_disp0.begin()));

  // Send number of items to post offices (destination) that I will be
  // sending
  std::vector<int> num_items_recv0(src.size());
  num_items_per_dest0.reserve(1);
  num_items_recv0.reserve(1);
  MPI_Neighbor_alltoall(num_items_per_dest0.data(), 1, MPI_INT,
                        num_items_recv0.data(), 1, MPI_INT, neigh_comm0);

  // Prepare receive displacement and buffers
  std::vector<std::int32_t> recv_disp0(num_items_recv0.size() + 1, 0);
  std::partial_sum(num_items_recv0.begin(), num_items_recv0.end(),
                   std::next(recv_disp0.begin()));

  // Pack send buffer
  std::vector<int> send_buffer0;
  send_buffer0.reserve(send_disp0.back());
  for (auto idx : dest_to_index)
    send_buffer0.push_back(indices[idx[1]]);

  // Send/receive global indices
  std::vector<int> recv_buffer0(recv_disp0.back());
  MPI_Neighbor_alltoallv(send_buffer0.data(), num_items_per_dest0.data(),
                         send_disp0.data(), MPI_INT, recv_buffer0.data(),
                         num_items_recv0.data(), recv_disp0.data(), MPI_INT,
                         neigh_comm0);
  MPI_Comm_free(&neigh_comm0);

  // -- Transpose

  // Build {global index, pos, src} list
  std::vector<std::array<std::int64_t, 3>> indices_list;
  for (std::size_t p = 0; p < recv_disp0.size() - 1; ++p)
  {
    for (std::int32_t i = recv_disp0[p]; i < recv_disp0[p + 1]; ++i)
      indices_list.push_back({recv_buffer0[i], i, int(p)});
  }
  std::sort(indices_list.begin(), indices_list.end());

  // Find which ranks have each index
  std::vector<std::int32_t> num_items_per_dest1(recv_disp0.size() - 1, 0);
  std::vector<std::int32_t> num_items_per_pos1(recv_disp0.back(), 0);

  std::vector<int> owner;
  std::vector<int> disp1 = {0};
  {
    std::mt19937 rng(dolfinx::MPI::rank(comm));
    auto it = indices_list.begin();
    while (it != indices_list.end())
    {
      // Find iterator to next different global index
      auto it1
          = std::find_if(it, indices_list.end(), [idx0 = (*it)[0]](auto& idx) {
              return idx[0] != idx0;
            });

      // Number of times index is repeated
      std::size_t num = std::distance(it, it1);

      // Pick an owner
      auto it_owner = it;
      if (num > 1)
      {
        std::uniform_int_distribution<int> distrib(0, num - 1);
        it_owner = std::next(it, distrib(rng));
      }
      owner.push_back((*it_owner)[2]);

      // Update number of items to be sent to each rank and record
      // owner
      for (auto itx = it; itx != it1; ++itx)
      {
        auto& data = *itx;
        num_items_per_pos1[data[1]] = num + 1;
        num_items_per_dest1[data[2]] += num + 1;
      }

      disp1.push_back(disp1.back() + num);

      // Advance iterator
      it = it1;
    }
  }

  // Compute send displacement
  std::vector<std::int32_t> send_disp1(num_items_per_dest1.size() + 1, 0);
  std::partial_sum(num_items_per_dest1.begin(), num_items_per_dest1.end(),
                   std::next(send_disp1.begin()));

  // Build send buffer
  std::vector<int> send_buffer1(send_disp1.back());
  {
    // Compute buffer  displacement
    std::vector<std::int32_t> bdisp1(num_items_per_pos1.size() + 1, 0);
    std::partial_sum(num_items_per_pos1.begin(), num_items_per_pos1.end(),
                     std::next(bdisp1.begin()));

    for (std::size_t i = 0; i < disp1.size() - 1; ++i)
    {
      // Get data for first occurrence of global index
      std::int32_t owner_rank = owner[i];
      std::int32_t num_sharing_ranks = disp1[i + 1] - disp1[i];

      // For each appearance of the global index the sharing ranks
      auto indices_it0 = std::next(indices_list.begin(), disp1[i]);
      auto indices_it1 = std::next(indices_it0, num_sharing_ranks);
      for (std::int32_t j = disp1[i]; j < disp1[i + 1]; ++j)
      {
        auto& data1 = indices_list[j];
        std::size_t pos = data1[1];
        std::int32_t bufferpos = bdisp1[pos];
        send_buffer1[bufferpos] = num_sharing_ranks;

        // Store indices (global)
        auto it0 = std::next(send_buffer1.begin(), bufferpos + 1);
        std::transform(indices_it0, indices_it1, it0,
                       [&src](auto& x) { return src[x[2]]; });

        auto it1 = std::next(it0, num_sharing_ranks);
        auto it_owner = std::find(it0, it1, src[owner_rank]);
        assert(it_owner != it1);
        std::iter_swap(it0, it_owner);
      }
    }
  }

  // Send back
  MPI_Comm neigh_comm1;
  MPI_Dist_graph_create_adjacent(comm, dest.size(), dest.data(), MPI_UNWEIGHTED,
                                 src.size(), src.data(), MPI_UNWEIGHTED,
                                 MPI_INFO_NULL, false, &neigh_comm1);

  // Send number of values to receive
  std::vector<int> num_items_recv1(dest.size());
  num_items_per_dest1.reserve(1);
  num_items_recv1.reserve(1);
  MPI_Neighbor_alltoall(num_items_per_dest1.data(), 1, MPI_INT,
                        num_items_recv1.data(), 1, MPI_INT, neigh_comm1);

  // Prepare receive displacements
  std::vector<std::int32_t> recv_disp1(num_items_recv1.size() + 1, 0);
  std::partial_sum(num_items_recv1.begin(), num_items_recv1.end(),
                   std::next(recv_disp1.begin()));

  // Send data
  std::vector<int> recv_buffer1(recv_disp1.back());
  MPI_Neighbor_alltoallv(send_buffer1.data(), num_items_per_dest1.data(),
                         send_disp1.data(), MPI_INT, recv_buffer1.data(),
                         num_items_recv1.data(), recv_disp1.data(), MPI_INT,
                         neigh_comm1);
  MPI_Comm_free(&neigh_comm1);

  // Build adjacency list
  std::vector<int> data;
  std::vector<std::int32_t> graph_offsets = {0};
  {
    auto it = recv_buffer1.begin();
    while (it != recv_buffer1.end())
    {
      const std::size_t d = std::distance(recv_buffer1.begin(), it);
      std::int64_t num_ranks = *it;

      xtl::span ranks(recv_buffer1.data() + d + 1, num_ranks);
      data.insert(data.end(), ranks.begin(), ranks.end());
      graph_offsets.push_back(graph_offsets.back() + num_ranks);

      std::advance(it, num_ranks + 1);
    }
  }

  return graph::AdjacencyList<int>(std::move(data), std::move(graph_offsets));
}
//-----------------------------------------------------------------------------

/// @brief For each vertex, assign a marker that indicates if the vertex
/// is connected to a ghost cell and build a list of vertices with
/// undetermined ownership.
///
/// The marker for each vertex is:
///
/// * (-1) for a vertex that is connected to a ghost cell
/// * (-2) for a vertex that connected **only** to owned (local) cells
///
/// The index of vertices that are connected to both owned and ghost
/// cells are added to a vector.
///
/// @todo Would it be helpful here to also make vertices that are
/// definitely not owned, i.e. are connect to ghost cells only?
///
/// @param cells Input mesh topology
/// @param num_local_cells Number of local (non-ghost) cells. These
/// comes before ghost cells in `cells`.
/// @return (global_index_to_maker for (-1) and (-2) cases, indices for
/// other vertices)

/// @return
/// 1. For each vertex, a (global_index, marker) pair, where the marker
/// is as described above.
/// 2. List if vertices (using global indices) with undetermined
/// ownership. These vertices are attached to owned and an un-owned
/// cells.
std::pair<std::vector<std::pair<std::int64_t, std::int32_t>>,
          std::vector<std::int64_t>>
compute_vertex_markers(const graph::AdjacencyList<std::int64_t>& cells,
                       int num_local_cells)
{
  common::Timer t0(
      "Topology: mark vertices by type (owned, possibly owned, ghost)");

  // Build set of 'local' cell vertices
  std::vector<std::int64_t> local_vertex_set(
      cells.array().begin(),
      std::next(cells.array().begin(), cells.offsets()[num_local_cells]));
  dolfinx::radix_sort(xtl::span(local_vertex_set));
  local_vertex_set.erase(
      std::unique(local_vertex_set.begin(), local_vertex_set.end()),
      local_vertex_set.end());

  // Build set of ghost cell vertices
  std::vector<std::int64_t> ghost_vertex_set(
      std::next(cells.array().begin(), cells.offsets()[num_local_cells]),
      cells.array().end());
  dolfinx::radix_sort(xtl::span(ghost_vertex_set));
  ghost_vertex_set.erase(
      std::unique(ghost_vertex_set.begin(), ghost_vertex_set.end()),
      ghost_vertex_set.end());

  // Vector to hold markers for vertices
  std::vector<std::pair<std::int64_t, std::int32_t>> global_to_local_v;

  // Add all vertices attached a ghost cell to `global_to_local_v` with
  // the marker -1
  std::transform(ghost_vertex_set.begin(), ghost_vertex_set.end(),
                 std::back_inserter(global_to_local_v),
                 [](auto idx) -> decltype(global_to_local_v)::value_type {
                   return {idx, -1};
                 });

  // For vertices that are attached to owned cells:
  // - If the vertex is also in the set of vertices attached to a ghost
  //   cell, add vertex to set of vertices with undetermined ownership
  // - Else, add vertex to `global_to_local_v` with marker '-2' to
  //   indicate that the vertex is owned by the current process
  std::vector<std::int64_t> unknown_indices_set;
  for (std::int64_t global_index : local_vertex_set)
  {
    // Check if vertex is attached to a ghost cell
    auto it = std::lower_bound(ghost_vertex_set.begin(), ghost_vertex_set.end(),
                               global_index);
    if (it != ghost_vertex_set.end() and *it == global_index)
    {
      // Vertex is attached to a ghost cell, therefore ownership is
      // undetermined at this stage
      unknown_indices_set.push_back(global_index);
    }
    else
    {
      // Vertex is not attached to a ghost cell, therefore is owned by
      // this rank. Set maker to -2.
      global_to_local_v.push_back({global_index, -2});
    }
  }

  return {std::move(global_to_local_v), std::move(unknown_indices_set)};
}
//-----------------------------------------------------------------------------

/// @brief Send the vertex numbering for owned vertices to processes
/// that also share them, returning a list of triplets received from
/// other ranks.
///
/// Each triplet consists of {old_global_vertex_index,
/// new_global_vertex_index, owning_rank}. The received vertices will be
/// "ghost" on this rank.
///
/// Input params as in mesh::create_topology()
///
//// @note Collective
/// @param[in] comm Neighbourhood communicator
/// @return list of triplets
std::vector<std::int64_t> exchange_vertex_numbering(
    const MPI_Comm& comm, const xtl::span<const int>& local_to_global_rank,
    const xtl::span<const std::int64_t>& vertices,
    const graph::AdjacencyList<int>& vertex_to_ranks,
    std::int64_t global_offset_v,
    const xtl::span<const std::pair<std::int64_t, std::int32_t>>&
        global_to_local_vertices)
{
  const int mpi_rank = dolfinx::MPI::rank(comm);

  // FIXME: Remove vector<vector> data structure
  // Pack send data
  std::vector<std::vector<std::int64_t>> send_buffer(
      local_to_global_rank.size());
  for (std::int32_t i = 0; i < vertex_to_ranks.num_nodes(); ++i)
  {
    // Get (global) ranks that share this vertex. Note that first rank
    // is the owner.
    auto vertex_ranks = vertex_to_ranks.links(i);
    if (vertex_ranks.front() == mpi_rank)
    {
      // Get local vertex index
      std::int64_t vertex_idx_global = vertices[i];
      auto vlocal_it = find_idx(global_to_local_vertices, vertex_idx_global);
      assert(vlocal_it != global_to_local_vertices.end());
      assert(vlocal_it->first == vertex_idx_global);
      assert(vlocal_it->second != -1);

      // Owned and shared with these processes (starting from 1, 0 is self)
      for (std::size_t j = 1; j < vertex_ranks.size(); ++j)
      {
        // Find rank on the neighborhood comm
        auto nrank_it
            = std::lower_bound(local_to_global_rank.begin(),
                               local_to_global_rank.end(), vertex_ranks[j]);
        assert(nrank_it != local_to_global_rank.end());
        assert(*nrank_it == vertex_ranks[j]);
        int rank_neighbor
            = std::distance(local_to_global_rank.begin(), nrank_it);

        // Add (old global vertex index, new  global vertex index, owner
        // rank (global))
        send_buffer[rank_neighbor].insert(
            send_buffer[rank_neighbor].end(),
            {vlocal_it->first, vlocal_it->second + global_offset_v, mpi_rank});
      }
    }
  }

  return dolfinx::MPI::neighbor_all_to_all(
             comm, graph::AdjacencyList<std::int64_t>(send_buffer))
      .array();
}
//---------------------------------------------------------------------

/// Send vertex numbering of vertices in ghost cells to neighbours.
/// These include vertices that were numbered remotely and received in a
/// previous round. This is only needed for meshes with shared cells,
/// i.e. ghost_mode=shared_facet. Returns a list of triplets,
/// {old_global_vertex_index, new_global_vertex_index, owner}.
/// Input params as in mesh::create_topology()
/// @param[in] comm Neigborhood communicator
/// @return list of triplets
std::vector<std::int64_t> exchange_ghost_vertex_numbering(
    MPI_Comm comm, const xtl::span<const int>& local_to_global_rank,
    const common::IndexMap& index_map_c,
    const graph::AdjacencyList<std::int64_t>& cells, int nlocal,
    std::int64_t global_offset_v,
    const xtl::span<const std::pair<std::int64_t, std::int32_t>>&
        global_to_local_vertices,
    const xtl::span<const std::int64_t>& ghost_vertices,
    const xtl::span<const int>& ghost_vertex_owners)
{
  // Receive index of ghost vertices that are not on the process
  // ('true') boundary from the ghost cell owner. Note: the ghost cell
  // owner might not be the same as the vertex owner.

  // Build map from vertices of owned and shared cells to the global of
  // the ghosts
  std::map<std::int64_t, std::set<std::int32_t>> fwd_shared_vertices;
  {
    // Get indices of owned cells that are ghosted on other ranks
    const graph::AdjacencyList<std::int32_t>& fwd_shared_cells
        = index_map_c.scatter_fwd_indices();

    // Get ranks that ghost cells owned by this rank
    const std::vector<int> fwd_ranks = dolfinx::MPI::neighbors(
        index_map_c.comm(common::IndexMap::Direction::forward))[1];

    for (int r = 0; r < fwd_shared_cells.num_nodes(); ++r)
    {
      // Iterate over cells that are shared by rank r
      for (std::int32_t c : fwd_shared_cells.links(r))
      {
        // Vertices in local cells that are shared forward
        for (std::int32_t v : cells.links(c))
          fwd_shared_vertices[v].insert(fwd_ranks[r]);
      }
    }
  }

  // Compute sizes and offsets
  std::vector<int> send_sizes(local_to_global_rank.size()),
      sdispl(local_to_global_rank.size() + 1);
  for (const auto& vertex_ranks : fwd_shared_vertices)
  {
    for (int rank : vertex_ranks.second)
    {
      auto rank_it = std::lower_bound(local_to_global_rank.begin(),
                                      local_to_global_rank.end(), rank);
      assert(rank_it != local_to_global_rank.end());
      assert(*rank_it == rank);
      int rank_neighbor = std::distance(local_to_global_rank.begin(), rank_it);
      send_sizes[rank_neighbor] += 3;
    }
  }
  std::partial_sum(send_sizes.begin(), send_sizes.end(),
                   std::next(sdispl.begin()));
  std::vector<int> tmp_offsets(sdispl.begin(), sdispl.end());

  // Pack data for neighbor alltoall
  const int mpi_rank = dolfinx::MPI::rank(comm);
  std::vector<std::int64_t> send_triplet_data(sdispl.back());
  for (const auto& vertex_ranks : fwd_shared_vertices)
  {
    std::int64_t global_idx_old = vertex_ranks.first;
    auto it = find_idx(global_to_local_vertices, global_idx_old);
    assert(it != global_to_local_vertices.end());
    assert(it->first == global_idx_old);
    assert(it->second != -1);

    std::int64_t global_idx = it->second < nlocal
                                  ? it->second + global_offset_v
                                  : ghost_vertices[it->second - nlocal];
    int owner_rank = it->second < nlocal
                         ? mpi_rank
                         : ghost_vertex_owners[it->second - nlocal];
    for (int rank : vertex_ranks.second)
    {
      auto rank_it = std::lower_bound(local_to_global_rank.begin(),
                                      local_to_global_rank.end(), rank);
      assert(rank_it != local_to_global_rank.end());
      assert(*rank_it == rank);
      int np = std::distance(local_to_global_rank.begin(), rank_it);

      send_triplet_data[tmp_offsets[np]++] = global_idx_old;
      send_triplet_data[tmp_offsets[np]++] = global_idx;
      send_triplet_data[tmp_offsets[np]++] = owner_rank;
    }
  }

  return dolfinx::MPI::neighbor_all_to_all(
             comm, graph::AdjacencyList<std::int64_t>(
                       std::move(send_triplet_data), std::move(sdispl)))
      .array();
}
//---------------------------------------------------------------------------------

/// @brief Convert adjacency list edges from global indexing to local
/// indexing.
///
/// Nodes beyond `num_local_nodes` are discarded.
///
/// @param[in] g Graph with global edge indices
/// @param[in] num_local_nodes Number of nodes to retain in the graph.
/// Typically used to trim ghost nodes.
/// @param[in] global_to_local Sorted array of (global, local) indices.
graph::AdjacencyList<std::int32_t> convert_to_local_indexing(
    const graph::AdjacencyList<std::int64_t>& g, std::size_t num_local_nodes,
    const xtl::span<const std::pair<std::int64_t, std::int32_t>>&
        global_to_local)
{
  std::vector<std::int32_t> offsets(
      g.offsets().begin(), std::next(g.offsets().begin(), num_local_nodes + 1));

  std::vector<std::int32_t> data(offsets.back());
  std::transform(g.array().begin(), std::next(g.array().begin(), data.size()),
                 data.begin(),
                 [&global_to_local](auto i)
                 {
                   auto it = find_idx(global_to_local, i);
                   assert(it != global_to_local.end());
                   assert(it->first == i);
                   return it->second;
                 });

  return graph::AdjacencyList<std::int32_t>(std::move(data),
                                            std::move(offsets));
}
} // namespace

//-----------------------------------------------------------------------------
std::vector<std::int8_t> mesh::compute_boundary_facets(const Topology& topology)
{
  const int tdim = topology.dim();
  auto facets = topology.index_map(tdim - 1);
  if (!facets)
    throw std::runtime_error("Facets have not been computed.");

  std::set<std::int32_t> fwd_shared_facets;
  if (facets->num_ghosts() == 0)
  {
    fwd_shared_facets
        = std::set<std::int32_t>(facets->scatter_fwd_indices().array().begin(),
                                 facets->scatter_fwd_indices().array().end());
  }

  auto fc = topology.connectivity(tdim - 1, tdim);
  if (!fc)
    throw std::runtime_error("Facet-cell connectivity missing.");
  std::vector<std::int8_t> _boundary_facet(facets->size_local(), false);
  for (std::size_t f = 0; f < _boundary_facet.size(); ++f)
  {
    if (fc->num_links(f) == 1
        and fwd_shared_facets.find(f) == fwd_shared_facets.end())
    {
      _boundary_facet[f] = true;
    }
  }

  return _boundary_facet;
}
//-----------------------------------------------------------------------------
Topology::Topology(MPI_Comm comm, mesh::CellType type)
    : _comm(comm), _cell_type(type),
      _connectivity(
          cell_dim(type) + 1,
          std::vector<std::shared_ptr<graph::AdjacencyList<std::int32_t>>>(
              cell_dim(type) + 1))
{
  // Do nothing
}
//-----------------------------------------------------------------------------
int Topology::dim() const noexcept { return _connectivity.size() - 1; }
//-----------------------------------------------------------------------------
void Topology::set_index_map(int dim,
                             const std::shared_ptr<const common::IndexMap>& map)
{
  assert(dim < (int)_index_map.size());
  _index_map[dim] = map;
}
//-----------------------------------------------------------------------------
std::shared_ptr<const common::IndexMap> Topology::index_map(int dim) const
{
  assert(dim < (int)_index_map.size());
  return _index_map[dim];
}
//-----------------------------------------------------------------------------
std::int32_t Topology::create_entities(int dim)
{
  // TODO: is this check sufficient/correct? Does not catch the cell_entity
  // entity case. Should there also be a check for
  // connectivity(this->dim(), dim) ?
  // Skip if already computed (vertices (dim=0) should always exist)
  if (connectivity(dim, 0))
    return -1;

  // Create local entities
  const auto [cell_entity, entity_vertex, index_map]
      = compute_entities(_comm.comm(), *this, dim);

  if (cell_entity)
    set_connectivity(cell_entity, this->dim(), dim);

  // TODO: is this check necessary? Seems redundant after to the "skip check"
  if (entity_vertex)
    set_connectivity(entity_vertex, dim, 0);

  assert(index_map);
  this->set_index_map(dim, index_map);

  return index_map->size_local();
}
//-----------------------------------------------------------------------------
void Topology::create_connectivity(int d0, int d1)
{
  // Make sure entities exist
  create_entities(d0);
  create_entities(d1);

  // Compute connectivity
  const auto [c_d0_d1, c_d1_d0] = compute_connectivity(*this, d0, d1);

  // NOTE: that to compute the (d0, d1) connections is it sometimes
  // necessary to compute the (d1, d0) connections. We store the (d1,
  // d0) for possible later use, but there is a memory overhead if they
  // are not required. It may be better to not automatically store
  // connectivity that was not requested, but advise in a docstring the
  // most efficient order in which to call this function if several
  // connectivities are needed.

  // TODO: Caching policy/strategy.
  // Concerning the note above: Provide an overload
  // create_connectivity(std::vector<std::pair<int, int>>)?

  // Attach connectivities
  if (c_d0_d1)
    set_connectivity(c_d0_d1, d0, d1);
  if (c_d1_d0)
    set_connectivity(c_d1_d0, d1, d0);
}
//-----------------------------------------------------------------------------
void Topology::create_entity_permutations()
{
  if (!_cell_permutations.empty())
    return;

  const int tdim = this->dim();

  // FIXME: Is this always required? Could it be made cheaper by doing a
  // local version? This call does quite a lot of parallel work
  // Create all mesh entities

  for (int d = 0; d < tdim; ++d)
    create_entities(d);

  auto [facet_permutations, cell_permutations]
      = compute_entity_permutations(*this);
  _facet_permutations = std::move(facet_permutations);
  _cell_permutations = std::move(cell_permutations);
}
//-----------------------------------------------------------------------------
std::shared_ptr<const graph::AdjacencyList<std::int32_t>>
Topology::connectivity(int d0, int d1) const
{
  assert(d0 < (int)_connectivity.size());
  assert(d1 < (int)_connectivity[d0].size());
  return _connectivity[d0][d1];
}
//-----------------------------------------------------------------------------
void Topology::set_connectivity(
    std::shared_ptr<graph::AdjacencyList<std::int32_t>> c, int d0, int d1)
{
  assert(d0 < (int)_connectivity.size());
  assert(d1 < (int)_connectivity[d0].size());
  _connectivity[d0][d1] = c;
}
//-----------------------------------------------------------------------------
const std::vector<std::uint32_t>& Topology::get_cell_permutation_info() const
{
  // Check if this process owns or ghosts any cells
  assert(this->index_map(this->dim()));
  if (auto i_map = this->index_map(this->dim());
      _cell_permutations.empty()
      and i_map->size_local() + i_map->num_ghosts() > 0)
  {
    throw std::runtime_error(
        "create_entity_permutations must be called before using this data.");
  }

  return _cell_permutations;
}
//-----------------------------------------------------------------------------
const std::vector<std::uint8_t>& Topology::get_facet_permutations() const
{
  if (auto i_map = this->index_map(this->dim() - 1);
      !i_map
      or (_facet_permutations.empty()
          and i_map->size_local() + i_map->num_ghosts() > 0))
  {
    throw std::runtime_error(
        "create_entity_permutations must be called before using this data.");
  }

  return _facet_permutations;
}
//-----------------------------------------------------------------------------
mesh::CellType Topology::cell_type() const noexcept { return _cell_type; }
//-----------------------------------------------------------------------------
MPI_Comm Topology::comm() const { return _comm.comm(); }
//-----------------------------------------------------------------------------
Topology
mesh::create_topology(MPI_Comm comm,
                      const graph::AdjacencyList<std::int64_t>& cells,
                      const xtl::span<const std::int64_t>& original_cell_index,
                      const xtl::span<const int>& ghost_owners,
                      const CellType& cell_type, mesh::GhostMode ghost_mode)
{
  common::Timer timer("Topology: create");

  LOG(INFO) << "Create topology";
  if (cells.num_nodes() > 0
      and cells.num_links(0) != num_cell_vertices(cell_type))
  {
    throw std::runtime_error(
        "Inconsistent number of cell vertices. Got "
        + std::to_string(cells.num_links(0)) + ", expected "
        + std::to_string(num_cell_vertices(cell_type)) + ".");
  }

  // Create an index map for cells
  const std::int32_t num_local_cells = cells.num_nodes() - ghost_owners.size();
  std::shared_ptr<common::IndexMap> index_map_c;
  if (ghost_mode == GhostMode::none)
    index_map_c = std::make_shared<common::IndexMap>(comm, num_local_cells);
  else
  {
    // Get global indices of ghost cells
    xtl::span cell_idx(original_cell_index);
    const std::vector cell_ghost_indices = graph::build::compute_ghost_indices(
        comm, cell_idx.first(cells.num_nodes() - ghost_owners.size()),
        cell_idx.last(ghost_owners.size()), ghost_owners);

    // Determine src ranks
    std::vector<int> src_ranks(ghost_owners.begin(), ghost_owners.end());
    std::sort(src_ranks.begin(), src_ranks.end());
    src_ranks.erase(std::unique(src_ranks.begin(), src_ranks.end()),
                    src_ranks.end());
    auto dest_ranks = dolfinx::MPI::compute_graph_edges_nbx(comm, src_ranks);
    index_map_c = std::make_shared<common::IndexMap>(
        comm, num_local_cells, dest_ranks, cell_ghost_indices, ghost_owners);
  }

  // Create a map from global index to a label, using the labels:
  //
  // * -2 for owned (not shared with any ghost cells)
  // * -1 for all other vertices (shared by a ghost cell)
  //
  // and a list of vertices whose ownership needs determining (vertices
  // that are attached to both owned and ghost cells)
  auto [global_to_local_vertices, unknown_indices_set]
      = compute_vertex_markers(cells, num_local_cells);
  std::sort(global_to_local_vertices.begin(), global_to_local_vertices.end());

  // FIXME: do we already have information in the ranks that we re-use
  // here?
  //
  // For each vertex whose ownership needs determining (indices in
  // unknown_indices_set), find the sharing ranks. The first index in
  // the vector of ranks is the owner as determined by
  // determine_sharing_ranks.
  const graph::AdjacencyList<int> global_vertex_to_ranks
      = determine_sharing_ranks(comm, unknown_indices_set);

  // Iterate over vertices that have 'unknown' ownership, and if flagged
  // as owned by determine_sharing_ranks update ownership status
  const int mpi_rank = dolfinx::MPI::rank(comm);
  for (std::size_t i = 0; i < unknown_indices_set.size(); ++i)
  {
    // Vertex is shared and owned by this rank if the first sharing rank
    // is my rank
    auto ranks = global_vertex_to_ranks.links(i);
    assert(!ranks.empty());
    if (ranks.front() == mpi_rank)
    {
      // TODO: avoid map lookup

      std::int64_t global_index = unknown_indices_set[i];
      auto it_gi = find_idx(global_to_local_vertices, global_index);
      assert(it_gi != global_to_local_vertices.end());
      assert(it_gi->first == global_index);
      assert(it_gi->second == -1);

      // Mark vertex as owned
      it_gi->second = -2;
    }
  }

  // Number all owned vertices, iterating over vertices cell-wise
  std::int32_t v = 0;
  for (std::int32_t c = 0; c < cells.num_nodes(); ++c)
  {
    for (auto vtx : cells.links(c))
    {
      auto it = find_idx(global_to_local_vertices, vtx);
      assert(it != global_to_local_vertices.end());
      assert(it->first == vtx);
      if (it->second == -2)
        it->second = v++;
    }
  }

  // Compute the global offset for local vertex indices
  std::int64_t global_offset_v = 0;
  const std::int64_t nlocal = v;
  MPI_Exscan(&nlocal, &global_offset_v, 1, MPI_INT64_T, MPI_SUM, comm);

  // FIXME: Has this already been computed elsewhere?
  // FIXME: Does the communicator have to be 'symmetric?
  //
  // Build list of neighborhood ranks and create a neighborhood
  // communicator for vertices
  std::vector<int> local_to_global_rank(global_vertex_to_ranks.array().begin(),
                                        global_vertex_to_ranks.array().end());
  dolfinx::radix_sort(xtl::span(local_to_global_rank));
  local_to_global_rank.erase(
      std::unique(local_to_global_rank.begin(), local_to_global_rank.end()),
      local_to_global_rank.end());
  MPI_Comm neighbor_comm;
  MPI_Dist_graph_create_adjacent(
      comm, local_to_global_rank.size(), local_to_global_rank.data(),
      MPI_UNWEIGHTED, local_to_global_rank.size(), local_to_global_rank.data(),
      MPI_UNWEIGHTED, MPI_INFO_NULL, false, &neighbor_comm);

  // Send and receive list of triplets map (input vertex index) -> (new
  // global index, owner rank) with neighbours (for vertices on 'true
  // domain boundary')
  auto recv_triplets = exchange_vertex_numbering(
      neighbor_comm, local_to_global_rank, unknown_indices_set,
      global_vertex_to_ranks, global_offset_v, global_to_local_vertices);
  assert(recv_triplets.size() % 3 == 0);

  // Unpack received data and build array of ghost vertices and owners
  // of the ghost vertices
  std::vector<std::int64_t> ghost_vertices;
  std::vector<int> ghost_vertex_owners;
  for (std::size_t i = 0; i < recv_triplets.size(); i += 3)
  {
    assert(i + 2 < recv_triplets.size());
    const std::int64_t gi = recv_triplets[i];

    auto it = find_idx(global_to_local_vertices, gi);
    assert(it != global_to_local_vertices.end());
    assert(it->first == gi);
    assert(it->second == -1);
    it->second = v++;

    ghost_vertices.push_back(recv_triplets[i + 1]);
    ghost_vertex_owners.push_back(recv_triplets[i + 2]);
  }

  if (ghost_mode != GhostMode::none)
  {
    // Send and receive global (from the ghost cell owner) indices for
    // ghost vertices that are not on the process boundary.
    //
    // Note: the ghost cell owner might not be the same as the vertex
    // owner
    const std::vector<std::int64_t> recv_triplets
        = exchange_ghost_vertex_numbering(
            neighbor_comm, local_to_global_rank, *index_map_c, cells, nlocal,
            global_offset_v, global_to_local_vertices, ghost_vertices,
            ghost_vertex_owners);

    // Unpack received data and add to arrays of ghost indices and ghost
    // owners
    for (std::size_t i = 0; i < recv_triplets.size(); i += 3)
    {
      assert(i < recv_triplets.size());
      const std::int64_t global_idx_old = recv_triplets[i];
      auto it = find_idx(global_to_local_vertices, global_idx_old);
      assert(it != global_to_local_vertices.end());
      assert(it->first == global_idx_old);
      if (it->second == -1)
      {
        assert(i + 2 < recv_triplets.size());
        it->second = v++;
        ghost_vertices.push_back(recv_triplets[i + 1]);
        ghost_vertex_owners.push_back(recv_triplets[i + 2]);
      }
    }
  }

  MPI_Comm_free(&neighbor_comm);

  // TODO: has this neighbourhood information been computed earlier?

  // Determine which ranks ghost data on this rank
  std::vector<int> out_edges;
  {
    std::vector<int> in_edges = ghost_vertex_owners;
    std::sort(in_edges.begin(), in_edges.end());
    in_edges.erase(std::unique(in_edges.begin(), in_edges.end()),
                   in_edges.end());
    out_edges = dolfinx::MPI::compute_graph_edges_nbx(comm, in_edges);
  }

  // Convert input cell topology to local vertex indexing
  const std::size_t num_local_nodes
      = (ghost_mode == GhostMode::none) ? num_local_cells : cells.num_nodes();
  std::shared_ptr<graph::AdjacencyList<std::int32_t>> cells_local_idx
      = std::make_shared<graph::AdjacencyList<std::int32_t>>(
          convert_to_local_indexing(cells, num_local_nodes,
                                    global_to_local_vertices));

  // Create Topology object

  Topology topology(comm, cell_type);
  const int tdim = topology.dim();

  auto index_map_v = std::make_shared<common::IndexMap>(
      comm, nlocal, out_edges, ghost_vertices, ghost_vertex_owners);
  auto c0 = std::make_shared<graph::AdjacencyList<std::int32_t>>(
      index_map_v->size_local() + index_map_v->num_ghosts());

  // Set vertex index map and 'connectivity'
  topology.set_index_map(0, index_map_v);
  topology.set_connectivity(c0, 0, 0);

  // Set cell index map and connectivity
  topology.set_index_map(tdim, index_map_c);
  topology.set_connectivity(cells_local_idx, tdim, 0);

  return topology;
}
//-----------------------------------------------------------------------------
std::vector<std::int32_t>
mesh::entities_to_index(const mesh::Topology& topology, int dim,
                        const graph::AdjacencyList<std::int32_t>& entities)
{
  LOG(INFO) << "Build list if mesh entity indices from the entity vertices.";

  // Tagged entity topological dimension
  auto map_e = topology.index_map(dim);
  if (!map_e)
  {
    throw std::runtime_error("Mesh entities of dimension " + std::to_string(dim)
                             + "have not been created.");
  }

  auto e_to_v = topology.connectivity(dim, 0);
  assert(e_to_v);

  const int num_vertices_per_entity = mesh::cell_num_entities(
      mesh::cell_entity_type(topology.cell_type(), dim, 0), 0);

  // Build map from ordered local vertex indices (key) to entity index
  // (value)
  std::map<std::vector<std::int32_t>, std::int32_t> entity_key_to_index;
  std::vector<std::int32_t> key(num_vertices_per_entity);
  const int num_entities_mesh = map_e->size_local() + map_e->num_ghosts();
  for (int e = 0; e < num_entities_mesh; ++e)
  {
    auto vertices = e_to_v->links(e);
    std::copy(vertices.begin(), vertices.end(), key.begin());
    std::sort(key.begin(), key.end());
    auto ins = entity_key_to_index.insert({key, e});
    if (!ins.second)
      throw std::runtime_error("Duplicate mesh entity detected.");
  }

  // Iterate over all entities and find index
  std::vector<std::int32_t> indices;
  indices.reserve(entities.num_nodes());
  std::vector<std::int32_t> vertices(num_vertices_per_entity);
  for (std::int32_t e = 0; e < entities.num_nodes(); ++e)
  {
    auto v = entities.links(e);
    assert(num_vertices_per_entity == entities.num_links(e));
    std::copy(v.begin(), v.end(), vertices.begin());
    std::sort(vertices.begin(), vertices.end());

    if (auto it = entity_key_to_index.find(vertices);
        it != entity_key_to_index.end())
    {
      indices.push_back(it->second);
    }
    else
      indices.push_back(-1);
  }

  return indices;
}
//-----------------------------------------------------------------------------