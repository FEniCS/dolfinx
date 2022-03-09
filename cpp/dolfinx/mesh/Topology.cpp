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
#include <unordered_map>

using namespace dolfinx;
using namespace dolfinx::mesh;

namespace
{
//-----------------------------------------------------------------------------

/// Compute list of processes sharing the same index
///
/// @note Collective
/// @param[in] comm MPI communicator
/// @param[in] unknown_idx List of indices on each process
/// @return a map to sharing processes for each index, with the (random)
/// owner as the first in the list
std::unordered_map<std::int64_t, std::vector<int>>
determine_sharing_ranks(MPI_Comm comm,
                        const xtl::span<const std::int64_t>& unknown_idx)
{
  const int mpi_size = dolfinx::MPI::size(comm);

  // Create a global address space to use with all_to_all post-office
  // algorithm and find the owner of each index within that space
  std::int64_t global_space = 0;
  std::int64_t max_index = 0;
  if (!unknown_idx.empty())
    max_index = *std::max_element(unknown_idx.begin(), unknown_idx.end());
  MPI_Allreduce(&max_index, &global_space, 1, MPI_INT64_T, MPI_MAX, comm);
  global_space += 1;

  std::vector<std::int32_t> send_sizes(mpi_size);
  std::vector<std::int32_t> send_offsets(mpi_size + 1, 0);
  for (std::int64_t global_i : unknown_idx)
  {
    const int index_owner
        = dolfinx::MPI::index_owner(mpi_size, global_i, global_space);
    send_sizes[index_owner]++;
  }
  std::partial_sum(send_sizes.begin(), send_sizes.end(),
                   std::next(send_offsets.begin()));

  std::vector<std::int64_t> send_indices(send_offsets.back());
  for (std::int64_t global_i : unknown_idx)
  {
    const int index_owner
        = dolfinx::MPI::index_owner(mpi_size, global_i, global_space);
    send_indices[send_offsets[index_owner]] = global_i;
    send_offsets[index_owner]++;
  }
  // Reset offsets
  send_offsets[0] = 0;
  std::partial_sum(send_sizes.begin(), send_sizes.end(),
                   std::next(send_offsets.begin()));

  LOG(INFO) << "Sending " << send_indices.size() << " indices";
  const graph::AdjacencyList<std::int64_t> send_index_adj(
      std::move(send_indices), std::move(send_offsets));

  const graph::AdjacencyList<std::int64_t> recv_indices
      = dolfinx::MPI::all_to_all(comm, send_index_adj);

  // Get index sharing - ownership will be first entry (randomised later)
  std::unordered_map<std::int64_t, std::vector<int>> index_to_owner;
  for (int p = 0; p < recv_indices.num_nodes(); ++p)
  {
    auto recv_p = recv_indices.links(p);
    for (std::size_t j = 0; j < recv_p.size(); ++j)
      index_to_owner[recv_p[j]].push_back(p);
  }

  // Randomise ownership
  std::mt19937 g(0);
  for (auto& map_entry : index_to_owner)
  {
    std::vector<int>& procs = map_entry.second;
    std::shuffle(procs.begin(), procs.end(), g);
  }

  // Send index ownership data back to all sharing processes
  std::vector<std::vector<int>> send_owner(mpi_size);
  for (int p = 0; p < recv_indices.num_nodes(); ++p)
  {
    auto recv_p = recv_indices.links(p);
    for (std::size_t j = 0; j < recv_p.size(); ++j)
    {
      const auto it = index_to_owner.find(recv_p[j]);
      assert(it != index_to_owner.end());
      const std::vector<int>& sharing_procs = it->second;
      send_owner[p].push_back(sharing_procs.size());
      send_owner[p].insert(send_owner[p].end(), sharing_procs.begin(),
                           sharing_procs.end());
    }
  }

  // Alltoall is necessary because cells which are shared by vertex are
  // not yet known to this process
  const graph::AdjacencyList<int> recv_owner
      = dolfinx::MPI::all_to_all(comm, graph::AdjacencyList<int>(send_owner));

  // Now fill index_to_owner with locally needed indices
  index_to_owner.clear();
  for (int p = 0; p < mpi_size; ++p)
  {
    xtl::span<const std::int64_t> send_v = send_index_adj.links(p);
    auto r_owner = recv_owner.links(p);
    std::size_t c(0), i(0);
    while (c < r_owner.size())
    {
      int count = r_owner[c++];
      for (int j = 0; j < count; ++j)
        index_to_owner[send_v[i]].push_back(r_owner[c++]);
      ++i;
    }
  }

  return index_to_owner;
}
//-----------------------------------------------------------------------------

/// Create a map from the 64-bit input vertex index to an index that
/// indicates:
///
/// * (-1) Vertex is connected to a ghost cell
/// * (-2) Vertex is connected to local cells only
///
/// The index of vertices that are connected to both owned and ghost
/// cells are added to a vector.
/// @param cells Input mesh topology
/// @param num_local_cells Number of local (non-ghost) cells
/// @return (global_index_to_maker for (-1) and (-2) cases, indices for
/// other vertices)
std::pair<std::unordered_map<std::int64_t, std::int32_t>,
          std::vector<std::int64_t>>
compute_vertex_markers(const graph::AdjacencyList<std::int64_t>& cells,
                       int num_local_cells)
{
  common::Timer t0(
      "Topology: mark vertices by type (owned, possibly owned, ghost)");

  // Build a set of 'local' cell vertices
  std::vector<std::int64_t> local_vertex_set(
      cells.array().begin(),
      std::next(cells.array().begin(), cells.offsets()[num_local_cells]));
  dolfinx::radix_sort(xtl::span(local_vertex_set));
  local_vertex_set.erase(
      std::unique(local_vertex_set.begin(), local_vertex_set.end()),
      local_vertex_set.end());

  // Build a set of ghost cell vertices
  std::vector<std::int64_t> ghost_vertex_set(
      std::next(cells.array().begin(), cells.offsets()[num_local_cells]),
      cells.array().end());
  dolfinx::radix_sort(xtl::span(ghost_vertex_set));
  ghost_vertex_set.erase(
      std::unique(ghost_vertex_set.begin(), ghost_vertex_set.end()),
      ghost_vertex_set.end());

  // Compute the intersection of local cell vertices and ghost cell
  // vertices

  // Any vertices which are in ghost cells set to -1
  std::unordered_map<std::int64_t, std::int32_t> global_to_local_v;
  std::transform(ghost_vertex_set.begin(), ghost_vertex_set.end(),
                 std::inserter(global_to_local_v, global_to_local_v.end()),
                 [](auto idx)
                 { return std::pair<std::int64_t, std::int32_t>(idx, -1); });

  std::vector<std::int64_t> unknown_indices_set;
  for (std::int64_t global_index : local_vertex_set)
  {
    // Check if already in a ghost cell
    if (auto it = global_to_local_v.find(global_index);
        it != global_to_local_v.end())
    {
      unknown_indices_set.push_back(global_index);
    }
    else
    {
      // This vertex is not shared: set to -2
      [[maybe_unused]] auto [it_ignore, insert]
          = global_to_local_v.insert({global_index, -2});
      assert(insert);
    }
  }

  return {std::move(global_to_local_v), std::move(unknown_indices_set)};
}
//-----------------------------------------------------------------------------

/// Compute a neighborhood comm from the ranks in
/// global_vertex_to_ranks, also returning a map from global rank number
/// to neighborhood rank
/// @note Collective
/// @param[in] comm The global communicator
/// @param[in] global_vertex_to_ranks Map from global vertex index to
/// sharing ranks
/// @return (neighbor_comm, global_to_neighbor_rank map)
std::pair<MPI_Comm, std::map<int, int>>
compute_neighbor_comm(const MPI_Comm& comm,
                      const std::unordered_map<std::int64_t, std::vector<int>>&
                          global_vertex_to_ranks)
{
  const int mpi_rank = dolfinx::MPI::rank(comm);

  // Create set of all ranks that share a vertex with this rank. Note
  // this can be 'wider' than the neighbor comm of shared cells.
  std::vector<int> neighbors;
  std::for_each(
      global_vertex_to_ranks.begin(), global_vertex_to_ranks.end(),
      [&neighbors](auto& q)
      { neighbors.insert(neighbors.end(), q.second.begin(), q.second.end()); });
  std::sort(neighbors.begin(), neighbors.end());
  neighbors.erase(std::unique(neighbors.begin(), neighbors.end()),
                  neighbors.end());

  // Remove self
  neighbors.erase(std::remove(neighbors.begin(), neighbors.end(), mpi_rank),
                  neighbors.end());

  // Build map from neighbor global rank to neighbor local rank
  std::map<int, int> global_to_neighbor_rank;
  for (std::size_t i = 0; i < neighbors.size(); ++i)
    global_to_neighbor_rank.insert({neighbors[i], i});

  // Create symmetric neighborhood communicator
  MPI_Comm neighbor_comm;
  MPI_Dist_graph_create_adjacent(comm, neighbors.size(), neighbors.data(),
                                 MPI_UNWEIGHTED, neighbors.size(),
                                 neighbors.data(), MPI_UNWEIGHTED,
                                 MPI_INFO_NULL, false, &neighbor_comm);

  return {neighbor_comm, std::move(global_to_neighbor_rank)};
}
//-------------------------------------------------------------------------------

/// Send the vertex numbering for owned vertices to processes that also
/// share them, returning a list of triplets received from other ranks.
/// Each triplet consists of {old_global_vertex_index,
/// new_global_vertex_index, owning_rank}. The received vertices will be
/// "ghost" on this rank.
/// Input params as in mesh::create_topology()
/// @note Collective
/// @param[in] comm Neighbourhood communicator
/// @return list of triplets
std::vector<std::int64_t> exchange_vertex_numbering(
    const MPI_Comm& comm, const std::map<int, int>& global_to_neighbor_rank,
    const std::unordered_map<std::int64_t, std::vector<int>>&
        global_vertex_to_ranks,
    std::int64_t global_offset_v,
    const std::unordered_map<std::int64_t, std::int32_t>&
        global_to_local_vertices)
{
  const int mpi_rank = dolfinx::MPI::rank(comm);

  // Pack send data
  std::vector<std::vector<std::int64_t>> send_buffer(
      global_to_neighbor_rank.size());
  for (const auto& vertex : global_vertex_to_ranks)
  {
    // Get (global) ranks that share this vertex. Note that first rank
    // is the owner.
    const std::vector<int>& vertex_ranks = vertex.second;
    if (vertex_ranks.front() == mpi_rank)
    {
      // Get local vertex index
      auto vlocal_it = global_to_local_vertices.find(vertex.first);
      assert(vlocal_it != global_to_local_vertices.end());
      assert(vlocal_it->second != -1);

      // Owned and shared with these processes (starting from 1, 0 is self)
      for (std::size_t j = 1; j < vertex_ranks.size(); ++j)
      {
        // Find rank on the neighborhood comm
        auto nrank_it = global_to_neighbor_rank.find(vertex_ranks[j]);
        assert(nrank_it != global_to_neighbor_rank.end());
        int rank_neighbor = nrank_it->second;

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
    MPI_Comm comm, const std::map<int, int>& global_to_neighbor_rank,
    const common::IndexMap& index_map_c,
    const graph::AdjacencyList<std::int64_t>& cells, int nlocal,
    std::int64_t global_offset_v,
    const std::unordered_map<std::int64_t, std::int32_t>&
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
  std::vector<int> send_sizes(global_to_neighbor_rank.size()),
      sdispl(global_to_neighbor_rank.size() + 1);
  for (const auto& vertex_ranks : fwd_shared_vertices)
  {
    for (int rank : vertex_ranks.second)
    {
      auto rank_it = global_to_neighbor_rank.find(rank);
      assert(rank_it != global_to_neighbor_rank.end());
      send_sizes[rank_it->second] += 3;
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
    auto it = global_to_local_vertices.find(global_idx_old);
    assert(it != global_to_local_vertices.end());
    assert(it->second != -1);
    std::int64_t global_idx = it->second < nlocal
                                  ? it->second + global_offset_v
                                  : ghost_vertices[it->second - nlocal];
    int owner_rank = it->second < nlocal
                         ? mpi_rank
                         : ghost_vertex_owners[it->second - nlocal];
    for (int rank : vertex_ranks.second)
    {
      auto rank_it = global_to_neighbor_rank.find(rank);
      assert(rank_it != global_to_neighbor_rank.end());
      int np = rank_it->second;
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
graph::AdjacencyList<std::int32_t> convert_cells_to_local_indexing(
    mesh::GhostMode ghost_mode, const graph::AdjacencyList<std::int64_t>& cells,
    std::int32_t num_local_cells,
    const std::unordered_map<std::int64_t, std::int32_t>
        global_to_local_vertices)
{
  std::vector<std::int32_t> local_offsets;
  if (ghost_mode == GhostMode::none)
  {
    // Discard ghost cells
    local_offsets.assign(
        cells.offsets().begin(),
        std::next(cells.offsets().begin(), num_local_cells + 1));
  }
  else
    local_offsets.assign(cells.offsets().begin(), cells.offsets().end());

  std::vector<std::int32_t> cells_array_local(local_offsets.back());
  std::transform(cells.array().begin(),
                 std::next(cells.array().begin(), cells_array_local.size()),
                 cells_array_local.begin(),
                 [&global_to_local_vertices](std::int64_t i)
                 { return global_to_local_vertices.at(i); });

  return graph::AdjacencyList<std::int32_t>(std::move(cells_array_local),
                                            std::move(local_offsets));
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
  if (_cell_permutations.empty())
  {
    throw std::runtime_error(
        "create_entity_permutations must be called before using this data.");
  }
  return _cell_permutations;
}
//-----------------------------------------------------------------------------
const std::vector<std::uint8_t>& Topology::get_facet_permutations() const
{
  if (_facet_permutations.empty())
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
  LOG(INFO) << "Create topology";
  if (cells.num_nodes() > 0
      and cells.num_links(0) != num_cell_vertices(cell_type))
  {
    throw std::runtime_error(
        "Inconsistent number of cell vertices. Got "
        + std::to_string(cells.num_links(0)) + ", expected "
        + std::to_string(num_cell_vertices(cell_type)) + ".");
  }

  // Create index map for cells
  const std::int32_t num_local_cells = cells.num_nodes() - ghost_owners.size();
  std::shared_ptr<common::IndexMap> index_map_c;
  if (ghost_mode == GhostMode::none)
    index_map_c = std::make_shared<common::IndexMap>(comm, num_local_cells);
  else
  {
    // Get global indices of ghost cells
    const std::vector cell_ghost_indices = graph::build::compute_ghost_indices(
        comm, original_cell_index, ghost_owners);

    // Determine src ranks
    std::vector<int> src_ranks(ghost_owners.begin(), ghost_owners.end());
    std::sort(src_ranks.begin(), src_ranks.end());
    src_ranks.erase(std::unique(src_ranks.begin(), src_ranks.end()),
                    src_ranks.end());
    auto dest_ranks = dolfinx::MPI::compute_graph_edges_nbx(comm, src_ranks);
    index_map_c = std::make_shared<common::IndexMap>(
        comm, num_local_cells, dest_ranks, cell_ghost_indices, ghost_owners);
  }

  // Create a map from global index to a label, using labels
  //
  // * -2 for owned (not shared with any ghost cells)
  // * -1 for all other vertices (shared by a ghost cell)
  //
  // and a list of vertices whose ownership needs determining (vertices
  // that are attached to both owned and ghost cells)
  auto [global_to_local_vertices, unknown_indices_set]
      = compute_vertex_markers(cells, num_local_cells);

  // For each vertex whose ownership needs determining (indices in
  // unknown_indices_set), compute the list of sharing ranks. The first
  // index in the vector of ranks is the owner as determined by
  // determine_sharing_ranks.
  std::unordered_map<std::int64_t, std::vector<int>> global_vertex_to_ranks
      = determine_sharing_ranks(comm, unknown_indices_set);

  // Iterate over vertices that have 'unknown' ownership, and if flagged
  // as owned by determine_sharing_ranks update ownership status
  const int mpi_rank = dolfinx::MPI::rank(comm);
  for (std::int64_t global_index : unknown_indices_set)
  {
    const auto it = global_vertex_to_ranks.find(global_index);
    assert(it != global_vertex_to_ranks.end());

    // Vertex is shared and owned by this rank if first owning rank is
    // my rank
    if (it->second[0] == mpi_rank)
    {
      // Should already be in map
      auto it_gi = global_to_local_vertices.find(global_index);
      assert(it_gi != global_to_local_vertices.end());
      assert(it_gi->second == -1);

      // Mark as locally owned
      it_gi->second = -2;
    }
  }

  // Number all owned vertices, iterating over vertices cell-wise
  std::int32_t v = 0;
  for (std::int32_t c = 0; c < cells.num_nodes(); ++c)
  {
    for (auto vtx : cells.links(c))
    {
      auto it = global_to_local_vertices.find(vtx);
      assert(it != global_to_local_vertices.end());
      if (it->second == -2)
        it->second = v++;
    }
  }

  // Compute the global offset for local vertex indices
  const std::int64_t nlocal = v;
  std::int64_t global_offset_v = 0;
  MPI_Exscan(&nlocal, &global_offset_v, 1, MPI_INT64_T, MPI_SUM, comm);

  // Create neighborhood communicator for vertices on the 'true'
  // boundary and a map from MPI rank on comm to rank on neighbor_comm
  auto [neighbor_comm, global_to_neighbor_rank]
      = compute_neighbor_comm(comm, global_vertex_to_ranks);

  // Send and receive list of triplets map (input vertex index) -> (new
  // global index, owner rank) with neighbours (for vertices on 'true
  // domain boundary')
  auto recv_triplets = exchange_vertex_numbering(
      neighbor_comm, global_to_neighbor_rank, global_vertex_to_ranks,
      global_offset_v, global_to_local_vertices);
  assert(recv_triplets.size() % 3 == 0);

  // Unpack received data and build array of ghost vertices and owners
  // of the ghost vertices
  std::vector<std::int64_t> ghost_vertices;
  std::vector<int> ghost_vertex_owners;
  for (std::size_t i = 0; i < recv_triplets.size(); i += 3)
  {
    assert(i + 2 < recv_triplets.size());
    const std::int64_t gi = recv_triplets[i];
    const auto it = global_to_local_vertices.find(gi);
    assert(it != global_to_local_vertices.end());
    assert(it->second == -1);
    it->second = v++;
    ghost_vertices.push_back(recv_triplets[i + 1]);
    ghost_vertex_owners.push_back(recv_triplets[i + 2]);
  }

  if (ghost_mode != GhostMode::none)
  {
    // Send and receive global (from the ghost cell owner) indices for
    // ghost vertices that are not on the process boundary.
    // Note: the ghost cell owner might not be the same as the vertex
    // owner
    const std::vector<std::int64_t> recv_triplets
        = exchange_ghost_vertex_numbering(
            neighbor_comm, global_to_neighbor_rank, *index_map_c, cells, nlocal,
            global_offset_v, global_to_local_vertices, ghost_vertices,
            ghost_vertex_owners);

    // Unpack received data and add to arrays of ghost indices and ghost
    // owners
    for (std::size_t i = 0; i < recv_triplets.size(); i += 3)
    {
      assert(i < recv_triplets.size());
      const std::int64_t global_idx_old = recv_triplets[i];
      const auto it = global_to_local_vertices.find(global_idx_old);
      assert(it != global_to_local_vertices.end());
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

  // TODO: is it possible to build neighbourhood communictor that is
  // larger than neighbor_comm to capture all ghost owners?

  // Determine which ranks ghost data on this rank by  sending '1' to
  // ranks that this rank has ghost vertices for
  std::vector<int> in_edges = ghost_vertex_owners;
  std::sort(in_edges.begin(), in_edges.end());
  in_edges.erase(std::unique(in_edges.begin(), in_edges.end()), in_edges.end());
  // Send '1' to ranks that I have an edge to
  std::vector<std::uint8_t> edge_count_send(dolfinx::MPI::size(comm), 0);
  std::for_each(in_edges.cbegin(), in_edges.cend(),
                [&edge_count_send](auto e) { edge_count_send[e] = 1; });
  MPI_Request request;
  std::vector<std::uint8_t> edge_count_recv(edge_count_send.size());
  MPI_Ialltoall(edge_count_send.data(), 1, MPI_UINT8_T, edge_count_recv.data(),
                1, MPI_UINT8_T, comm, &request);

  // Convert input cell topology to local vertex indexing
  std::shared_ptr<graph::AdjacencyList<std::int32_t>> cells_local_idx
      = std::make_shared<graph::AdjacencyList<std::int32_t>>(
          convert_cells_to_local_indexing(ghost_mode, cells, num_local_cells,
                                          global_to_local_vertices));

  // Create Topology object

  Topology topology(comm, cell_type);
  const int tdim = topology.dim();

  // Create vertex index map
  MPI_Wait(&request, MPI_STATUS_IGNORE);
  std::vector<int> out_edges;
  for (std::size_t i = 0; i < edge_count_recv.size(); ++i)
  {
    if (edge_count_recv[i] > 0)
      out_edges.push_back(i);
  }

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