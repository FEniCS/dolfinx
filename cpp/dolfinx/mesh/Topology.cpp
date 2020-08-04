// Copyright (C) 2006-2020 Anders Logg and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "Topology.h"
#include "Partitioning.h"
#include "PermutationComputation.h"
#include "TopologyComputation.h"
#include "utils.h"
#include <algorithm>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/log.h>
#include <dolfinx/common/utils.h>
#include <dolfinx/fem/ElementDofLayout.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/graph/Partitioning.h>
#include <dolfinx/mesh/Mesh.h>
#include <numeric>
#include <unordered_map>

using namespace dolfinx;
using namespace dolfinx::mesh;

namespace
{
//-----------------------------------------------------------------------------
// Compute list of processes sharing the same index
// @param unknown_indices List of indices on each process
// @return a map to sharing processes for each index, with the (random) owner as
// the first in the list
std::unordered_map<std::int64_t, std::vector<int>>
compute_index_sharing(MPI_Comm comm, std::vector<std::int64_t>& unknown_indices)
{
  const int mpi_size = dolfinx::MPI::size(comm);
  // Create a global address space to use with all_to_all post-office
  // algorithm and find the owner of each index within that space
  std::int64_t global_space = 0;
  std::int64_t max_index = 0;
  if (!unknown_indices.empty())
  {
    max_index
        = *std::max_element(unknown_indices.begin(), unknown_indices.end());
  }
  MPI_Allreduce(&max_index, &global_space, 1, MPI_INT64_T, MPI_SUM, comm);
  global_space += 1;

  std::vector<std::vector<std::int64_t>> send_indices(mpi_size);
  for (std::int64_t global_i : unknown_indices)
  {
    const int index_owner
        = dolfinx::MPI::index_owner(mpi_size, global_i, global_space);
    send_indices[index_owner].push_back(global_i);
  }

  const graph::AdjacencyList<std::int64_t> recv_indices
      = dolfinx::MPI::all_to_all(
          comm, graph::AdjacencyList<std::int64_t>(send_indices));

  // Get index sharing - ownership will be first entry (randomised later)
  std::unordered_map<std::int64_t, std::vector<int>> index_to_owner;
  for (int p = 0; p < recv_indices.num_nodes(); ++p)
  {
    auto recv_p = recv_indices.links(p);
    for (int j = 0; j < recv_p.rows(); ++j)
      index_to_owner[recv_p[j]].push_back(p);
  }

  std::mt19937 g(0);
  // Randomise ownership
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
    for (int j = 0; j < recv_p.rows(); ++j)
    {
      const auto it = index_to_owner.find(recv_p[j]);
      assert(it != index_to_owner.end());
      const std::vector<int>& sharing_procs = it->second;
      send_owner[p].push_back(sharing_procs.size());
      for (int sp : sharing_procs)
        send_owner[p].push_back(sp);
    }
  }

  // Alltoall is necessary because cells which are shared by vertex are not yet
  // known to this process
  const graph::AdjacencyList<int> recv_owner
      = dolfinx::MPI::all_to_all(comm, graph::AdjacencyList<int>(send_owner));

  // Now fill index_to_owner with locally needed indices
  index_to_owner.clear();
  for (int p = 0; p < mpi_size; ++p)
  {
    const std::vector<std::int64_t>& send_v = send_indices[p];
    auto r_owner = recv_owner.links(p);
    int c(0), i(0);
    while (c < r_owner.rows())
    {
      int count = r_owner[c++];
      for (int j = 0; j < count; ++j)
        index_to_owner[send_v[i]].push_back(r_owner[c++]);
      ++i;
    }
  }

  return index_to_owner;
}
} // namespace

//-----------------------------------------------------------------------------
std::vector<bool> mesh::compute_boundary_facets(const Topology& topology)
{
  const int tdim = topology.dim();

  auto facets = topology.index_map(tdim - 1);
  if (!facets)
    throw std::runtime_error("Facets have not been computed.");
  std::set<std::int32_t> fwd_shared_facets;
  if (facets->num_ghosts() == 0)
  {
    fwd_shared_facets = std::set<std::int32_t>(facets->shared_indices().begin(),
                                               facets->shared_indices().end());
  }

  std::shared_ptr<const graph::AdjacencyList<std::int32_t>> fc
      = topology.connectivity(tdim - 1, tdim);
  if (!fc)
    throw std::runtime_error("Facet-cell connectivity missing.");
  std::vector<bool> _boundary_facet(facets->size_local(), false);
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

//-----------------------------------------------------------------------------
int Topology::dim() const { return _connectivity.rows() - 1; }
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
      = TopologyComputation::compute_entities(_mpi_comm.comm(), *this, dim);

  if (cell_entity)
    set_connectivity(cell_entity, this->dim(), dim);

  // TODO: is this check necessary? Seems redundant after to the "skip check"
  if (entity_vertex)
    set_connectivity(entity_vertex, dim, 0);

  assert(index_map);
  set_index_map(dim, index_map);

  return index_map->size_local();
}
//-----------------------------------------------------------------------------
void Topology::create_connectivity(int d0, int d1)
{
  // Make sure entities exist
  create_entities(d0);
  create_entities(d1);

  // Compute connectivity
  const auto [c_d0_d1, c_d1_d0]
      = TopologyComputation::compute_connectivity(*this, d0, d1);

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
  if (_cell_permutations.size() > 0)
    return;

  const int tdim = this->dim();

  // FIXME: Is this always required? Could it be made cheaper by doing a
  // local version? This call does quite a lot of parallel work
  // Create all mesh entities
  for (int d = 0; d < tdim; ++d)
    create_entities(d);

  auto [facet_permutations, cell_permutations]
      = PermutationComputation::compute_entity_permutations(*this);
  _facet_permutations = std::move(facet_permutations);
  _cell_permutations = std::move(cell_permutations);
}
//-----------------------------------------------------------------------------
void Topology::create_connectivity_all()
{
  // Compute all entities
  for (int d = 0; d <= dim(); d++)
    create_entities(d);

  // Compute all connectivity
  for (int d0 = 0; d0 <= dim(); d0++)
    for (int d1 = 0; d1 <= dim(); d1++)
      create_connectivity(d0, d1);
}
//-----------------------------------------------------------------------------
std::shared_ptr<const graph::AdjacencyList<std::int32_t>>
Topology::connectivity(int d0, int d1) const
{
  assert(d0 < _connectivity.rows());
  assert(d1 < _connectivity.cols());
  return _connectivity(d0, d1);
}
//-----------------------------------------------------------------------------
void Topology::set_connectivity(
    std::shared_ptr<graph::AdjacencyList<std::int32_t>> c, int d0, int d1)
{
  assert(d0 < _connectivity.rows());
  assert(d1 < _connectivity.cols());
  _connectivity(d0, d1) = c;
}
//-----------------------------------------------------------------------------
size_t Topology::hash() const
{
  if (!this->connectivity(dim(), 0))
    throw std::runtime_error("AdjacencyList has not been computed.");
  return this->connectivity(dim(), 0)->hash();
}
//-----------------------------------------------------------------------------
const Eigen::Array<std::uint32_t, Eigen::Dynamic, 1>&
Topology::get_cell_permutation_info() const
{
  if (_cell_permutations.size() == 0)
  {
    throw std::runtime_error(
        "create_entity_permutations must be called before using this data.");
  }
  return _cell_permutations;
}
//-----------------------------------------------------------------------------
const Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>&
Topology::get_facet_permutations() const
{
  if (_cell_permutations.size() == 0)
  {
    throw std::runtime_error(
        "create_entity_permutations must be called before using this data.");
  }
  return _facet_permutations;
}
//-----------------------------------------------------------------------------
mesh::CellType Topology::cell_type() const { return _cell_type; }
//-----------------------------------------------------------------------------
MPI_Comm Topology::mpi_comm() const { return _mpi_comm.comm(); }
//-----------------------------------------------------------------------------
Topology
mesh::create_topology(MPI_Comm comm,
                      const graph::AdjacencyList<std::int64_t>& cells,
                      const std::vector<std::int64_t>& original_cell_index,
                      const std::vector<int>& ghost_owners,
                      const CellType& cell_type, mesh::GhostMode ghost_mode)
{
  LOG(INFO) << "Create topology";

  if (cells.num_nodes() > 0
      and cells.num_links(0) != mesh::num_cell_vertices(cell_type))
  {
    throw std::runtime_error(
        "Inconsistent number of cell vertices. Got "
        + std::to_string(cells.num_links(0)) + ", expected "
        + std::to_string(mesh::num_cell_vertices(cell_type)) + ".");
  }

  // Create cell IndexMap
  const int num_local_cells = cells.num_nodes() - ghost_owners.size();
  std::shared_ptr<common::IndexMap> index_map_c;
  if (ghost_mode == mesh::GhostMode::none)
    index_map_c = std::make_shared<common::IndexMap>(comm, num_local_cells, 1);
  else
  {
    // Get global indices of ghost cells
    const std::vector cell_ghost_indices
        = graph::Partitioning::compute_ghost_indices(comm, original_cell_index,
                                                     ghost_owners);
    index_map_c = std::make_shared<common::IndexMap>(
        comm, num_local_cells,
        dolfinx::MPI::compute_graph_edges(
            comm, std::set<int>(ghost_owners.begin(), ghost_owners.end())),
        cell_ghost_indices, ghost_owners, 1);
  }

  // Create map from existing global vertex index to local index,
  // putting ghost indices last
  std::unordered_map<std::int64_t, std::int32_t> global_to_local_index;

  // Any vertices which are in ghost cells set to -1 since we need to
  // determine ownership
  for (std::size_t i = 0; i < ghost_owners.size(); ++i)
  {
    auto v = cells.links(num_local_cells + i);
    for (int j = 0; j < v.size(); ++j)
      global_to_local_index.insert({v[j], -1});
  }

  // Get all vertices which appear in both ghost and non-ghost cells
  // and vertices which are local
  // FIXME: optimize
  std::set<std::int64_t> ghost_boundary_vertices;
  std::set<std::int64_t> local_vertex_set;
  for (int i = 0; i < num_local_cells; ++i)
  {
    auto v = cells.links(i);
    for (int j = 0; j < v.size(); ++j)
    {
      if (auto it = global_to_local_index.find(v[j]);
          it != global_to_local_index.end())
      {
        ghost_boundary_vertices.insert(v[j]);
      }
      else
        local_vertex_set.insert(v[j]);
    }
  }

  int mpi_rank = MPI::rank(comm);

  // Make a list of all vertex indices whose ownership needs determining
  std::vector<std::int64_t> unknown_indices(ghost_boundary_vertices.begin(),
                                            ghost_boundary_vertices.end());
  std::unordered_map<std::int64_t, std::vector<int>> global_to_procs
      = compute_index_sharing(comm, unknown_indices);

  // Number all indices which this process now owns
  std::int32_t c = 0;
  for (std::int64_t global_index : local_vertex_set)
  {
    // Locally owned
    auto [it_ignore, insert]
        = global_to_local_index.insert({global_index, c++});
    assert(insert);
  }

  for (std::int64_t global_index : ghost_boundary_vertices)
  {
    const auto it = global_to_procs.find(global_index);
    assert(it != global_to_procs.end());

    // Shared and locally owned
    if (it->second[0] == mpi_rank)
    {
      // Should already be in map, but needs index
      auto it_gi = global_to_local_index.find(global_index);
      assert(it_gi != global_to_local_index.end());
      assert(it_gi->second == -1);
      it_gi->second = c++;
    }
  }
  const std::int32_t nlocal = c;

  // Get global offset for local indices
  std::int64_t global_offset = dolfinx::MPI::global_offset(comm, nlocal, true);

  // Find all vertex-sharing neighbors, and process-to-neighbor map
  std::set<int> vertex_neighbors;
  for (auto q : global_to_procs)
    vertex_neighbors.insert(q.second.begin(), q.second.end());
  vertex_neighbors.erase(mpi_rank);
  std::vector<int> neighbors(vertex_neighbors.begin(), vertex_neighbors.end());
  std::unordered_map<int, int> proc_to_neighbors;
  for (std::size_t i = 0; i < neighbors.size(); ++i)
    proc_to_neighbors.insert({neighbors[i], i});

  // Communicate new global index to neighbors
  MPI_Comm neighbor_comm;
  MPI_Dist_graph_create_adjacent(comm, neighbors.size(), neighbors.data(),
                                 MPI_UNWEIGHTED, neighbors.size(),
                                 neighbors.data(), MPI_UNWEIGHTED,
                                 MPI_INFO_NULL, false, &neighbor_comm);

  std::vector<std::vector<std::int64_t>> send_pairs(neighbors.size());
  for (const auto& q : global_to_procs)
  {
    const std::vector<int>& procs = q.second;
    if (procs[0] == mpi_rank)
    {
      auto it = global_to_local_index.find(q.first);
      assert(it != global_to_local_index.end());
      assert(it->second != -1);

      // Owned and shared with these processes
      // NB starting from 1. 0 is self.
      for (std::size_t j = 1; j < procs.size(); ++j)
      {
        int np = proc_to_neighbors[procs[j]];
        send_pairs[np].push_back(it->first);
        send_pairs[np].push_back(it->second + global_offset);
      }
    }
  }

  std::vector<int> qsend_offsets = {0};
  std::vector<std::int64_t> qsend_data;
  for (const std::vector<std::int64_t>& q : send_pairs)
  {
    qsend_data.insert(qsend_data.end(), q.begin(), q.end());
    qsend_offsets.push_back(qsend_data.size());
  }

  Eigen::Array<std::int64_t, Eigen::Dynamic, 1> recv_pairs
      = dolfinx::MPI::neighbor_all_to_all(neighbor_comm, qsend_offsets,
                                          qsend_data)
            .array();

  std::vector<std::int64_t> ghost_vertices;
  // Unpack received data and make list of ghosts
  for (int i = 0; i < recv_pairs.rows(); i += 2)
  {
    std::int64_t gi = recv_pairs[i];
    const auto it = global_to_local_index.find(gi);
    assert(it != global_to_local_index.end());
    assert(it->second == -1);
    it->second = c++;
    ghost_vertices.push_back(recv_pairs[i + 1]);
  }

  if (ghost_mode != mesh::GhostMode::none)
  {
    // Receive index of ghost vertices that are not on the process
    // boundary from the ghost cell owner. Note: the ghost cell owner
    // might not be the same as the vertex owner.
    std::map<std::int32_t, std::set<std::int32_t>> shared_cells
        = index_map_c->compute_shared_indices();
    std::map<std::int64_t, std::set<std::int32_t>> fwd_shared_vertices;
    for (int i = 0; i < index_map_c->size_local(); ++i)
    {
      if (auto it = shared_cells.find(i); it != shared_cells.end())
      {
        auto v = cells.links(i);
        for (int j = 0; j < v.size(); ++j)
        {
          if (auto vit = fwd_shared_vertices.find(v[j]);
              vit == fwd_shared_vertices.end())
          {
            fwd_shared_vertices.insert({v[j], it->second});
          }
          else
            vit->second.insert(it->second.begin(), it->second.end());
        }
      }
    }

    // Precompute sizes and offsets
    std::vector<int> send_sizes(neighbors.size());
    std::vector<int> send_offsets(neighbors.size() + 1);
    for (const auto& q : fwd_shared_vertices)
    {
      for (int p : q.second)
      {
        const int np = proc_to_neighbors[p];
        send_sizes[np] += 2;
      }
    }
    std::partial_sum(send_sizes.begin(), send_sizes.end(),
                     send_offsets.begin() + 1);
    std::vector<int> tmp_offsets(send_offsets.begin(), send_offsets.end());

    // Fill data for neighbor alltoall
    std::vector<std::int64_t> send_pair_data(send_offsets.back());
    for (const auto& q : fwd_shared_vertices)
    {
      auto it = global_to_local_index.find(q.first);
      assert(it != global_to_local_index.end());
      assert(it->second != -1);

      std::int64_t gi;
      if (it->second < nlocal)
        gi = it->second + global_offset;
      else
        gi = ghost_vertices[it->second - nlocal];

      for (int p : q.second)
      {
        const int np = proc_to_neighbors[p];
        send_pair_data[tmp_offsets[np]++] = q.first;
        send_pair_data[tmp_offsets[np]++] = gi;
      }
    }

    Eigen::Array<std::int64_t, Eigen::Dynamic, 1> recv_pairs
        = dolfinx::MPI::neighbor_all_to_all(neighbor_comm, send_offsets,
                                            send_pair_data)
              .array();

    // Unpack received data and add to ghosts
    for (int i = 0; i < recv_pairs.rows(); i += 2)
    {
      std::int64_t gi = recv_pairs[i];
      const auto it = global_to_local_index.find(gi);
      assert(it != global_to_local_index.end());
      if (it->second == -1)
      {
        it->second = c++;
        ghost_vertices.push_back(recv_pairs[i + 1]);
      }
    }
  }

  // Get global owners of ghost vertices
  // TODO: Get vertice owner from cell owner? Can use neighborhood
  // communication?
  int mpi_size = -1;
  MPI_Comm_size(neighbor_comm, &mpi_size);
  std::vector<std::int32_t> local_sizes(mpi_size);
  MPI_Allgather(&nlocal, 1, MPI_INT32_T, local_sizes.data(), 1, MPI_INT32_T,
                neighbor_comm);

  std::vector<std::int64_t> all_ranges(mpi_size + 1, 0);
  std::partial_sum(local_sizes.begin(), local_sizes.end(),
                   all_ranges.begin() + 1);

  // Compute rank of ghost owners
  std::vector<int> ghost_vertices_owners(ghost_vertices.size(), -1);
  for (size_t i = 0; i < ghost_vertices.size(); ++i)
  {
    auto it = std::upper_bound(all_ranges.begin(), all_ranges.end(),
                               ghost_vertices[i]);
    const int p = std::distance(all_ranges.begin(), it) - 1;
    ghost_vertices_owners[i] = p;
  }

  MPI_Comm_free(&neighbor_comm);

  const Eigen::Array<std::int64_t, Eigen::Dynamic, 1>& cells_array
      = cells.array();
  std::shared_ptr<graph::AdjacencyList<std::int32_t>> my_local_cells;
  if (ghost_mode == mesh::GhostMode::none)
  {
    // Convert non-ghost cells (global indexing) to my_local_cells (local
    // indexing) and discard ghost cells
    std::vector<std::int32_t> local_offsets(
        cells.offsets().data(), cells.offsets().data() + num_local_cells + 1);
    std::vector<std::int32_t> my_local_cells_array(local_offsets.back());
    for (std::size_t i = 0; i < my_local_cells_array.size(); ++i)
      my_local_cells_array[i] = global_to_local_index[cells_array[i]];
    my_local_cells = std::make_shared<graph::AdjacencyList<std::int32_t>>(
        my_local_cells_array, local_offsets);
  }
  else
  {
    // Convert my_cells (global indexing) to my_local_cells (local indexing)
    Eigen::Array<std::int32_t, Eigen::Dynamic, 1> my_local_cells_array(
        cells_array.size());
    for (int i = 0; i < my_local_cells_array.size(); ++i)
      my_local_cells_array[i] = global_to_local_index[cells_array[i]];
    my_local_cells = std::make_shared<graph::AdjacencyList<std::int32_t>>(
        my_local_cells_array, cells.offsets());
  }

  Topology topology(comm, cell_type);
  const int tdim = topology.dim();

  // Vertex IndexMap
  auto index_map_v = std::make_shared<common::IndexMap>(
      comm, nlocal,
      dolfinx::MPI::compute_graph_edges(
          comm, std::set<int>(ghost_vertices_owners.begin(),
                              ghost_vertices_owners.end())),
      ghost_vertices, ghost_vertices_owners, 1);
  topology.set_index_map(0, index_map_v);
  auto c0 = std::make_shared<graph::AdjacencyList<std::int32_t>>(
      index_map_v->size_local() + index_map_v->num_ghosts());
  topology.set_connectivity(c0, 0, 0);

  // Cell IndexMap
  topology.set_index_map(tdim, index_map_c);
  topology.set_connectivity(my_local_cells, tdim, 0);

  return topology;
}
//-----------------------------------------------------------------------------
