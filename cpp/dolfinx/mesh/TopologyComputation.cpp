// Copyright (C) 2006-2020 Anders Logg, Garth N. Wells and Chris Richardson
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "TopologyComputation.h"
#include "Topology.h"
#include "cell_types.h"
#include <Eigen/Dense>
#include <algorithm>
#include <boost/unordered_map.hpp>
#include <cstdint>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/MPI.h>
#include <dolfinx/common/Timer.h>
#include <dolfinx/common/log.h>
#include <dolfinx/common/utils.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace dolfinx;
using namespace dolfinx::mesh;

namespace
{
//-----------------------------------------------------------------------------
/// Get the ownership of an entity shared over several processes
/// @param processes Set of sharing processes
/// @param vertices Global vertex indices of entity
/// @return owning process number
int get_ownership(std::set<int>& processes, std::vector<std::int64_t>& vertices)
{
  // Use a deterministic random number generator, seeded with global vertex
  // indices ensuring all processes get the same answer
  std::mt19937 gen;
  std::seed_seq seq(vertices.begin(), vertices.end());
  gen.seed(seq);
  std::vector<int> p(processes.begin(), processes.end());
  int index = gen() % p.size();
  int owner = p[index];
  return owner;
}

/// Takes an array and computes the sort permutation that would reorder
/// the rows in ascending order
/// @param[in] array The input array
/// @return The permutation vector that would order the rows in
///   ascending order
/// @pre Each row of @p array must be sorted
template <typename T>
std::vector<int>
sort_by_perm(const Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic,
                                Eigen::RowMajor>& array)
{
  // Sort an Eigen::Array by creating a permutation vector
  std::vector<int> index(array.rows());
  std::iota(index.begin(), index.end(), 0);
  const int cols = array.cols();

  // Lambda with capture for sort comparison
  const auto cmp = [&array, &cols](int a, int b) {
    const T* row_a = array.row(a).data();
    const T* row_b = array.row(b).data();
    return std::lexicographical_compare(row_a, row_a + cols, row_b,
                                        row_b + cols);
  };

  std::sort(index.begin(), index.end(), cmp);
  return index;
}
//-----------------------------------------------------------------------------

/// Communicate with sharing processes to find out which entities are
/// ghosts and return a map (vector) to move these local indices to the
/// end of the local range. Also returns the index map, and shared
/// entities, i.e. the set of all processes which share each shared
/// entity.
/// @param[in] comm MPI Communicator
/// @param[in] cell_indexmap IndexMap for cells
/// @param[in] vertex_indexmap IndexMap for vertices
/// @param[in] entity_list List of entities as 2D array, each entity
///   represented by its local vertex indices
/// @param[in] entity_index Initial numbering for each row in
///   entity_list
/// @returns Tuple of (local_indices, index map, shared entities)
std::tuple<std::vector<int>, std::shared_ptr<common::IndexMap>>
get_local_indexing(
    MPI_Comm comm, const std::shared_ptr<const common::IndexMap>& cell_indexmap,
    const std::shared_ptr<const common::IndexMap>& vertex_indexmap,
    const Eigen::Ref<const Eigen::Array<std::int32_t, Eigen::Dynamic,
                                        Eigen::Dynamic, Eigen::RowMajor>>&
        entity_list,
    const std::vector<std::int32_t>& entity_index)
{
  // Get first occurence in entity list of each entity
  std::vector<std::int32_t> unique_row(entity_list.rows(), -1);
  std::int32_t entity_count = 0;
  for (int i = 0; i < entity_list.rows(); ++i)
  {
    const std::int32_t idx = entity_index[i];
    if (unique_row[idx] == -1)
    {
      unique_row[idx] = i;
      ++entity_count;
    }
  }
  unique_row.resize(entity_count);

  //---------
  // Set ghost status array values
  // 1 = entities that are only in local cells (i.e. owned)
  // 2 = entities that are only in ghost cells (i.e. not owned)
  // 3 = entities with ownership that needs deciding (used also for unghosted
  // case)
  std::vector<std::int8_t> ghost_status(entity_count, 0);
  {
    if (cell_indexmap->num_ghosts() == 0)
      std::fill(ghost_status.begin(), ghost_status.end(), 3);
    else
    {
      std::int32_t cell_count
          = cell_indexmap->size_local() + cell_indexmap->num_ghosts();
      assert(entity_list.rows() % cell_count == 0);
      std::int32_t num_entities_per_cell = entity_list.rows() / cell_count;
      std::int32_t ghost_offset
          = cell_indexmap->size_local() * num_entities_per_cell;

      // Tag all entities in local cells with 1
      for (int i = 0; i < ghost_offset; ++i)
      {
        const std::int32_t idx = entity_index[i];
        ghost_status[idx] = 1;
      }
      // Set entities in ghost cells to 2 (purely ghost) or 3 (border)
      for (int i = ghost_offset; i < entity_list.rows(); ++i)
      {
        const std::int32_t idx = entity_index[i];
        ghost_status[idx] |= 2;
      }
    }
  }

  //---------
  // Create an expanded neighbor_comm from shared_vertices
  const std::map<std::int32_t, std::set<std::int32_t>> shared_vertices
      = vertex_indexmap->compute_shared_indices();

  std::set<std::int32_t> neighbor_set;
  for (auto& q : shared_vertices)
    neighbor_set.insert(q.second.begin(), q.second.end());
  std::vector<std::int32_t> neighbors(neighbor_set.begin(), neighbor_set.end());
  MPI_Comm neighbor_comm;
  MPI_Dist_graph_create_adjacent(comm, neighbors.size(), neighbors.data(),
                                 MPI_UNWEIGHTED, neighbors.size(),
                                 neighbors.data(), MPI_UNWEIGHTED,
                                 MPI_INFO_NULL, false, &neighbor_comm);

  const int neighbor_size = neighbors.size();
  std::unordered_map<int, int> proc_to_neighbor;
  for (int i = 0; i < neighbor_size; ++i)
    proc_to_neighbor.insert({neighbors[i], i});

  std::vector<std::vector<std::int64_t>> send_entities(neighbor_size);
  std::vector<std::vector<std::int32_t>> send_index(neighbor_size);

  // Get all "possibly shared" entities, based on vertex sharing. Send to
  // other processes, and see if we get the same back.

  // Map for entities to entity index, using global vertex indices
  std::map<std::vector<std::int64_t>, std::int32_t>
      global_entity_to_entity_index;

  // Set of sharing procs for each entity, counting vertex hits
  std::unordered_map<int, int> procs;
  const int num_vertices = entity_list.cols();
  std::vector<std::int32_t> vlocal(num_vertices);
  std::vector<std::int64_t> vglobal(num_vertices);
  for (int i : unique_row)
  {
    procs.clear();
    for (int j = 0; j < num_vertices; ++j)
    {
      const int v = entity_list(i, j);
      if (auto it = shared_vertices.find(v); it != shared_vertices.end())
      {
        for (std::int32_t p : it->second)
          ++procs[p];
      }
    }

    for (const auto& q : procs)
    {
      // If any process shares all vertices, then add to list
      if (q.second == num_vertices)
      {
        const int p = q.first;
        auto it = proc_to_neighbor.find(p);
        assert(it != proc_to_neighbor.end());
        const int np = it->second;

        vlocal.assign(entity_list.row(i).data(),
                      entity_list.row(i).data() + num_vertices);
        vglobal = vertex_indexmap->local_to_global(vlocal, false);
        std::sort(vglobal.begin(), vglobal.end());

        global_entity_to_entity_index.insert({vglobal, entity_index[i]});

        // Do not send entities which are known to be ghosts
        if (ghost_status[entity_index[i]] != 2)
        {
          // Entity entity_index[i] may be shared with process p
          send_entities[np].insert(send_entities[np].end(), vglobal.begin(),
                                   vglobal.end());

          send_index[np].push_back(entity_index[i]);
        }
      }
    }
  }

  // Get shared entities of this dimension, and also match up an index
  // for the received entities (from other processes) with the indices
  // of the sent entities (to other processes)
  std::unordered_map<std::int32_t, std::set<std::int32_t>> shared_entities;
  std::unordered_map<std::int32_t, std::vector<std::int64_t>>
      shared_entity_to_global_vertices;

  // Prepare data for neighbor all to all
  std::vector<std::int64_t> send_entities_data;
  std::vector<int> send_offsets = {0};
  for (std::size_t i = 0; i < send_entities.size(); ++i)
  {
    send_entities_data.insert(send_entities_data.end(),
                              send_entities[i].begin(), send_entities[i].end());
    send_offsets.push_back(send_entities_data.size());
  }

  const graph::AdjacencyList<std::int64_t> recv_data
      = dolfinx::MPI::neighbor_all_to_all(neighbor_comm, send_offsets,
                                          send_entities_data);
  const Eigen::Array<std::int64_t, Eigen::Dynamic, 1>& recv_entities_data
      = recv_data.array();
  const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>& recv_offsets
      = recv_data.offsets();

  // Compare received with sent for each process
  // Any which are not found will have -1 in recv_index
  std::vector<std::int32_t> recv_index;
  std::vector<std::int64_t> recv_vec(num_vertices);
  for (int np = 0; np < neighbor_size; ++np)
  {
    const int p = neighbors[np];

    for (int j = recv_offsets[np]; j < recv_offsets[np + 1]; j += num_vertices)
    {
      recv_vec.assign(recv_entities_data.data() + j,
                      recv_entities_data.data() + j + num_vertices);
      if (auto it = global_entity_to_entity_index.find(recv_vec);
          it != global_entity_to_entity_index.end())
      {
        shared_entities[it->second].insert(p);
        shared_entity_to_global_vertices.insert({it->second, recv_vec});
        recv_index.push_back(it->second);
      }
      else
        recv_index.push_back(-1);
    }
  }

  // Add this rank to the list of sharing processes
  const int mpi_rank = dolfinx::MPI::rank(comm);
  for (auto& q : shared_entities)
    q.second.insert(mpi_rank);

  //---------
  // Determine ownership
  std::vector<std::int32_t> local_index(entity_count, -1);
  std::int32_t num_local;
  {
    std::int32_t c = 0;
    // Index non-ghost entities
    for (int i = 0; i < entity_count; ++i)
    {
      auto it = shared_entities.find(i);
      std::int8_t gs = ghost_status[i];
      assert(gs > 0);
      // Definitely ghost
      if (gs == 2)
        continue;

      // Definitely local
      if (gs == 1 or it == shared_entities.end())
      {
        local_index[i] = c;
        ++c;
      }
      else
      {
        const auto global_vertices_it
            = shared_entity_to_global_vertices.find(i);
        assert(global_vertices_it != shared_entity_to_global_vertices.end());
        int owner_rank = get_ownership(it->second, global_vertices_it->second);
        if (owner_rank == mpi_rank)
        {
          // Take ownership
          local_index[i] = c;
          ++c;
        }
      }
    }
    num_local = c;

    for (int i = 0; i < entity_count; ++i)
    {
      // Unmapped global index (ghost)
      if (local_index[i] == -1)
      {
        local_index[i] = c;
        ++c;
      }
    }
    assert(c == entity_count);
  }

  //---------
  // Communicate global indices to other processes
  std::vector<int> ghost_owners(entity_count - num_local, -1);
  std::vector<std::int64_t> ghost_indices(entity_count - num_local, -1);
  {
    const std::int64_t local_offset
        = dolfinx::MPI::global_offset(comm, num_local, true);

    std::vector<std::int64_t> send_global_index_data;
    std::vector<int> send_global_index_offsets = {0};

    // Send global indices for same entities that we sent before. This
    // uses the same pattern as before, so we can match up the received
    // data to the indices in recv_index
    for (int np = 0; np < neighbor_size; ++np)
    {
      for (std::int32_t index : send_index[np])
      {
        // If not in our local range, send -1.
        const std::int64_t gi = (local_index[index] < num_local)
                                    ? (local_offset + local_index[index])
                                    : -1;

        send_global_index_data.push_back(gi);
      }

      send_global_index_offsets.push_back(send_global_index_data.size());
    }
    const graph::AdjacencyList<std::int64_t> recv_data
        = dolfinx::MPI::neighbor_all_to_all(
            neighbor_comm, send_global_index_offsets, send_global_index_data);

    const Eigen::Array<std::int64_t, Eigen::Dynamic, 1>& recv_global_index_data
        = recv_data.array();
    const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>& recv_offsets
        = recv_data.offsets();

    assert(recv_global_index_data.size() == (int)recv_index.size());

    // Map back received indices
    for (int j = 0; j < recv_global_index_data.size(); ++j)
    {
      const std::int64_t gi = recv_global_index_data[j];
      const std::int32_t idx = recv_index[j];
      if (gi != -1 and idx != -1)
      {
        assert(local_index[idx] >= num_local);
        ghost_indices[local_index[idx] - num_local] = gi;
        const auto pos = std::upper_bound(
            recv_offsets.data(), recv_offsets.data() + recv_offsets.rows(), j);
        const int owner = std::distance(recv_offsets.data(), pos) - 1;
        ghost_owners[local_index[idx] - num_local] = neighbors[owner];
      }
    }
    for (std::int64_t idx : ghost_indices)
      assert(idx != -1);
  }

  MPI_Comm_free(&neighbor_comm);

  auto index_map = std::make_shared<common::IndexMap>(
      comm, num_local,
      dolfinx::MPI::compute_graph_edges(
          comm, std::set<int>(ghost_owners.begin(), ghost_owners.end())),
      ghost_indices, ghost_owners, 1);

  // Map from initial numbering to new local indices
  std::vector<std::int32_t> new_entity_index(entity_index.size());
  for (std::size_t i = 0; i < entity_index.size(); ++i)
    new_entity_index[i] = local_index[entity_index[i]];

  return {std::move(new_entity_index), index_map};
}
//-----------------------------------------------------------------------------

/// Compute entities of dimension d
/// @param[in] comm MPI communicator (TODO: full or neighbor hood?)
/// @param[in] cells Adjacency list for cell-vertex connectivity
/// @param[in] shared_vertices TODO
/// @param[in] cell_type Cell type
/// @param[in] dim Topological dimension of the entities to be computed
/// @return Returns the (cell-entity connectivity, entity-cell
///   connectivity, index map for the entity distribution across
///   processes, shared entities)
std::tuple<std::shared_ptr<graph::AdjacencyList<std::int32_t>>,
           std::shared_ptr<graph::AdjacencyList<std::int32_t>>,
           std::shared_ptr<common::IndexMap>>
compute_entities_by_key_matching(
    MPI_Comm comm, const graph::AdjacencyList<std::int32_t>& cells,
    const std::shared_ptr<const common::IndexMap>& vertex_index_map,
    const std::shared_ptr<const common::IndexMap>& cell_index_map,
    mesh::CellType cell_type, int dim)
{
  if (dim == 0)
  {
    throw std::runtime_error(
        "Cannot create vertices for topology. Should already exist.");
  }

  // Start timer
  common::Timer timer("Compute entities of dim = " + std::to_string(dim));

  // Initialize local array of entities
  const std::int8_t num_entities_per_cell
      = mesh::cell_num_entities(cell_type, dim);
  const int num_vertices_per_entity
      = mesh::num_cell_vertices(mesh::cell_entity_type(cell_type, dim));

  // Create map from cell vertices to entity vertices
  Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> e_vertices
      = mesh::get_entity_vertices(cell_type, dim);

  const int num_cells = cells.num_nodes();

  // List of vertices for each entity in each cell
  Eigen::Array<std::int32_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      entity_list(num_cells * num_entities_per_cell, num_vertices_per_entity);
  int k = 0;
  for (int c = 0; c < num_cells; ++c)
  {
    // Get vertices from cell
    auto vertices = cells.links(c);

    // Iterate over entities of cell
    for (int i = 0; i < num_entities_per_cell; ++i)
    {
      // Get entity vertices
      for (int j = 0; j < num_vertices_per_entity; ++j)
        entity_list(k, j) = vertices[e_vertices(i, j)];

      ++k;
    }
  }
  assert(k == entity_list.rows());

  std::vector<std::int32_t> entity_index(entity_list.rows());
  std::int32_t entity_count = 0;

  // Copy list and sort vertices of each entity into order
  Eigen::Array<std::int32_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      entity_list_sorted = entity_list;
  for (int i = 0; i < entity_list_sorted.rows(); ++i)
  {
    std::sort(entity_list_sorted.row(i).data(),
              entity_list_sorted.row(i).data() + num_vertices_per_entity);
  }

  // Sort the list and label uniquely
  const std::vector sort_order = sort_by_perm<std::int32_t>(entity_list_sorted);
  std::int32_t last = sort_order[0];
  entity_index[last] = 0;
  for (std::size_t i = 1; i < sort_order.size(); ++i)
  {
    std::int32_t j = sort_order[i];
    if ((entity_list_sorted.row(j) != entity_list_sorted.row(last)).any())
      ++entity_count;
    entity_index[j] = entity_count;
    last = j;
  }
  ++entity_count;

  // Communicate with other processes to find out which entities are
  // ghosted and shared. Remap the numbering so that ghosts are at the
  // end.
  const auto [local_index, index_map] = get_local_indexing(
      comm, cell_index_map, vertex_index_map, entity_list, entity_index);

  Eigen::Array<std::int32_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      connectivity_ce(num_cells, num_entities_per_cell);
  std::copy(local_index.begin(), local_index.end(), connectivity_ce.data());

  // Cell-entity connectivity
  auto ce
      = std::make_shared<graph::AdjacencyList<std::int32_t>>(connectivity_ce);

  // Entity-vertex connectivity
  Eigen::Array<std::int32_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      connectivity_ev(entity_count, num_vertices_per_entity);
  for (int i = 0; i < entity_list.rows(); ++i)
    connectivity_ev.row(local_index[i]) = entity_list.row(i);

  auto ev
      = std::make_shared<graph::AdjacencyList<std::int32_t>>(connectivity_ev);

  return {ce, ev, index_map};
}
//-----------------------------------------------------------------------------

/// Compute connectivity from entities of dimension d0 to entities of
/// dimension d1 using the transpose connectivity (d1 -> d0)
/// @param[in] c_d1_d0 The connectivity from entities of dimension d1 to
///   entities of dimension d0
/// @param[in] num_entities_d0 The number of entities of dimension d0
/// @return The connectivity from entities of dimension d0 to entities
///   of dimension d1
graph::AdjacencyList<std::int32_t>
compute_from_transpose(const graph::AdjacencyList<std::int32_t>& c_d1_d0,
                       const int num_entities_d0, int d0, int d1)
{
  LOG(INFO) << "Computing mesh connectivity " << d0 << " - " << d1
            << "from transpose.";

  // Compute number of connections for each e0
  std::vector<std::int32_t> num_connections(num_entities_d0, 0);
  for (int e1 = 0; e1 < c_d1_d0.num_nodes(); ++e1)
  {
    auto e = c_d1_d0.links(e1);
    for (int i = 0; i < e.rows(); ++i)
      num_connections[e[i]]++;
  }

  // Compute offsets
  Eigen::Array<std::int32_t, Eigen::Dynamic, 1> offsets
      = Eigen::Array<std::int32_t, Eigen::Dynamic, 1>::Zero(
          num_connections.size() + 1);
  std::partial_sum(num_connections.begin(), num_connections.end(),
                   offsets.data() + 1);

  std::vector<std::int32_t> counter(num_connections.size(), 0);
  Eigen::Array<std::int32_t, Eigen::Dynamic, 1> connections(
      offsets[offsets.rows() - 1]);
  for (int e1 = 0; e1 < c_d1_d0.num_nodes(); ++e1)
  {
    auto e = c_d1_d0.links(e1);
    for (int e0 = 0; e0 < e.rows(); ++e0)
      connections[offsets[e[e0]] + counter[e[e0]]++] = e1;
  }

  return graph::AdjacencyList<std::int32_t>(std::move(connections),
                                            std::move(offsets));
}
//-----------------------------------------------------------------------------

/// Compute the d0 -> d1 connectivity, where d0 > d1
/// @param[in] c_d0_0 The d0 -> 0 (entity (d0) to vertex) connectivity
/// @param[in] c_d0_0 The d1 -> 0 (entity (d1) to vertex) connectivity
/// @param[in] cell_type_d0 The cell type for entities of dimension d0
/// @param[in] d0 Topological dimension
/// @param[in] d1 Topological dimension
/// @return The d0 -> d1 connectivity
graph::AdjacencyList<std::int32_t>
compute_from_map(const graph::AdjacencyList<std::int32_t>& c_d0_0,
                 const graph::AdjacencyList<std::int32_t>& c_d1_0,
                 CellType cell_type_d0, int d0, int d1)
{
  assert(d1 > 0);
  assert(d0 > d1);

  // Make a map from the sorted d1 entity vertices to the d1 entity
  // index
  boost::unordered_map<std::vector<std::int32_t>, std::int32_t> entity_to_index;
  entity_to_index.reserve(c_d1_0.num_nodes());

  const std::size_t num_verts_d1
      = mesh::num_cell_vertices(mesh::cell_entity_type(cell_type_d0, d1));

  std::vector<std::int32_t> key(num_verts_d1);
  for (int e = 0; e < c_d1_0.num_nodes(); ++e)
  {
    const std::int32_t* v = c_d1_0.links_ptr(e);
    std::partial_sort_copy(v, v + num_verts_d1, key.begin(), key.end());
    entity_to_index.insert({key, e});
  }

  Eigen::Array<std::int32_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      connections(c_d0_0.num_nodes(),
                  mesh::cell_num_entities(cell_type_d0, d1));

  // Search for d1 entities of d0 in map, and recover index
  std::vector<std::int32_t> entities;
  const Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      e_vertices_ref = mesh::get_entity_vertices(cell_type_d0, d1);
  Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> keys
      = e_vertices_ref;
  for (int e = 0; e < c_d0_0.num_nodes(); ++e)
  {
    entities.clear();
    auto e0 = c_d0_0.links(e);
    for (Eigen::Index i = 0; i < e_vertices_ref.rows(); ++i)
      for (Eigen::Index j = 0; j < e_vertices_ref.cols(); ++j)
        keys(i, j) = e0[e_vertices_ref(i, j)];
    for (Eigen::Index i = 0; i < keys.rows(); ++i)
    {
      std::partial_sort_copy(keys.row(i).data(),
                             keys.row(i).data() + keys.row(i).cols(),
                             key.begin(), key.end());
      const auto it = entity_to_index.find(key);
      assert(it != entity_to_index.end());
      entities.push_back(it->second);
    }
    for (std::size_t k = 0; k < entities.size(); ++k)
      connections(e, k) = entities[k];
  }

  return graph::AdjacencyList<std::int32_t>(connections);
}
//-----------------------------------------------------------------------------
} // namespace

//-----------------------------------------------------------------------------
std::tuple<std::shared_ptr<graph::AdjacencyList<std::int32_t>>,
           std::shared_ptr<graph::AdjacencyList<std::int32_t>>,
           std::shared_ptr<common::IndexMap>>
TopologyComputation::compute_entities(MPI_Comm comm, const Topology& topology,
                                      int dim)
{
  LOG(INFO) << "Computing mesh entities of dimension " << dim;
  const int tdim = topology.dim();

  // Vertices must always exist
  if (dim == 0)
    return {nullptr, nullptr, nullptr};

  if (topology.connectivity(dim, 0))
  {
    // Make sure we really have the connectivity
    if (!topology.connectivity(tdim, dim))
    {
      throw std::runtime_error(
          "Cannot compute topological entities. Entities of topological "
          "dimension "
          + std::to_string(dim)
          + " exist but cell-dim connectivity is missing.");
    }
    return {nullptr, nullptr, nullptr};
  }

  auto cells = topology.connectivity(tdim, 0);
  if (!cells)
    throw std::runtime_error("Cell connectivity missing.");

  auto vertex_map = topology.index_map(0);
  assert(vertex_map);
  auto cell_map = topology.index_map(tdim);
  assert(cell_map);
  std::tuple<std::shared_ptr<graph::AdjacencyList<std::int32_t>>,
             std::shared_ptr<graph::AdjacencyList<std::int32_t>>,
             std::shared_ptr<common::IndexMap>>
      data = compute_entities_by_key_matching(
          comm, *cells, vertex_map, cell_map, topology.cell_type(), dim);

  return data;
}
//-----------------------------------------------------------------------------
std::array<std::shared_ptr<graph::AdjacencyList<std::int32_t>>, 2>
TopologyComputation::compute_connectivity(const Topology& topology, int d0,
                                          int d1)
{
  LOG(INFO) << "Requesting connectivity " << d0 << " - " << d1;

  // Return if connectivity has already been computed
  if (topology.connectivity(d0, d1))
    return {nullptr, nullptr};

  // Get entities exist
  std::shared_ptr<const graph::AdjacencyList<std::int32_t>> c_d0_0
      = topology.connectivity(d0, 0);
  if (d0 > 0 and !topology.connectivity(d0, 0))
  {
    throw std::runtime_error("Missing entities of dimension "
                             + std::to_string(d0) + ".");
  }

  std::shared_ptr<const graph::AdjacencyList<std::int32_t>> c_d1_0
      = topology.connectivity(d1, 0);
  if (d1 > 0 and !topology.connectivity(d1, 0))
  {
    throw std::runtime_error("Missing entities of dimension "
                             + std::to_string(d1) + ".");
  }

  // Start timer
  common::Timer timer("Compute connectivity " + std::to_string(d0) + "-"
                      + std::to_string(d1));

  // Decide how to compute the connectivity
  if (d0 == d1)
  {
    return {std::make_shared<graph::AdjacencyList<std::int32_t>>(
                c_d0_0->num_nodes()),
            nullptr};
  }
  else if (d0 < d1)
  {
    // Compute connectivity d1 - d0 (if needed), and take transpose
    if (!topology.connectivity(d1, d0))
    {
      auto c_d1_d0 = std::make_shared<graph::AdjacencyList<std::int32_t>>(
          compute_from_map(*c_d1_0, *c_d0_0,
                           mesh::cell_entity_type(topology.cell_type(), d1), d1,
                           d0));
      auto c_d0_d1 = std::make_shared<graph::AdjacencyList<std::int32_t>>(
          compute_from_transpose(*c_d1_d0, c_d0_0->num_nodes(), d0, d1));
      return {c_d0_d1, c_d1_d0};
    }
    else
    {
      assert(c_d0_0);
      assert(topology.connectivity(d1, d0));
      auto c_d0_d1 = std::make_shared<graph::AdjacencyList<std::int32_t>>(
          compute_from_transpose(*topology.connectivity(d1, d0),
                                 c_d0_0->num_nodes(), d0, d1));
      return {c_d0_d1, nullptr};
    }
  }
  else if (d0 > d1)
  {
    // Compute by mapping vertices from a lower dimension entity to
    // those of a higher dimension entity
    auto c_d0_d1
        = std::make_shared<graph::AdjacencyList<std::int32_t>>(compute_from_map(
            *c_d0_0, *c_d1_0, mesh::cell_entity_type(topology.cell_type(), d0),
            d0, d1));
    return {c_d0_d1, nullptr};
  }
  else
    throw std::runtime_error("Entity dimension error when computing topology.");
}
//--------------------------------------------------------------------------
