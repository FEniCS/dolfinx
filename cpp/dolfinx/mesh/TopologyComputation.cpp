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
#include <string>
#include <tuple>
#include <utility>
#include <vector>

using namespace dolfinx;
using namespace dolfinx::mesh;

namespace
{
//-----------------------------------------------------------------------------

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

/// Get the shared entities, a map from local index to the set of
/// sharing processes.
///
/// @param[in] neighbour_comm The MPI neighborhood communicator for all
///   processes sharing vertices
/// @param[in] send_entities Lists of entities (as vertex indices) to
///   send to other processes
/// @param[in] send_index Local index of sent entities (one for each in
///   send_entities)
/// @param[in] num_vertices Number of vertices per entity
/// @return Tuple of (shared_entities and recv_index) where recv_index
///   is the matching received index to send_index, if the entities
///   exist, -1 otherwise.
std::tuple<std::map<std::int32_t, std::set<std::int32_t>>,
           std::vector<std::vector<std::int32_t>>>
get_shared_entities(MPI_Comm neighbour_comm,
                    const std::vector<std::vector<std::int64_t>>& send_entities,
                    const std::vector<std::vector<std::int32_t>>& send_index,
                    int num_vertices)
{
  const std::vector<int> neighbours = dolfinx::MPI::neighbors(neighbour_comm);
  const int neighbour_size = neighbours.size();

  // Items to return
  std::map<std::int32_t, std::set<std::int32_t>> shared_entities;
  std::vector<std::vector<std::int32_t>> recv_index(neighbour_size);

  // Prepare data for neighbour all to all
  std::vector<std::int64_t> send_entities_data;
  std::vector<std::int64_t> recv_entities_data;
  std::vector<int> send_offsets = {0};
  std::vector<int> recv_offsets;
  for (std::size_t i = 0; i < send_entities.size(); ++i)
  {
    send_entities_data.insert(send_entities_data.end(),
                              send_entities[i].begin(), send_entities[i].end());
    send_offsets.push_back(send_entities_data.size());
  }

  dolfinx::MPI::neighbor_all_to_all(neighbour_comm, send_offsets,
                                    send_entities_data, recv_offsets,
                                    recv_entities_data);

  // Compare received with sent for each process
  for (int np = 0; np < neighbour_size; ++np)
  {
    Eigen::Map<const Eigen::Array<std::int64_t, Eigen::Dynamic, Eigen::Dynamic,
                                  Eigen::RowMajor>>
        send_array(send_entities_data.data() + send_offsets[np],
                   (send_offsets[np + 1] - send_offsets[np]) / num_vertices,
                   num_vertices);

    Eigen::Map<const Eigen::Array<std::int64_t, Eigen::Dynamic, Eigen::Dynamic,
                                  Eigen::RowMajor>>
        recv_array(recv_entities_data.data() + recv_offsets[np],
                   (recv_offsets[np + 1] - recv_offsets[np]) / num_vertices,
                   num_vertices);

    recv_index[np].resize(recv_array.rows(), -1);

    // Compare with sent values
    for (int i = 0; i < send_array.rows(); ++i)
    {
      for (int j = 0; j < recv_array.rows(); ++j)
      {
        if ((recv_array.row(j) == send_array.row(i)).all())
        {
          // This entity was sent, and the same was received back,
          // confirming sharing
          const int p = neighbours[np];
          shared_entities[send_index[np][i]].insert(p);
          recv_index[np][j] = send_index[np][i];
          break;
        }
      }
    }
  }

  return {shared_entities, recv_index};
}
//-----------------------------------------------------------------------------

/// Communicate with sharing processes to find out which entities are
/// ghosts and return a mapping vector to move these local indices to
/// the end of the local range. Also returns the index map, and shared
/// entities, i.e. the set of all processes which share each shared
/// entity.
/// @param[in] comm MPI Communicator
/// @param[in] IndexMap for vertices
/// @param[in] global_vertex_indices Global indices of vertices
/// @param[in] entity_list List of entities as 2D array, each entity
///   represented by its local vertex indices
/// @param[in] entity_index Initial numbering for each row in
///   entity_list
/// @param[in] entity_count Number of unique entities
/// @returns Tuple of (local_indices, index map, shared entities)
std::tuple<std::vector<int>, std::shared_ptr<common::IndexMap>>
get_local_indexing(
    MPI_Comm comm,
    const std::shared_ptr<const common::IndexMap> vertex_indexmap,
    const std::vector<std::int64_t>& global_vertex_indices,
    const Eigen::Ref<const Eigen::Array<std::int32_t, Eigen::Dynamic,
                                        Eigen::Dynamic, Eigen::RowMajor>>&
        entity_list,
    const std::vector<std::int32_t>& entity_index, int entity_count)
{
  const int num_vertices = entity_list.cols();

  std::map<std::int32_t, std::set<std::int32_t>> shared_vertices
      = vertex_indexmap->compute_shared_indices();

  // Get a single row in entity list for each entity
  std::vector<std::int32_t> unique_row(entity_count);
  for (int i = 0; i < entity_list.rows(); ++i)
    unique_row[entity_index[i]] = i;

  // Create an expanded neighbour_comm from shared_vertices
  std::set<std::int32_t> neighbour_set;
  for (auto q : shared_vertices)
    neighbour_set.insert(q.second.begin(), q.second.end());
  std::vector<std::int32_t> neighbours(neighbour_set.begin(),
                                       neighbour_set.end());
  MPI_Comm neighbour_comm;
  MPI_Dist_graph_create_adjacent(comm, neighbours.size(), neighbours.data(),
                                 MPI_UNWEIGHTED, neighbours.size(),
                                 neighbours.data(), MPI_UNWEIGHTED,
                                 MPI_INFO_NULL, false, &neighbour_comm);

  const int neighbour_size = neighbours.size();
  std::map<int, int> proc_to_neighbour;
  for (int i = 0; i < neighbour_size; ++i)
    proc_to_neighbour.insert({neighbours[i], i});

  std::vector<std::vector<std::int64_t>> send_entities(neighbour_size);
  std::vector<std::vector<std::int32_t>> send_index(neighbour_size);

  // Get all "possibly shared" entities, based on vertex sharing. Send to
  // other processes, and see if we get the same back.

  // Set of sharing procs for each entity, counting vertex hits
  std::map<int, int> procs;
  for (int i : unique_row)
  {
    procs.clear();
    for (int j = 0; j < num_vertices; ++j)
    {
      const int v = entity_list(i, j);
      const auto it = shared_vertices.find(v);
      if (it != shared_vertices.end())
      {
        for (std::int32_t p : it->second)
          ++procs[p];
      }
    }
    for (const auto& q : procs)
    {
      if (q.second == num_vertices)
      {
        const int p = q.first;
        auto it = proc_to_neighbour.find(p);
        assert(it != proc_to_neighbour.end());
        const int np = it->second;

        // Entity entity_index[i] may be shared with process p
        for (int j = 0; j < num_vertices; ++j)
        {
          std::int64_t vglobal = global_vertex_indices[entity_list(i, j)];
          send_entities[np].push_back(vglobal);
        }
        send_index[np].push_back(entity_index[i]);
        std::sort(send_entities[np].end() - num_vertices,
                  send_entities[np].end());
      }
    }
  }

  // Get shared entities of this dimension, and also match up an index
  // for the received entities (from other processes) with the indices
  // of the sent entities (to other processes)
  const auto [shared_entities, recv_index] = get_shared_entities(
      neighbour_comm, send_entities, send_index, num_vertices);

  int mpi_rank = dolfinx::MPI::rank(comm);

  // Determine ownership
  std::vector<std::int32_t> local_index(entity_count, -1);
  std::int32_t c = 0;
  // Index non-ghost entities
  for (int i = 0; i < entity_count; ++i)
  {
    const auto it = shared_entities.find(i);
    if (it == shared_entities.end() or *(it->second.begin()) > mpi_rank)
    {
      // Owned index
      local_index[i] = c;
      ++c;
    }
  }
  const std::int32_t num_local = c;
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

  Eigen::Array<std::int64_t, Eigen::Dynamic, 1> ghost_indices(entity_count
                                                              - num_local);
  // Communicate global indices to other processes
  {
    const std::int64_t local_offset
        = dolfinx::MPI::global_offset(comm, num_local, true);

    std::vector<std::int64_t> send_global_index_data;
    std::vector<int> send_global_index_offsets = {0};
    std::vector<std::int64_t> recv_global_index_data;
    std::vector<int> recv_global_index_offsets;

    // Send global indices for same entities that we sent before
    // This uses the same pattern as before, so we can match up
    // the received data to the indices in recv_index
    for (int np = 0; np < neighbour_size; ++np)
    {
      for (std::int32_t index : send_index[np])
      {
        std::int64_t gi = (local_index[index] < num_local)
                              ? (local_offset + local_index[index])
                              : -1;

        send_global_index_data.push_back(gi);
      }

      send_global_index_offsets.push_back(send_global_index_data.size());
    }

    dolfinx::MPI::neighbor_all_to_all(
        neighbour_comm, send_global_index_offsets, send_global_index_data,
        recv_global_index_offsets, recv_global_index_data);

    // Map back received indices
    for (int np = 0; np < neighbour_size; ++np)
    {
      for (int j = 0; j < (recv_global_index_offsets[np + 1]
                           - recv_global_index_offsets[np]);
           ++j)
      {
        const std::int64_t gi
            = recv_global_index_data[j + recv_global_index_offsets[np]];
        if (gi != -1 and recv_index[np][j] != -1)
        {
          const std::int32_t idx = recv_index[np][j];
          assert(local_index[idx] >= num_local);
          ghost_indices[local_index[idx] - num_local] = gi;
        }
      }
    }
  }

  std::shared_ptr<common::IndexMap> index_map
      = std::make_shared<common::IndexMap>(comm, num_local, ghost_indices, 1);

  return {std::move(local_index), index_map};
}
//-----------------------------------------------------------------------------

/// Compute entities of dimension d
/// @param[in] comm MPI communicator (TODO: full or neighbour hood?)
/// @param[in] cells Adjacency list for cell-vertex connectivity
/// @param[in] shared_vertices TODO
/// @param[in] global_vertex_indices TODO: user global?
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
    const std::shared_ptr<const common::IndexMap> vertex_index_map,
    const std::vector<std::int64_t>& global_vertex_indices,
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

  // Sort the list and label (first pass)
  std::vector<std::int32_t> sort_order
      = sort_by_perm<std::int32_t>(entity_list_sorted);
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
  auto [local_index, index_map]
      = get_local_indexing(comm, vertex_index_map, global_vertex_indices,
                           entity_list, entity_index, entity_count);

  // Map from initial numbering to local indices
  for (std::int32_t& q : entity_index)
    q = local_index[q];

  Eigen::Array<std::int32_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      connectivity_ce(num_cells, num_entities_per_cell);
  std::copy(entity_index.begin(), entity_index.end(), connectivity_ce.data());

  // Cell-entity connectivity
  auto ce
      = std::make_shared<graph::AdjacencyList<std::int32_t>>(connectivity_ce);

  // Entity-vertex connectivity
  Eigen::Array<std::int32_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      connectivity_ev(entity_count, num_vertices_per_entity);
  for (int i = 0; i < entity_list.rows(); ++i)
    connectivity_ev.row(entity_index[i]) = entity_list.row(i);

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
  std::vector<std::int32_t> offsets(num_connections.size() + 1, 0);
  std::partial_sum(num_connections.begin(), num_connections.end(),
                   offsets.begin() + 1);

  std::vector<std::int32_t> counter(num_connections.size(), 0);
  std::vector<std::int32_t> connections(offsets.back());
  for (int e1 = 0; e1 < c_d1_d0.num_nodes(); ++e1)
  {
    auto e = c_d1_d0.links(e1);
    for (int e0 = 0; e0 < e.rows(); ++e0)
      connections[offsets[e[e0]] + counter[e[e0]]++] = e1;
  }

  return graph::AdjacencyList<std::int32_t>(connections, offsets);
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
} // namespace

//-----------------------------------------------------------------------------
std::tuple<std::shared_ptr<graph::AdjacencyList<std::int32_t>>,
           std::shared_ptr<graph::AdjacencyList<std::int32_t>>,
           std::shared_ptr<common::IndexMap>>
TopologyComputation::compute_entities(MPI_Comm comm, const Topology& topology,
                                      int dim)
{
  LOG(INFO) << "Computing mesh entities of dimension " << dim;

  // Vertices must always exist
  if (dim == 0)
    return {nullptr, nullptr, nullptr};

  if (topology.connectivity(dim, 0))
  {
    // Make sure we really have the connectivity
    if (!topology.connectivity(topology.dim(), dim))
    {
      throw std::runtime_error(
          "Cannot compute topological entities. Entities of topological "
          "dimension "
          + std::to_string(dim)
          + " exist but cell-dim connectivity is missing.");
    }
    return {nullptr, nullptr, nullptr};
  }

  const int tdim = topology.dim();
  auto cells = topology.connectivity(tdim, 0);
  if (!cells)
    throw std::runtime_error("Cell connectivity missing.");

  auto map = topology.index_map(0);
  assert(map);
  const std::vector<std::int64_t> global_vertices = map->global_indices(false);
  std::tuple<std::shared_ptr<graph::AdjacencyList<std::int32_t>>,
             std::shared_ptr<graph::AdjacencyList<std::int32_t>>,
             std::shared_ptr<common::IndexMap>>
      data
      = compute_entities_by_key_matching(comm, *cells, topology.index_map(0),
                                         global_vertices, topology.cell_type(),
                                         dim);

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
    std::runtime_error("Missing entities of dimension " + std::to_string(d0)
                       + ".");
  }

  std::shared_ptr<const graph::AdjacencyList<std::int32_t>> c_d1_0
      = topology.connectivity(d1, 0);
  if (d1 > 0 and !topology.connectivity(d1, 0))
  {
    std::runtime_error("Missing entities of dimension " + std::to_string(d1)
                       + ".");
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
