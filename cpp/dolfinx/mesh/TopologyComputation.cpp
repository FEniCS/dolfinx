// Copyright (C) 2006-2017 Anders Logg and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "TopologyComputation.h"
#include "Mesh.h"
#include "MeshEntity.h"
#include "MeshIterator.h"
#include "Topology.h"
#include "cell_types.h"
#include <Eigen/Dense>
#include <algorithm>
#include <boost/unordered_map.hpp>
#include <cstdint>
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
// Takes an Eigen::Array and obtains the sort permutation to reorder the
// rows in ascending order. Each row must be sorted beforehand.
template <typename T>
std::vector<int> sort_by_perm(
    const Eigen::Ref<const Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic,
                                        Eigen::RowMajor>>& arr_data)
{
  // Sort an Eigen::Array by creating a permutation vector
  std::vector<int> index(arr_data.rows());
  std::iota(index.begin(), index.end(), 0);
  const int cols = arr_data.cols();

  // Lambda with capture for sort comparison
  const auto cmp = [&arr_data, &cols](int a, int b) {
    const T* row_a = arr_data.row(a).data();
    const T* row_b = arr_data.row(b).data();
    return std::lexicographical_compare(row_a, row_a + cols, row_b,
                                        row_b + cols);
  };

  std::sort(index.begin(), index.end(), cmp);
  return index;
}
//-----------------------------------------------------------------------------
// Communicate with sharing processes to find out which entities are ghost
// and return a mapping vector to move them to the end of the local range.
std::vector<int> get_ghost_mapping(
    MPI_Comm comm, const Topology& topology,
    const Eigen::Ref<const Eigen::Array<std::int32_t, Eigen::Dynamic,
                                        Eigen::Dynamic, Eigen::RowMajor>>&
        entity_list,
    const std::vector<std::int32_t>& entity_index, int entity_count)
{
  const int num_vertices = entity_list.cols();

  // Get a single row in entity list for each entity
  std::vector<std::int32_t> unique_row(entity_count);
  for (int i = 0; i < entity_list.rows(); ++i)
    unique_row[entity_index[i]] = i;

  // Create an expanded neighbour_comm from shared_vertices
  const std::map<std::int32_t, std::set<std::int32_t>>& shared_vertices
      = topology.shared_entities(0);
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
  std::vector<std::vector<std::int32_t>> recv_index(neighbour_size);

  // Get all "possibly shared" entities, based on vertex sharing
  // Send to other processes, and see if we get the same back
  const std::vector<std::int64_t>& global_vertex_indices
      = topology.global_indices(0);

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

  // prepare data for neighbour all to all
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

  std::map<std::int32_t, std::set<std::int32_t>> shared_entities;
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
          // This entity was sent, and the same was received back, confirming
          // sharing.
          const int p = neighbours[np];
          shared_entities[send_index[np][i]].insert(p);
          recv_index[np][j] = send_index[np][i];
          break;
        }
      }
    }
  }

  int mpi_rank = dolfinx::MPI::rank(comm);

  // Put ghosts at end of range by remapping
  std::vector<std::int32_t> mapping(entity_count, -1);
  std::int32_t c = 0;
  // Index non-ghost entities
  for (int i = 0; i < entity_count; ++i)
  {
    const auto it = shared_entities.find(i);
    if (it == shared_entities.end() or *(it->second.begin()) > mpi_rank)
    {
      // Owned index
      mapping[i] = c;
      ++c;
    }
  }
  const std::int32_t num_local = c;

  // Create global indices
  const std::int64_t local_offset
      = dolfinx::MPI::global_offset(comm, num_local, true);
  std::vector<std::int64_t> global_indexing(entity_count, -1);
  for (int i = 0; i < entity_count; ++i)
  {
    if (mapping[i] != -1)
      global_indexing[i] = local_offset + mapping[i];
  }

  std::vector<std::int64_t> send_global_index_data;
  std::vector<int> send_global_index_offsets = {0};
  std::vector<std::int64_t> recv_global_index_data;
  std::vector<int> recv_global_index_offsets;

  // Send global indices for same entities that we sent before
  for (int np = 0; np < neighbour_size; ++np)
  {
    for (std::int32_t index : send_index[np])
      send_global_index_data.push_back(global_indexing[index]);
    send_global_index_offsets.push_back(send_global_index_data.size());
  }

  dolfinx::MPI::neighbor_all_to_all(
      neighbour_comm, send_global_index_offsets, send_global_index_data,
      recv_global_index_offsets, recv_global_index_data);

  // Map back received indices - need to check
  for (int np = 0; np < neighbour_size; ++np)
  {
    for (int j = 0; j < (recv_global_index_offsets[np + 1]
                         - recv_global_index_offsets[np]);
         ++j)
    {
      const std::int64_t gi
          = recv_global_index_data[j + recv_global_index_offsets[np]];
      if (gi != -1 and recv_index[np][j] != -1)
        global_indexing[recv_index[np][j]] = gi;
    }
  }

  // std::stringstream s;
  // s << mpi_rank << "] gi = [";
  // for (auto q : global_indexing)
  //   s << q << " ";
  // s << "]\n";
  // std::cout << s.str();

  // Now index the ghosts locally
  for (int i = 0; i < entity_count; ++i)
    if (mapping[i] == -1)
    {
      mapping[i] = c;
      ++c;
    }
  assert(c == entity_count);

  // FIXME - Remap shared entities to new indices

  // FIXME - also return shared_entities, and global numbering
  return mapping;
}
//-----------------------------------------------------------------------------
std::tuple<std::shared_ptr<graph::AdjacencyList<std::int32_t>>,
           std::shared_ptr<graph::AdjacencyList<std::int32_t>>, std::int32_t>
compute_entities_by_key_matching(MPI_Comm comm, const Topology& topology,
                                 mesh::CellType cell_type, int dim)
{
  if (dim == 0)
  {
    throw std::runtime_error(
        "Cannot create vertices for topology. Should already exist.");
  }

  // Get mesh topology and connectivity
  // const Topology& topology = mesh.topology();
  const int tdim = topology.dim();

  // Check if entities have already been computed
  if (topology.connectivity(dim, 0))
  {
    // Check that we have cell-entity connectivity
    if (!topology.connectivity(tdim, dim))
      throw std::runtime_error("Missing cell-entity connectivity");

    return {nullptr, nullptr, -1};
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

  auto cells = topology.connectivity(tdim, 0);
  assert(cells);
  const int num_cells = cells->num_nodes();

  // List of vertices for each entity in each cell
  Eigen::Array<std::int32_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      entity_list(num_cells * num_entities_per_cell, num_vertices_per_entity);
  int k = 0;
  for (int c = 0; c < num_cells; ++c)
  {
    // Get vertices from cell
    auto vertices = cells->edges(c);

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

  // Communicate with other processes to find out which entities are ghosted
  // and shared. Remap the numbering so that ghosts are at the end.
  std::vector<int> mapping = get_ghost_mapping(comm, topology, entity_list,
                                               entity_index, entity_count);

  // Do the actual remap
  for (std::int32_t& q : entity_index)
    q = mapping[q];

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

  return {ce, ev, entity_count};
}
//-----------------------------------------------------------------------------
// Compute connectivity from transpose
graph::AdjacencyList<std::int32_t> compute_from_transpose(const Mesh& mesh,
                                                          int d0, int d1)
{
  // The transpose is computed in three steps:
  //
  //   1. Iterate over entities of dimension d1 and count the number
  //      of connections for each entity of dimension d0
  //
  //   2. Allocate memory / prepare data structures
  //
  //   3. Iterate again over entities of dimension d1 and add connections
  //      for each entity of dimension d0

  LOG(INFO) << "Computing mesh connectivity " << d0 << " - " << d1
            << "from transpose.";

  // Get mesh topology and connectivity
  const Topology& topology = mesh.topology();

  // Need connectivity d1 - d0
  if (!topology.connectivity(d1, d0))
    throw std::runtime_error("Missing required connectivity d1-d0.");

  // Compute number of connections for each e0
  auto map_d0 = topology.index_map(d0);
  assert(map_d0);
  const int size_d0 = map_d0->size_local() + map_d0->num_ghosts();
  std::vector<std::int32_t> num_connections(size_d0, 0);
  for (auto& e1 : MeshRange(mesh, d1, MeshRangeType::ALL))
    for (auto& e0 : EntityRange(e1, d0))
      num_connections[e0.index()]++;

  // Compute offsets
  std::vector<std::int32_t> offsets(num_connections.size() + 1, 0);
  std::partial_sum(num_connections.begin(), num_connections.end(),
                   offsets.begin() + 1);

  std::vector<std::int32_t> counter(num_connections.size(), 0);
  std::vector<std::int32_t> connections(offsets.back());
  for (auto& e1 : MeshRange(mesh, d1, MeshRangeType::ALL))
    for (auto& e0 : EntityRange(e1, d0))
      connections[offsets[e0.index()] + counter[e0.index()]++] = e1.index();

  return graph::AdjacencyList<std::int32_t>(connections, offsets);
}
//-----------------------------------------------------------------------------
// Direct lookup of entity from vertices in a map
graph::AdjacencyList<std::int32_t> compute_from_map(const Topology& topology,
                                                    CellType cell_type_d0,
                                                    int d0, int d1)
{
  assert(d1 > 0);
  assert(d0 > d1);

  auto c_d1_0 = topology.connectivity(d1, 0);
  assert(c_d1_0);

  // Make a map from the sorted d1 entity vertices to the d1 entity index
  boost::unordered_map<std::vector<std::int32_t>, std::int32_t> entity_to_index;
  entity_to_index.reserve(c_d1_0->num_nodes());

  const std::size_t num_verts_d1
      = mesh::num_cell_vertices(mesh::cell_entity_type(cell_type_d0, d1));

  std::vector<std::int32_t> key(num_verts_d1);
  for (int e = 0; e < c_d1_0->num_nodes(); ++e)
  {
    const std::int32_t* v = c_d1_0->edges_ptr(e);
    std::partial_sort_copy(v, v + num_verts_d1, key.begin(), key.end());
    entity_to_index.insert({key, e});
  }

  auto c_d0_0 = topology.connectivity(d0, 0);
  assert(c_d0_0);
  Eigen::Array<std::int32_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      connections(c_d0_0->num_nodes(),
                  mesh::cell_num_entities(cell_type_d0, d1));

  // Search for d1 entities of d0 in map, and recover index
  std::vector<std::int32_t> entities;
  const Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      e_vertices_ref = mesh::get_entity_vertices(cell_type_d0, d1);
  Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> keys
      = e_vertices_ref;
  for (int e = 0; e < c_d0_0->num_nodes(); ++e)
  {
    entities.clear();
    auto e0 = c_d0_0->edges(e);
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
           std::shared_ptr<graph::AdjacencyList<std::int32_t>>, std::int32_t>
TopologyComputation::compute_entities(MPI_Comm comm, const Topology& topology,
                                      mesh::CellType cell_type, int dim)
{
  LOG(INFO) << "Computing mesh entities of dimension " << dim;

  // Vertices must always exist
  if (dim == 0)
    return {nullptr, nullptr, -1};

  if (topology.connectivity(dim, 0))
  {
    // Make sure we really have the connectivity
    if (!topology.connectivity(topology.dim(), dim))
    {
      throw std::runtime_error(
          "Cannot compute topological entities. Entities of topological "
          "dimension "
          + std::to_string(dim) + " exist but connectivity is missing.");
    }
    return {nullptr, nullptr, -1};
  }

  std::tuple<std::shared_ptr<graph::AdjacencyList<std::int32_t>>,
             std::shared_ptr<graph::AdjacencyList<std::int32_t>>, std::int32_t>
      data = compute_entities_by_key_matching(comm, topology, cell_type, dim);

  return data;
}
//-----------------------------------------------------------------------------
void TopologyComputation::compute_connectivity(Mesh& mesh, int d0, int d1)
{
  // This is where all the logic takes place to find a strategy for
  // the connectivity computation. For any given pair (d0, d1), the
  // connectivity is computed by suitably combining the following
  // basic building blocks:
  //
  //   1. compute_entities():     d  - 0  from dim - 0
  //   2. compute_transpose():    d0 - d1 from d1 - d0
  //   4. compute_from_map():     d0 - d1 from d1 - 0 and d0 - 0
  // Each of these functions assume a set of preconditions that we
  // need to satisfy.

  LOG(INFO) << "Requesting connectivity " << d0 << " - " << d1;

  // Get mesh topology and connectivity
  Topology& topology = mesh.topology();

  // Return if connectivity has already been computed
  if (topology.connectivity(d0, d1))
    return;

  // Make sure entities exist
  if (d0 > 0)
    assert(topology.connectivity(d0, 0));
  if (d1 > 0)
    assert(topology.connectivity(d1, 0));

  // Start timer
  common::Timer timer("Compute connectivity " + std::to_string(d0) + "-"
                      + std::to_string(d1));

  // Decide how to compute the connectivity
  if (d0 == d1)
  {
    // For d0==d1, use identity connectivity
    auto map_d0 = topology.index_map(d0);
    assert(map_d0);
    const int size_d0 = map_d0->size_local() + map_d0->num_ghosts();
    std::vector<std::vector<std::size_t>> connectivity_dd(
        size_d0, std::vector<std::size_t>(1));
    for (auto& e : MeshRange(mesh, d0, MeshRangeType::ALL))
      connectivity_dd[e.index()][0] = e.index();
    auto connectivity
        = std::make_shared<graph::AdjacencyList<std::int32_t>>(connectivity_dd);
    topology.set_connectivity(connectivity, d0, d1);
  }
  else if (d0 < d1)
  {
    // Compute connectivity d1 - d0 and take transpose
    compute_connectivity(mesh, d1, d0);
    auto c = std::make_shared<graph::AdjacencyList<std::int32_t>>(
        compute_from_transpose(mesh, d0, d1));
    topology.set_connectivity(c, d0, d1);
  }
  else if (d0 > d1)
  {
    // Compute by mapping vertices from a lower dimension entity to
    // those of a higher dimension entity
    auto c = std::make_shared<graph::AdjacencyList<std::int32_t>>(
        compute_from_map(mesh.topology(),
                         mesh::cell_entity_type(mesh.cell_type(), d0), d0, d1));
    topology.set_connectivity(c, d0, d1);
  }
  else
    throw std::runtime_error("Entity dimension error when computing topology.");
}
//--------------------------------------------------------------------------
