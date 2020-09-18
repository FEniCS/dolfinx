// Copyright (C) 2013-2020 Chris Richardson
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "ParallelRefinement.h"
#include <dolfinx/common/MPI.h>
#include <dolfinx/common/types.h>
#include <dolfinx/fem/ElementDofLayout.h>
#include <dolfinx/graph/Partitioning.h>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/MeshTags.h>
#include <dolfinx/mesh/Partitioning.h>
#include <dolfinx/mesh/Topology.h>
#include <dolfinx/mesh/TopologyComputation.h>
#include <dolfinx/mesh/utils.h>
#include <map>
#include <vector>

using namespace dolfinx;
using namespace dolfinx::refinement;

namespace
{

/// Compute markers for interior/boundary vertices
/// @param[in] topology_local Local topology
/// @return Array where the ith entry is true if the ith vertex is on
///   the boundary
std::vector<bool>
compute_vertex_exterior_markers(const mesh::Topology& topology_local)
{
  // Get list of boundary vertices
  const int dim = topology_local.dim();
  auto facet_cell = topology_local.connectivity(dim - 1, dim);
  if (!facet_cell)
  {
    throw std::runtime_error(
        "Need facet-cell connectivity to build distributed adjacency list.");
  }

  auto facet_vertex = topology_local.connectivity(dim - 1, 0);
  if (!facet_vertex)
  {
    throw std::runtime_error(
        "Need facet-vertex connectivity to build distributed adjacency list.");
  }

  auto map_vertex = topology_local.index_map(0);
  if (!map_vertex)
    throw std::runtime_error("Need vertex IndexMap from topology.");
  assert(map_vertex->num_ghosts() == 0);

  std::vector<bool> exterior_vertex(map_vertex->size_local(), false);
  for (int f = 0; f < facet_cell->num_nodes(); ++f)
  {
    if (facet_cell->num_links(f) == 1)
    {
      auto vertices = facet_vertex->links(f);
      for (int j = 0; j < vertices.rows(); ++j)
        exterior_vertex[vertices[j]] = true;
    }
  }

  return exterior_vertex;
}
//-------------------------------------------------------------

std::int64_t local_to_global(std::int32_t local_index,
                             const common::IndexMap& map)
{
  assert(local_index >= 0);
  const std::array local_range = map.local_range();
  const std::int32_t local_size = (local_range[1] - local_range[0]);
  if (local_index < local_size)
  {
    const std::int64_t global_offset = local_range[0];
    return global_offset + local_index;
  }
  else
  {
    const Eigen::Array<std::int64_t, Eigen::Dynamic, 1>& ghosts = map.ghosts();
    assert((local_index - local_size) < ghosts.size());
    return ghosts[local_index - local_size];
  }
}

//-----------------------------------------------------------------------------
// Create geometric points of new Mesh, from current Mesh and a edge_to_vertex
// map listing the new local points (midpoints of those edges)
// @param Mesh
// @param local_edge_to_new_vertex
// @return array of points
Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
create_new_geometry(
    const mesh::Mesh& mesh,
    const std::map<std::int32_t, std::int64_t>& local_edge_to_new_vertex)
{
  // Build map from vertex -> geometry dof
  const graph::AdjacencyList<std::int32_t>& x_dofmap = mesh.geometry().dofmap();
  const int tdim = mesh.topology().dim();
  auto c_to_v = mesh.topology().connectivity(tdim, 0);
  assert(c_to_v);
  auto map_v = mesh.topology().index_map(0);
  assert(map_v);
  std::vector<std::int32_t> vertex_to_x(map_v->size_local()
                                        + map_v->num_ghosts());
  auto map_c = mesh.topology().index_map(tdim);
  assert(map_c);
  for (int c = 0; c < map_c->size_local() + map_c->num_ghosts(); ++c)
  {
    auto vertices = c_to_v->links(c);
    auto dofs = x_dofmap.links(c);
    for (int i = 0; i < vertices.rows(); ++i)
    {
      // FIXME: We are making an assumption here on the
      // ElementDofLayout. We should use an ElementDofLayout to map
      // between local vertex index and x dof index.
      vertex_to_x[vertices[i]] = dofs(i);
    }
  }

  // Copy over existing mesh vertices
  const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>& x_g
      = mesh.geometry().x();

  const std::int32_t num_vertices = map_v->size_local();
  const std::int32_t num_new_vertices = local_edge_to_new_vertex.size();
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      new_vertex_coordinates(num_vertices + num_new_vertices, 3);

  for (int v = 0; v < num_vertices; ++v)
    new_vertex_coordinates.row(v) = x_g.row(vertex_to_x[v]);

  Eigen::Array<int, Eigen::Dynamic, 1> edges(num_new_vertices);
  int i = 0;
  for (auto& e : local_edge_to_new_vertex)
    edges[i++] = e.first;

  const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor> midpoints
      = mesh::midpoints(mesh, 1, edges);
  new_vertex_coordinates.bottomRows(num_new_vertices) = midpoints;

  const int gdim = mesh.geometry().dim();
  return new_vertex_coordinates.leftCols(gdim);
}
} // namespace

//-----------------------------------------------------------------------------
ParallelRefinement::ParallelRefinement(const mesh::Mesh& mesh) : _mesh(mesh)
{
  if (!_mesh.topology().connectivity(1, 0))
    throw std::runtime_error("Edges must be initialised");

  auto map_e = mesh.topology().index_map(1);
  assert(map_e);
  const std::int32_t num_edges = map_e->size_local() + map_e->num_ghosts();
  _marked_edges = std::vector<bool>(num_edges, false);

  // Create shared edges, for both owned and ghost indices
  // returning edge -> set(global process numbers)
  std::map<std::int32_t, std::set<int>> shared_edges
      = _mesh.topology().index_map(1)->compute_shared_indices();

  // Compute a slightly wider neighborhood for direct communication of shared
  // edges
  std::set<int> all_neighbor_set;
  for (const auto& q : shared_edges)
    all_neighbor_set.insert(q.second.begin(), q.second.end());
  std::vector<int> neighbors(all_neighbor_set.begin(), all_neighbor_set.end());

  MPI_Dist_graph_create_adjacent(
      mesh.mpi_comm(), neighbors.size(), neighbors.data(), MPI_UNWEIGHTED,
      neighbors.size(), neighbors.data(), MPI_UNWEIGHTED, MPI_INFO_NULL, false,
      &_neighbor_comm);

  // Create a "shared_edge to neighbor map"
  std::map<int, int> proc_to_neighbor;
  for (std::size_t i = 0; i < neighbors.size(); ++i)
    proc_to_neighbor.insert({neighbors[i], i});

  for (auto& q : shared_edges)
  {
    std::set<int> neighbor_set;
    for (int r : q.second)
      neighbor_set.insert(proc_to_neighbor[r]);
    _shared_edges.insert({q.first, neighbor_set});
  }

  _marked_for_update.resize(neighbors.size());
}
//-----------------------------------------------------------------------------
ParallelRefinement::~ParallelRefinement() { MPI_Comm_free(&_neighbor_comm); }
//-----------------------------------------------------------------------------
const std::vector<bool>& ParallelRefinement::marked_edges() const
{
  return _marked_edges;
}
//-----------------------------------------------------------------------------
bool ParallelRefinement::mark(std::int32_t edge_index)
{
  auto map1 = _mesh.topology().index_map(1);
  assert(map1);
  assert(edge_index < (map1->size_local() + map1->num_ghosts()));

  // Already marked, so nothing to do
  if (_marked_edges[edge_index])
    return false;

  _marked_edges[edge_index] = true;

  // If it is a shared edge, add all sharing neighbors to update set
  if (auto map_it = _shared_edges.find(edge_index);
      map_it != _shared_edges.end())
  {
    const std::int64_t global_index = local_to_global(edge_index, *map1);
    for (int p : map_it->second)
      _marked_for_update[p].push_back(global_index);
  }

  return true;
}
//-----------------------------------------------------------------------------
void ParallelRefinement::mark_all()
{
  auto edge_im = _mesh.topology().index_map(1);
  _marked_edges.assign(edge_im->size_local() + edge_im->num_ghosts(), true);
}
//-----------------------------------------------------------------------------
void ParallelRefinement::mark(
    const mesh::MeshTags<std::int8_t>& refinement_marker)
{
  const std::size_t entity_dim = refinement_marker.dim();

  const std::vector<std::int32_t>& marker_indices = refinement_marker.indices();

  std::shared_ptr<const mesh::Mesh> mesh = refinement_marker.mesh();
  auto map_ent = mesh->topology().index_map(entity_dim);
  assert(map_ent);

  auto ent_to_edge = mesh->topology().connectivity(entity_dim, 1);
  if (!ent_to_edge)
    throw std::runtime_error("Connectivity missing: ("
                             + std::to_string(entity_dim) + ", 1)");

  for (const auto& i : marker_indices)
  {
    const auto edges = ent_to_edge->links(i);
    for (int j = 0; j < edges.rows(); ++j)
      mark(edges[j]);
  }
}
//-----------------------------------------------------------------------------
void ParallelRefinement::update_logical_edgefunction()
{
  std::vector<std::int32_t> send_offsets = {0};
  std::vector<std::int64_t> data_to_send;
  int num_neighbors = _marked_for_update.size();
  for (int i = 0; i < num_neighbors; ++i)
  {
    data_to_send.insert(data_to_send.end(), _marked_for_update[i].begin(),
                        _marked_for_update[i].end());
    _marked_for_update[i].clear();
    send_offsets.push_back(data_to_send.size());
  }

  // Send all shared edges marked for update and receive from other
  // processes
  const Eigen::Array<std::int64_t, Eigen::Dynamic, 1> data_to_recv
      = MPI::neighbor_all_to_all(_neighbor_comm, send_offsets, data_to_send)
            .array();

  // Flatten received values and set _marked_edges at each index received
  const std::vector local_indices
      = _mesh.topology().index_map(1)->global_to_local(data_to_recv);
  for (std::int32_t local_index : local_indices)
    _marked_edges[local_index] = true;
}
//-----------------------------------------------------------------------------
std::map<std::int32_t, std::int64_t> ParallelRefinement::create_new_vertices()
{
  // Take marked_edges and use to create new vertices
  const std::shared_ptr<const common::IndexMap> edge_index_map
      = _mesh.topology().index_map(1);

  // Add new edge midpoints to list of vertices
  int n = 0;
  std::map<std::int32_t, std::int64_t> local_edge_to_new_vertex;
  for (int local_i = 0; local_i < edge_index_map->size_local(); ++local_i)
  {
    if (_marked_edges[local_i] == true)
    {
      auto it = local_edge_to_new_vertex.insert({local_i, n});
      assert(it.second);
      ++n;
    }
  }
  const int num_new_vertices = n;
  const std::size_t global_offset
      = MPI::global_offset(_mesh.mpi_comm(), num_new_vertices, true)
        + _mesh.topology().index_map(0)->local_range()[1];

  for (auto& e : local_edge_to_new_vertex)
    e.second += global_offset;

  // Create actual points
  _new_vertex_coordinates
      = create_new_geometry(_mesh, local_edge_to_new_vertex);

  // If they are shared, then the new global vertex index needs to be
  // sent off-process.

  int num_neighbors = _marked_for_update.size();
  std::vector<std::vector<std::int64_t>> values_to_send(num_neighbors);
  for (auto& local_edge : local_edge_to_new_vertex)
  {
    const std::size_t local_i = local_edge.first;
    // shared, but locally owned : remote owned are not in list.

    if (auto shared_edge_i = _shared_edges.find(local_i);
        shared_edge_i != _shared_edges.end())
    {
      for (int remote_process : shared_edge_i->second)
      {
        // send map from global edge index to new global vertex index
        values_to_send[remote_process].push_back(
            local_to_global(local_edge.first, *edge_index_map));
        values_to_send[remote_process].push_back(local_edge.second);
      }
    }
  }

  // Send new vertex indices to edge neighbors and receive
  std::vector<std::int64_t> send_values;
  std::vector<int> send_offsets = {0};
  for (int i = 0; i < num_neighbors; ++i)
  {
    send_values.insert(send_values.end(), values_to_send[i].begin(),
                       values_to_send[i].end());
    send_offsets.push_back(send_values.size());
  }

  const Eigen::Array<std::int64_t, Eigen::Dynamic, 1> received_values
      = MPI::neighbor_all_to_all(_neighbor_comm, send_offsets, send_values)
            .array();

  // Add received remote global vertex indices to map
  std::vector<std::int64_t> recv_global_edge;
  assert(received_values.size() % 2 == 0);
  for (int i = 0; i < received_values.size() / 2; ++i)
    recv_global_edge.push_back(received_values[i * 2]);
  std::vector recv_local_edge
      = _mesh.topology().index_map(1)->global_to_local(recv_global_edge);
  for (int i = 0; i < received_values.size() / 2; ++i)
  {
    auto it = local_edge_to_new_vertex.insert(
        {recv_local_edge[i], received_values[i * 2 + 1]});
    assert(it.second);
  }

  return local_edge_to_new_vertex;
}
//-----------------------------------------------------------------------------
std::vector<std::int64_t> ParallelRefinement::adjust_indices(
    const std::shared_ptr<const common::IndexMap>& index_map, std::int32_t n)
{
  // Add in an extra "n" indices at the end of the current local_range
  // of "index_map", and adjust existing indices to match.

  // Get number of new indices on all processes
  int mpi_size = dolfinx::MPI::size(index_map->comm());
  int mpi_rank = dolfinx::MPI::rank(index_map->comm());
  std::vector<std::int32_t> recvn(mpi_size);
  MPI_Allgather(&n, 1, MPI_INT32_T, recvn.data(), 1, MPI_INT32_T,
                index_map->comm());
  std::vector<std::int64_t> global_offsets = {0};
  for (std::int32_t r : recvn)
    global_offsets.push_back(global_offsets.back() + r);

  std::vector global_indices = index_map->global_indices(true);

  Eigen::Array<int, Eigen::Dynamic, 1> ghost_owners
      = index_map->ghost_owner_rank();
  int local_size = index_map->size_local();
  for (int i = 0; i < local_size; ++i)
    global_indices[i] += global_offsets[mpi_rank];
  for (int i = 0; i < ghost_owners.size(); ++i)
    global_indices[local_size + i] += global_offsets[ghost_owners[i]];

  return global_indices;
}
//-----------------------------------------------------------------------------
mesh::Mesh ParallelRefinement::build_local(
    const std::vector<std::int64_t>& cell_topology) const
{
  const std::size_t tdim = _mesh.topology().dim();
  const std::size_t num_cell_vertices = tdim + 1;
  assert(cell_topology.size() % num_cell_vertices == 0);
  const std::size_t num_cells = cell_topology.size() / num_cell_vertices;

  Eigen::Map<const Eigen::Array<std::int64_t, Eigen::Dynamic, Eigen::Dynamic,
                                Eigen::RowMajor>>
      cells(cell_topology.data(), num_cells, num_cell_vertices);

  mesh::Mesh mesh = mesh::create_mesh(
      _mesh.mpi_comm(), graph::AdjacencyList<std::int64_t>(cells),
      _mesh.geometry().cmap(), _new_vertex_coordinates, mesh::GhostMode::none);
  assert(mesh.geometry().dim() == _mesh.geometry().dim());
  return mesh;
}
//-----------------------------------------------------------------------------
mesh::Mesh
ParallelRefinement::partition(const std::vector<std::int64_t>& cell_topology,
                              int num_ghost_cells, bool redistribute) const
{
  const int num_vertices_per_cell
      = mesh::cell_num_entities(_mesh.topology().cell_type(), 0);

  const std::int32_t num_local_cells
      = cell_topology.size() / num_vertices_per_cell;
  std::vector<std::int64_t> global_cell_indices(num_local_cells);
  const std::size_t idx_global_offset
      = MPI::global_offset(_mesh.mpi_comm(), num_local_cells, true);
  for (std::int32_t i = 0; i < num_local_cells; i++)
    global_cell_indices[i] = idx_global_offset + i;

  // Check if mesh has ghost cells on any rank
  int max_ghost_cells = 0;
  MPI_Allreduce(&num_ghost_cells, &max_ghost_cells, 1, MPI_INT, MPI_MAX,
                _mesh.mpi_comm());

  // Build mesh
  if (redistribute)
  {
    Eigen::Map<const Eigen::Array<std::int64_t, Eigen::Dynamic, Eigen::Dynamic,
                                  Eigen::RowMajor>>
        cells(cell_topology.data(), num_local_cells - num_ghost_cells,
              num_vertices_per_cell);

    if (max_ghost_cells == 0)
    {
      return mesh::create_mesh(_mesh.mpi_comm(),
                               graph::AdjacencyList<std::int64_t>(cells),
                               _mesh.geometry().cmap(), _new_vertex_coordinates,
                               mesh::GhostMode::none);
    }
    else
    {
      return mesh::create_mesh(_mesh.mpi_comm(),
                               graph::AdjacencyList<std::int64_t>(cells),
                               _mesh.geometry().cmap(), _new_vertex_coordinates,
                               mesh::GhostMode::shared_facet);
    }
  }

  if (max_ghost_cells > 0)
  {
    throw std::runtime_error("Refinement of ghosted meshes without "
                             "re-partitioning is not supported yet.");
  }

  Eigen::Map<const Eigen::Array<std::int64_t, Eigen::Dynamic, Eigen::Dynamic,
                                Eigen::RowMajor>>
      cells(cell_topology.data(), num_local_cells - num_ghost_cells,
            num_vertices_per_cell);
  MPI_Comm comm = _mesh.mpi_comm();
  mesh::Topology topology(comm, _mesh.geometry().cmap().cell_shape());
  const graph::AdjacencyList<std::int64_t> my_cells(cells);
  {
    auto [cells_local, local_to_global_vertices]
        = graph::Partitioning::create_local_adjacency_list(my_cells);

    // Create (i) local topology object and (ii) IndexMap for cells, and
    // set cell-vertex topology
    mesh::Topology topology_local(comm, _mesh.geometry().cmap().cell_shape());
    const int tdim = topology_local.dim();
    auto map = std::make_shared<common::IndexMap>(
        comm, cells_local.num_nodes(), std::vector<int>(),
        std::vector<std::int64_t>(), std::vector<int>(), 1);
    topology_local.set_index_map(tdim, map);
    auto _cells_local
        = std::make_shared<graph::AdjacencyList<std::int32_t>>(cells_local);
    topology_local.set_connectivity(_cells_local, tdim, 0);

    const int n = local_to_global_vertices.size();
    map = std::make_shared<common::IndexMap>(comm, n, std::vector<int>(),
                                             std::vector<std::int64_t>(),
                                             std::vector<int>(), 1);
    topology_local.set_index_map(0, map);
    auto _vertices_local
        = std::make_shared<graph::AdjacencyList<std::int32_t>>(n);
    topology_local.set_connectivity(_vertices_local, 0, 0);

    // Create facets for local topology, and attach to the topology
    // object. This will be used to find possibly shared cells
    auto [cf, fv, map0] = mesh::TopologyComputation::compute_entities(
        comm, topology_local, tdim - 1);
    if (cf)
      topology_local.set_connectivity(cf, tdim, tdim - 1);
    if (map0)
      topology_local.set_index_map(tdim - 1, map0);
    if (fv)
      topology_local.set_connectivity(fv, tdim - 1, 0);
    auto [fc, ignore] = mesh::TopologyComputation::compute_connectivity(
        topology_local, tdim - 1, tdim);
    if (fc)
      topology_local.set_connectivity(fc, tdim - 1, tdim);

    // Get facets that are on the boundary of the local topology, i.e
    // are connect to one cell only
    const std::vector boundary = mesh::compute_boundary_facets(topology_local);

    // Build distributed cell-vertex AdjacencyList, IndexMap for
    // vertices, and map from local index to old global index
    const std::vector<bool>& exterior_vertices
        = compute_vertex_exterior_markers(topology_local);
    auto [cells_d, vertex_map]
        = graph::Partitioning::create_distributed_adjacency_list(
            comm, *_cells_local, local_to_global_vertices, exterior_vertices);

    // Set vertex IndexMap, and vertex-vertex connectivity
    auto _vertex_map
        = std::make_shared<common::IndexMap>(std::move(vertex_map));
    topology.set_index_map(0, _vertex_map);
    auto c0 = std::make_shared<graph::AdjacencyList<std::int32_t>>(
        _vertex_map->size_local() + _vertex_map->num_ghosts());
    topology.set_connectivity(c0, 0, 0);

    // Set cell IndexMap and cell-vertex connectivity
    auto index_map_c = std::make_shared<common::IndexMap>(
        comm, cells_d.num_nodes(), std::vector<int>(),
        std::vector<std::int64_t>(), std::vector<int>(), 1);
    topology.set_index_map(tdim, index_map_c);
    auto _cells_d
        = std::make_shared<graph::AdjacencyList<std::int32_t>>(cells_d);
    topology.set_connectivity(_cells_d, tdim, 0);
  }

  mesh::Geometry geometry
      = mesh::create_geometry(comm, topology, _mesh.geometry().cmap(), my_cells,
                              _new_vertex_coordinates);

  return mesh::Mesh(comm, std::move(topology), std::move(geometry));
}
//-----------------------------------------------------------------------------
