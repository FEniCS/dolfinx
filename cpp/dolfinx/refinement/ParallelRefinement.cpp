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
#include <dolfinx/mesh/DistributedMeshTools.h>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/MeshFunction.h>
#include <dolfinx/mesh/Partitioning.h>
#include <dolfinx/mesh/Topology.h>
#include <dolfinx/mesh/TopologyComputation.h>
#include <dolfinx/mesh/utils.h>
#include <map>
#include <vector>

using namespace dolfinx;
using namespace dolfinx::refinement;

//-----------------------------------------------------------------------------
ParallelRefinement::ParallelRefinement(const mesh::Mesh& mesh)
    : _mesh(mesh), _marked_edges(mesh.num_entities(1), false)
{
  if (!_mesh.topology().connectivity(1, 0))
    throw std::runtime_error("Edges must be initialised");

  // Create shared edges, for both owned and ghost indices
  // returning edge -> set(global process numbers)
  std::map<std::int32_t, std::set<int>> shared_edges
      = _mesh.topology().index_map(1)->compute_shared_indices();

  // Compute a slightly wider neighbourhood for direct communication of shared
  // edges
  std::set<int> all_neighbour_set;
  for (auto q : shared_edges)
    all_neighbour_set.insert(q.second.begin(), q.second.end());
  std::vector<int> neighbours(all_neighbour_set.begin(),
                              all_neighbour_set.end());

  MPI_Dist_graph_create_adjacent(
      mesh.mpi_comm(), neighbours.size(), neighbours.data(), MPI_UNWEIGHTED,
      neighbours.size(), neighbours.data(), MPI_UNWEIGHTED, MPI_INFO_NULL,
      false, &_neighbour_comm);

  // Create a "shared_edge to neighbour map"
  std::map<int, int> proc_to_neighbour;
  for (std::size_t i = 0; i < neighbours.size(); ++i)
    proc_to_neighbour.insert({neighbours[i], i});

  for (auto& q : shared_edges)
  {
    std::set<int> neighbour_set;
    for (int r : q.second)
      neighbour_set.insert(proc_to_neighbour[r]);
    _shared_edges.insert({q.first, neighbour_set});
  }

  _marked_for_update.resize(neighbours.size());
}
//-----------------------------------------------------------------------------
ParallelRefinement::~ParallelRefinement() { MPI_Comm_free(&_neighbour_comm); }
//-----------------------------------------------------------------------------
const std::vector<bool>& ParallelRefinement::marked_edges() const
{
  return _marked_edges;
}
//-----------------------------------------------------------------------------
bool ParallelRefinement::mark(std::int32_t edge_index)
{
  assert(edge_index < _mesh.num_entities(1));

  // Already marked, so nothing to do
  if (_marked_edges[edge_index])
    return false;

  _marked_edges[edge_index] = true;

  // If it is a shared edge, add all sharing neighbours to update set
  auto map_it = _shared_edges.find(edge_index);
  if (map_it != _shared_edges.end())
  {
    const std::int64_t global_index
        = _mesh.topology().index_map(1)->local_to_global(edge_index);
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
const std::map<std::int32_t, std::int64_t>&
ParallelRefinement::edge_to_new_vertex() const
{
  return _local_edge_to_new_vertex;
}
//-----------------------------------------------------------------------------
void ParallelRefinement::mark(const mesh::MeshFunction<int>& refinement_marker)
{
  const std::size_t entity_dim = refinement_marker.dim();

  // Get reference to mesh function data array
  const Eigen::Array<int, Eigen::Dynamic, 1>& mf_values
      = refinement_marker.values();

  std::shared_ptr<const mesh::Mesh> mesh = refinement_marker.mesh();
  auto map_ent = mesh->topology().index_map(entity_dim);
  assert(map_ent);
  const int num_entities = map_ent->size_local() + map_ent->num_ghosts();

  auto ent_to_edge = mesh->topology().connectivity(entity_dim, 1);
  if (!ent_to_edge)
    throw std::runtime_error("Connectivity missing: ("
                             + std::to_string(entity_dim) + ", 1)");

  for (int i = 0; i < num_entities; ++i)
  {
    if (mf_values[i] == 1)
    {
      auto edges = ent_to_edge->links(i);
      for (int j = 0; j < edges.rows(); ++j)
        mark(edges[j]);
    }
  }
}
//-----------------------------------------------------------------------------
void ParallelRefinement::update_logical_edgefunction()
{
  std::vector<std::int32_t> send_offsets = {0};
  std::vector<std::int32_t> recv_offsets;
  std::vector<std::int64_t> data_to_send, data_to_recv;
  int num_neighbours = _marked_for_update.size();
  for (int i = 0; i < num_neighbours; ++i)
  {
    data_to_send.insert(data_to_send.end(), _marked_for_update[i].begin(),
                        _marked_for_update[i].end());
    _marked_for_update[i].clear();
    send_offsets.push_back(data_to_send.size());
  }

  // Send all shared edges marked for update and receive from other
  // processes
  MPI::neighbor_all_to_all(_neighbour_comm, send_offsets, data_to_send,
                           recv_offsets, data_to_recv);

  // Flatten received values and set _marked_edges at each index received
  std::vector<std::int32_t> local_indices
      = _mesh.topology().index_map(1)->global_to_local(data_to_recv);
  for (std::int32_t local_index : local_indices)
    _marked_edges[local_index] = true;
}
//-----------------------------------------------------------------------------
void ParallelRefinement::create_new_vertices()
{
  // Take marked_edges and use to create new vertices
  const std::shared_ptr<const common::IndexMap> edge_index_map
      = _mesh.topology().index_map(1);

  // Build map from vertex -> geometry dof
  const graph::AdjacencyList<std::int32_t>& x_dofmap
      = _mesh.geometry().dofmap();
  const int tdim = _mesh.topology().dim();
  auto c_to_v = _mesh.topology().connectivity(tdim, 0);
  assert(c_to_v);
  auto map_v = _mesh.topology().index_map(0);
  assert(map_v);
  const std::int32_t num_vertices = map_v->size_local() + map_v->num_ghosts();
  std::vector<std::int32_t> vertex_to_x(num_vertices);
  auto map_c = _mesh.topology().index_map(tdim);
  assert(map_c);
  for (int c = 0; c < map_c->size_local() + map_c->num_ghosts(); ++c)
  {
    auto vertices = c_to_v->links(c);
    auto dofs = x_dofmap.links(c);
    for (int i = 0; i < vertices.rows(); ++i)
    {
      // FIXME: We are making an assumption here on the
      // ElementDofLayout. We should use an ElementDofLayout to map
      // between local vertex index an x dof index.
      vertex_to_x[vertices[i]] = dofs(i);
    }
  }

  // Copy over existing mesh vertices
  const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>& x_g
      = _mesh.geometry().x();

  std::int32_t num_new_vertices
      = std::count(_marked_edges.begin(), _marked_edges.end(), true);

  _new_vertex_coordinates.resize(num_vertices + num_new_vertices, 3);

  for (int v = 0; v < num_vertices; ++v)
    _new_vertex_coordinates.row(v) = x_g.row(vertex_to_x[v]);

  // Compute all edge mid-points
  Eigen::Array<int, Eigen::Dynamic, 1> edges(_mesh.num_entities(1));
  std::iota(edges.data(), edges.data() + edges.rows(), 0);
  const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor> midpoints
      = mesh::midpoints(_mesh, 1, edges);

  // Tally up unshared marked edges, and shared marked edges which are
  // owned on this process. Index them sequentially from zero.
  // Locally owned edges

  std::int64_t n = 0;
  for (int local_i = 0; local_i < edge_index_map->size_local(); ++local_i)
  {
    if (_marked_edges[local_i] == true)
    {
      _new_vertex_coordinates.row(n + num_vertices) = midpoints.row(local_i);
      auto it = _local_edge_to_new_vertex.insert({local_i, n});
      assert(it.second);
      ++n;
    }
  }

  // Calculate global range for new local vertices
  const std::size_t global_offset
      = MPI::global_offset(_mesh.mpi_comm(), num_new_vertices, true)
        + _mesh.topology().index_map(0)->size_global();

  // If they are shared, then the new global vertex index needs to be
  // sent off-process.  Add offset to map, and collect up any shared
  // new vertices that need to send the new index off-process

  int num_neighbours = _marked_for_update.size();
  std::vector<std::vector<std::int64_t>> values_to_send(num_neighbours);
  for (auto& local_edge : _local_edge_to_new_vertex)
  {
    // Add global_offset to map, to get new global index of new
    // vertices
    local_edge.second += global_offset;

    const std::size_t local_i = local_edge.first;
    // shared, but locally owned : remote owned are not in list.
    auto shared_edge_i = _shared_edges.find(local_i);
    if (shared_edge_i != _shared_edges.end())
    {
      for (int remote_process : shared_edge_i->second)
      {
        // send mapping from global edge index to new global vertex index
        values_to_send[remote_process].push_back(
            edge_index_map->local_to_global(local_edge.first));
        values_to_send[remote_process].push_back(local_edge.second);
      }
    }
  }

  // Send new vertex indices to edge neighbours and receive
  std::vector<std::int64_t> send_values;
  std::vector<int> send_offsets = {0};
  for (int i = 0; i < num_neighbours; ++i)
  {
    send_values.insert(send_values.end(), values_to_send[i].begin(),
                       values_to_send[i].end());
    send_offsets.push_back(send_values.size());
  }

  std::vector<int> recv_offsets;
  std::vector<std::int64_t> received_values;
  MPI::neighbor_all_to_all(_neighbour_comm, send_offsets, send_values,
                           recv_offsets, received_values);

  // Add received remote global vertex indices to map
  std::vector<std::int64_t> recv_global_edge;
  assert(received_values.size() % 2 == 0);
  for (std::size_t i = 0; i < received_values.size() / 2; ++i)
    recv_global_edge.push_back(received_values[i * 2]);
  std::vector<std::int32_t> recv_local_edge
      = _mesh.topology().index_map(1)->global_to_local(recv_global_edge);
  for (std::size_t i = 0; i < received_values.size() / 2; ++i)
  {
    auto it = _local_edge_to_new_vertex.insert(
        {recv_local_edge[i], received_values[i * 2 + 1]});
    assert(it.second);
  }

  // Attach global indices to each vertex, old and new, and sort
  // them across processes into this order
  std::vector<std::int64_t> global_indices
      = _mesh.topology().index_map(0)->global_indices(false);
  for (std::int32_t i = 0; i < num_new_vertices; i++)
    global_indices.push_back(i + global_offset);

  _new_vertex_coordinates
      = mesh::DistributedMeshTools::reorder_by_global_indices(
          _mesh.mpi_comm(), _new_vertex_coordinates, global_indices);
}
//-----------------------------------------------------------------------------
mesh::Mesh ParallelRefinement::build_local() const
{
  const std::size_t tdim = _mesh.topology().dim();
  const std::size_t num_cell_vertices = tdim + 1;
  assert(_new_cell_topology.size() % num_cell_vertices == 0);
  const std::size_t num_cells = _new_cell_topology.size() / num_cell_vertices;

  Eigen::Map<const Eigen::Array<std::int64_t, Eigen::Dynamic, Eigen::Dynamic,
                                Eigen::RowMajor>>
      cells(_new_cell_topology.data(), num_cells, num_cell_vertices);

  const fem::ElementDofLayout layout
      = fem::geometry_layout(_mesh.topology().cell_type(), cells.cols());
  mesh::Mesh mesh = mesh::create(
      _mesh.mpi_comm(), graph::AdjacencyList<std::int64_t>(cells), layout,
      _new_vertex_coordinates, mesh::GhostMode::none);

  return mesh;
}
//-----------------------------------------------------------------------------
mesh::Mesh ParallelRefinement::partition(bool redistribute) const
{
  const int num_vertices_per_cell
      = mesh::cell_num_entities(_mesh.topology().cell_type(), 0);

  const std::int32_t num_local_cells
      = _new_cell_topology.size() / num_vertices_per_cell;
  std::vector<std::int64_t> global_cell_indices(num_local_cells);
  const std::size_t idx_global_offset
      = MPI::global_offset(_mesh.mpi_comm(), num_local_cells, true);
  for (std::int32_t i = 0; i < num_local_cells; i++)
    global_cell_indices[i] = idx_global_offset + i;

  Eigen::Map<const Eigen::Array<std::int64_t, Eigen::Dynamic, Eigen::Dynamic,
                                Eigen::RowMajor>>
      cells(_new_cell_topology.data(), num_local_cells, num_vertices_per_cell);

  // Build mesh

  const fem::ElementDofLayout layout
      = fem::geometry_layout(_mesh.topology().cell_type(), cells.cols());
  if (redistribute)
  {
    return mesh::create(_mesh.mpi_comm(),
                        graph::AdjacencyList<std::int64_t>(cells), layout,
                        _new_vertex_coordinates, mesh::GhostMode::none);
  }

  MPI_Comm comm = _mesh.mpi_comm();
  mesh::Topology topology(layout.cell_type());
  const graph::AdjacencyList<std::int64_t> my_cells(cells);
  {
    auto [cells_local, local_to_global_vertices]
        = graph::Partitioning::create_local_adjacency_list(my_cells);

    // Create (i) local topology object and (ii) IndexMap for cells, and
    // set cell-vertex topology
    mesh::Topology topology_local(layout.cell_type());
    const int tdim = topology_local.dim();
    auto map = std::make_shared<common::IndexMap>(
        comm, cells_local.num_nodes(), std::vector<std::int64_t>(), 1);
    topology_local.set_index_map(tdim, map);
    auto _cells_local
        = std::make_shared<graph::AdjacencyList<std::int32_t>>(cells_local);
    topology_local.set_connectivity(_cells_local, tdim, 0);

    const int n = local_to_global_vertices.size();
    map = std::make_shared<common::IndexMap>(comm, n,
                                             std::vector<std::int64_t>(), 1);
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

    // FIXME: This looks weird. Revise.
    // Get facets that are on the boundary of the local topology, i.e
    // are connect to one cell only
    std::vector<bool> boundary = compute_interior_facets(topology_local);
    topology_local.set_interior_facets(boundary);
    boundary = topology_local.on_boundary(tdim - 1);

    // Build distributed cell-vertex AdjacencyList, IndexMap for
    // vertices, and map from local index to old global index
    const std::vector<bool>& exterior_vertices
        = mesh::Partitioning::compute_vertex_exterior_markers(topology_local);
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
        comm, cells_d.num_nodes(), std::vector<std::int64_t>(), 1);
    topology.set_index_map(tdim, index_map_c);
    auto _cells_d
        = std::make_shared<graph::AdjacencyList<std::int32_t>>(cells_d);
    topology.set_connectivity(_cells_d, tdim, 0);
  }

  const mesh::Geometry geometry = mesh::create_geometry(
      comm, topology, layout, my_cells, _new_vertex_coordinates);

  return mesh::Mesh(comm, topology, geometry);
}
//-----------------------------------------------------------------------------
void ParallelRefinement::new_cells(const std::vector<std::int64_t>& idx)
{
  _new_cell_topology.insert(_new_cell_topology.end(), idx.begin(), idx.end());
}
//-----------------------------------------------------------------------------
