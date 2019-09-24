// Copyright (C) 2013-2018 Chris Richardson
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "ParallelRefinement.h"
#include <dolfin/common/MPI.h>
#include <dolfin/common/types.h>
#include <dolfin/mesh/DistributedMeshTools.h>
#include <dolfin/mesh/Geometry.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshEntity.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/mesh/MeshIterator.h>
#include <dolfin/mesh/Partitioning.h>
#include <dolfin/mesh/utils.h>
#include <map>
#include <vector>

using namespace dolfin;
using namespace dolfin::refinement;

//-----------------------------------------------------------------------------
ParallelRefinement::ParallelRefinement(const mesh::Mesh& mesh)
    : _mesh(mesh), _marked_edges(mesh.num_entities(1), false),
      _marked_for_update(MPI::size(mesh.mpi_comm()))
{
  _mesh.create_global_indices(1);

  // Create a global-to-local map for shared edges
  const std::map<std::int32_t, std::set<std::int32_t>>& shared_edges
      = _mesh.topology().shared_entities(1);
  const std::vector<std::int64_t>& global_edge_indices
      = _mesh.topology().global_indices(1);

  for (const auto& edge : shared_edges)
  {
    _global_to_local_edge_map.insert(
        {global_edge_indices[edge.first], edge.first});
  }
}
//-----------------------------------------------------------------------------
const mesh::Mesh& ParallelRefinement::mesh() const { return _mesh; }
//-----------------------------------------------------------------------------
bool ParallelRefinement::is_marked(std::int32_t edge_index) const
{
  assert(edge_index < _mesh.num_entities(1));
  return _marked_edges[edge_index];
}
//-----------------------------------------------------------------------------
void ParallelRefinement::mark(std::int32_t edge_index)
{
  assert(edge_index < _mesh.num_entities(1));

  // Already marked, so nothing to do
  if (_marked_edges[edge_index])
    return;

  const std::map<std::int32_t, std::set<std::int32_t>>& shared_edges
      = _mesh.topology().shared_entities(1);
  const std::vector<std::int64_t>& global_edge_indices
      = _mesh.topology().global_indices(1);

  _marked_edges[edge_index] = true;
  auto map_it = shared_edges.find(edge_index);

  // If it is a shared edge, add all sharing procs to update set
  if (map_it != shared_edges.end())
  {
    std::int64_t global_index = global_edge_indices[edge_index];
    for (auto it : map_it->second)
      _marked_for_update[it].push_back(global_index);
  }
}
//-----------------------------------------------------------------------------
void ParallelRefinement::mark_all()
{
  _marked_edges.assign(_mesh.num_entities(1), true);
}
//-----------------------------------------------------------------------------
const std::map<std::size_t, std::size_t>&
ParallelRefinement::edge_to_new_vertex() const
{
  return _local_edge_to_new_vertex;
}
//-----------------------------------------------------------------------------
void ParallelRefinement::mark(const mesh::MeshEntity& entity)
{
  for (const auto& edge : mesh::EntityRange(entity, 1))
    mark(edge.index());
}
//-----------------------------------------------------------------------------
void ParallelRefinement::mark(const mesh::MeshFunction<int>& refinement_marker)
{
  const std::size_t entity_dim = refinement_marker.dim();

  // Get reference to mesh function data array
  Eigen::Ref<const Eigen::Array<int, Eigen::Dynamic, 1>> mf_values
      = refinement_marker.values();

  for (const auto& entity : mesh::MeshRange(_mesh, entity_dim))
  {
    if (mf_values[entity.index()] == 1)
    {
      for (const auto& edge : mesh::EntityRange(entity, 1))
        mark(edge.index());
    }
  }
}
//-----------------------------------------------------------------------------
std::vector<std::size_t>
ParallelRefinement::marked_edge_list(const mesh::MeshEntity& cell) const
{
  std::vector<std::size_t> result;

  std::size_t i = 0;
  for (const auto& edge : mesh::EntityRange(cell, 1))
  {
    if (_marked_edges[edge.index()])
      result.push_back(i);
    ++i;
  }
  return result;
}
//-----------------------------------------------------------------------------
void ParallelRefinement::update_logical_edgefunction()
{
  const std::size_t mpi_size = MPI::size(_mesh.mpi_comm());

  // Send all shared edges marked for update and receive from other
  // processes
  std::vector<std::int64_t> received_values;
  MPI::all_to_all(_mesh.mpi_comm(), _marked_for_update, received_values);

  // Clear marked_for_update vectors
  _marked_for_update = std::vector<std::vector<std::int64_t>>(mpi_size);

  // Flatten received values and set edges mesh::MeshFunction true at
  // each index received
  for (auto const& global_index : received_values)
  {
    const auto map_it = _global_to_local_edge_map.find(global_index);
    assert(map_it != _global_to_local_edge_map.end());
    _marked_edges[map_it->second] = true;
  }
}
//-----------------------------------------------------------------------------
void ParallelRefinement::create_new_vertices()
{
  // Take marked_edges and use to create new vertices

  const std::int32_t mpi_size = MPI::size(_mesh.mpi_comm());
  const std::int32_t mpi_rank = MPI::rank(_mesh.mpi_comm());

  const std::map<std::int32_t, std::set<std::int32_t>>& shared_edges
      = _mesh.topology().shared_entities(1);
  const std::vector<std::int64_t>& global_edge_indices
      = _mesh.topology().global_indices(1);

  // Copy over existing mesh vertices
  _new_vertex_coordinates = std::vector<double>(
      _mesh.geometry().points().data(),
      _mesh.geometry().points().data() + _mesh.geometry().points().size());

  // Compute all edge mid-points
  Eigen::Array<int, Eigen::Dynamic, 1> edges(_mesh.num_entities(1));
  std::iota(edges.data(), edges.data() + edges.rows(), 0);
  const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor> midpoints
      = mesh::midpoints(_mesh, 1, edges);

  // Tally up unshared marked edges, and shared marked edges which are
  // owned on this process.  Index them sequentially from zero.
  std::size_t n = 0;
  for (std::int32_t local_i = 0; local_i < _mesh.num_entities(1); ++local_i)
  {
    if (_marked_edges[local_i] == true)
    {
      // Assume this edge is owned locally
      bool owner = true;

      // If shared, check that this is true
      auto shared_edge_i = shared_edges.find(local_i);
      if (shared_edge_i != shared_edges.end())
      {
        // check if any other sharing process has a lower rank
        for (auto proc_edge : shared_edge_i->second)
        {
          if (proc_edge < mpi_rank)
            owner = false;
        }
      }

      // If it is still believed to be owned on this process, add to
      // list
      if (owner)
      {
        for (std::size_t j = 0; j < 3; ++j)
          _new_vertex_coordinates.push_back(midpoints(local_i, j));
        _local_edge_to_new_vertex[local_i] = n++;
      }
    }
  }

  // Calculate global range for new local vertices
  const std::size_t num_new_vertices = n;
  const std::size_t global_offset
      = MPI::global_offset(_mesh.mpi_comm(), num_new_vertices, true)
        + _mesh.num_entities_global(0);

  // If they are shared, then the new global vertex index needs to be
  // sent off-process.  Add offset to map, and collect up any shared
  // new vertices that need to send the new index off-process
  std::vector<std::vector<std::int64_t>> values_to_send(mpi_size);
  for (auto& local_edge : _local_edge_to_new_vertex)
  {
    // Add global_offset to map, to get new global index of new
    // vertices
    local_edge.second += global_offset;

    const std::size_t local_i = local_edge.first;
    // shared, but locally owned : remote owned are not in list.
    auto shared_edge_i = shared_edges.find(local_i);
    if (shared_edge_i != shared_edges.end())
    {
      for (auto remote_process : shared_edge_i->second)
      {
        // send mapping from global edge index to new global vertex index
        values_to_send[remote_process].push_back(
            global_edge_indices[local_edge.first]);
        values_to_send[remote_process].push_back(local_edge.second);
      }
    }
  }

  // Send new vertex indices to remote processes and receive
  std::vector<std::int64_t> received_values;
  MPI::all_to_all(_mesh.mpi_comm(), values_to_send, received_values);

  // Add received remote global vertex indices to map
  for (auto q = received_values.begin(); q != received_values.end(); q += 2)
  {
    const auto local_it = _global_to_local_edge_map.find(*q);
    assert(local_it != _global_to_local_edge_map.end());
    _local_edge_to_new_vertex[local_it->second] = *(q + 1);
  }

  // Attach global indices to each vertex, old and new, and sort
  // them across processes into this order

  std::vector<std::int64_t> global_indices(_mesh.topology().global_indices(0));
  for (std::size_t i = 0; i < num_new_vertices; i++)
    global_indices.push_back(i + global_offset);

  Eigen::Map<EigenRowArrayXXd> old_tmp(_new_vertex_coordinates.data(),
                                       _new_vertex_coordinates.size() / 3, 3);
  EigenRowArrayXXd tmp = mesh::DistributedMeshTools::reorder_by_global_indices(
      _mesh.mpi_comm(), old_tmp, global_indices);

  _new_vertex_coordinates
      = std::vector<double>(tmp.data(), tmp.data() + tmp.size());
}
//-----------------------------------------------------------------------------
mesh::Mesh ParallelRefinement::build_local() const
{
  const std::size_t tdim = _mesh.topology().dim();
  assert(_new_vertex_coordinates.size() % 3 == 0);
  const std::size_t num_vertices = _new_vertex_coordinates.size() / 3;

  const std::size_t num_cell_vertices = tdim + 1;
  assert(_new_cell_topology.size() % num_cell_vertices == 0);
  const std::size_t num_cells = _new_cell_topology.size() / num_cell_vertices;

  Eigen::Map<const EigenRowArrayXXd> geometry(_new_vertex_coordinates.data(),
                                              num_vertices, 3);
  Eigen::Map<const EigenRowArrayXXi64> topology(_new_cell_topology.data(),
                                                num_cells, num_cell_vertices);

  mesh::Mesh mesh(_mesh.mpi_comm(), _mesh.cell_type(), geometry, topology, {},
                  _mesh.get_ghost_mode());

  return mesh;
}
//-----------------------------------------------------------------------------
mesh::Mesh ParallelRefinement::partition(bool redistribute) const
{
  const int num_vertices_per_cell = mesh::cell_num_entities(_mesh.cell_type(), 0);

  // Copy data to mesh::LocalMeshData structures
  const std::int32_t num_local_cells
      = _new_cell_topology.size() / num_vertices_per_cell;
  std::vector<std::int64_t> global_cell_indices(num_local_cells);
  const std::size_t idx_global_offset
      = MPI::global_offset(_mesh.mpi_comm(), num_local_cells, true);
  for (std::int32_t i = 0; i < num_local_cells; i++)
    global_cell_indices[i] = idx_global_offset + i;

  Eigen::Map<const EigenRowArrayXXi64> cells(
      _new_cell_topology.data(), num_local_cells, num_vertices_per_cell);

  const std::size_t num_local_vertices = _new_vertex_coordinates.size() / 3;
  Eigen::Map<const EigenRowArrayXXd> points(_new_vertex_coordinates.data(),
                                            num_local_vertices, 3);

  if (redistribute)
  {
    return mesh::Partitioning::build_distributed_mesh(
        _mesh.mpi_comm(), _mesh.cell_type(), points, cells, global_cell_indices,
        _mesh.get_ghost_mode());
  }

  mesh::Mesh mesh(_mesh.mpi_comm(), _mesh.cell_type(), points, cells,
                  global_cell_indices, _mesh.get_ghost_mode());

  mesh::DistributedMeshTools::init_facet_cell_connections(mesh);

  return mesh;
}
//-----------------------------------------------------------------------------
void ParallelRefinement::new_cells(const std::vector<std::int64_t>& idx)
{
  _new_cell_topology.insert(_new_cell_topology.end(), idx.begin(), idx.end());
}
//-----------------------------------------------------------------------------
