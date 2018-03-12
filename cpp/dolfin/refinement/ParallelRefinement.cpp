// Copyright (C) 2013-2018 Chris Richardson
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <dolfin/common/MPI.h>
#include <dolfin/common/types.h>
#include <dolfin/mesh/DistributedMeshTools.h>
#include <dolfin/mesh/Edge.h>
#include <dolfin/mesh/LocalMeshData.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/mesh/MeshIterator.h>
#include <dolfin/mesh/MeshPartitioning.h>
#include <map>
#include <vector>

#include "ParallelRefinement.h"

using namespace dolfin;
using namespace dolfin::refinement;

//-----------------------------------------------------------------------------
ParallelRefinement::ParallelRefinement(const mesh::Mesh& mesh)
    : _mesh(mesh),
      shared_edges(
          mesh::DistributedMeshTools::compute_shared_entities(_mesh, 1)),
      marked_edges(mesh.num_entities(1), false),
      marked_for_update(MPI::size(mesh.mpi_comm()))
{
  // Do nothing
}
//-----------------------------------------------------------------------------
ParallelRefinement::~ParallelRefinement()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
bool ParallelRefinement::is_marked(std::int32_t edge_index) const
{
  dolfin_assert(edge_index < _mesh.num_entities(1));
  return marked_edges[edge_index];
}
//-----------------------------------------------------------------------------
void ParallelRefinement::mark(std::int32_t edge_index)
{
  dolfin_assert(edge_index < _mesh.num_entities(1));

  // Already marked, so nothing to do
  if (marked_edges[edge_index])
    return;

  marked_edges[edge_index] = true;
  auto map_it = shared_edges.find(edge_index);

  // If it is a shared edge, add all sharing procs to update set
  if (map_it != shared_edges.end())
  {
    for (auto const& it : map_it->second)
      marked_for_update[it.first].push_back(it.second);
  }
}
//-----------------------------------------------------------------------------
void ParallelRefinement::mark_all()
{
  marked_edges.assign(_mesh.num_entities(1), true);
}
//-----------------------------------------------------------------------------
const std::map<std::size_t, std::size_t>&
ParallelRefinement::edge_to_new_vertex() const
{
  return local_edge_to_new_vertex;
}
//-----------------------------------------------------------------------------
void ParallelRefinement::mark(const mesh::MeshEntity& entity)
{
  for (const auto& edge : mesh::EntityRange<mesh::Edge>(entity))
    mark(edge.index());
}
//-----------------------------------------------------------------------------
void ParallelRefinement::mark(const mesh::MeshFunction<bool>& refinement_marker)
{
  const std::size_t entity_dim = refinement_marker.dim();
  for (const auto& entity :
       mesh::MeshRange<mesh::MeshEntity>(_mesh, entity_dim))
  {
    if (refinement_marker[entity])
    {
      for (const auto& edge : mesh::EntityRange<mesh::Edge>(entity))
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
  for (const auto& edge : mesh::EntityRange<mesh::Edge>(cell))
  {
    if (marked_edges[edge.index()])
      result.push_back(i);
    ++i;
  }
  return result;
}
//-----------------------------------------------------------------------------
void ParallelRefinement::update_logical_edgefunction()
{
  const std::size_t mpi_size = MPI::size(_mesh.mpi_comm());

  // Send all shared edges marked for update and receive from other processes
  std::vector<std::size_t> received_values;
  MPI::all_to_all(_mesh.mpi_comm(), marked_for_update, received_values);

  // Clear marked_for_update vectors
  marked_for_update = std::vector<std::vector<std::size_t>>(mpi_size);

  // Flatten received values and set edges mesh::MeshFunction true at each index
  // received
  for (auto const& local_index : received_values)
    marked_edges[local_index] = true;
}
//-----------------------------------------------------------------------------
void ParallelRefinement::create_new_vertices()
{
  // Take marked_edges and use to create new vertices

  const std::size_t mpi_size = MPI::size(_mesh.mpi_comm());
  const std::size_t mpi_rank = MPI::rank(_mesh.mpi_comm());

  // Copy over existing mesh vertices
  new_vertex_coordinates = _mesh.geometry().x();

  // Tally up unshared marked edges, and shared marked edges which are
  // owned on this process.  Index them sequentially from zero.
  const std::size_t gdim = _mesh.geometry().dim();
  std::size_t n = 0;
  for (std::int32_t local_i = 0; local_i < _mesh.num_entities(1); ++local_i)
  {
    if (marked_edges[local_i] == true)
    {
      // Assume this edge is owned locally
      bool owner = true;

      // If shared, check that this is true
      auto shared_edge_i = shared_edges.find(local_i);
      if (shared_edge_i != shared_edges.end())
      {
        // check if any other sharing process has a lower rank
        for (auto const& proc_edge : shared_edge_i->second)
        {
          if (proc_edge.first < mpi_rank)
            owner = false;
        }
      }

      // If it is still believed to be owned on this process, add to
      // list
      if (owner)
      {
        const geometry::Point& midpoint = mesh::Edge(_mesh, local_i).midpoint();
        for (std::size_t j = 0; j < gdim; ++j)
          new_vertex_coordinates.push_back(midpoint[j]);
        local_edge_to_new_vertex[local_i] = n++;
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
  std::vector<std::vector<std::size_t>> values_to_send(mpi_size);
  for (auto& local_edge : local_edge_to_new_vertex)
  {
    // Add global_offset to map, to get new global index of new
    // vertices
    local_edge.second += global_offset;

    const std::size_t local_i = local_edge.first;
    // shared, but locally owned : remote owned are not in list.
    auto shared_edge_i = shared_edges.find(local_i);
    if (shared_edge_i != shared_edges.end())
    {
      for (auto const& remote_process_edge : shared_edges[local_i])
      {
        const std::size_t remote_proc_num = remote_process_edge.first;
        // send mapping from remote local edge index to new global vertex index
        values_to_send[remote_proc_num].push_back(remote_process_edge.second);
        values_to_send[remote_proc_num].push_back(local_edge.second);
      }
    }
  }

  // Send new vertex indices to remote processes and receive
  std::vector<std::size_t> received_values;
  MPI::all_to_all(_mesh.mpi_comm(), values_to_send, received_values);

  // Add received remote global vertex indices to map
  for (auto q = received_values.begin(); q != received_values.end(); q += 2)
    local_edge_to_new_vertex[*q] = *(q + 1);

  // Attach global indices to each vertex, old and new, and sort
  // them across processes into this order

  std::vector<std::int64_t> global_indices(_mesh.topology().global_indices(0));
  for (std::size_t i = 0; i < num_new_vertices; i++)
    global_indices.push_back(i + global_offset);

  mesh::DistributedMeshTools::reorder_values_by_global_indices(
      _mesh.mpi_comm(), new_vertex_coordinates, _mesh.geometry().dim(),
      global_indices);
}
//-----------------------------------------------------------------------------
mesh::Mesh ParallelRefinement::build_local() const
{
  const std::size_t tdim = _mesh.topology().dim();
  const std::size_t gdim = _mesh.geometry().dim();
  dolfin_assert(new_vertex_coordinates.size() % gdim == 0);
  const std::size_t num_vertices = new_vertex_coordinates.size() / gdim;

  const std::size_t num_cell_vertices = tdim + 1;
  dolfin_assert(new_cell_topology.size() % num_cell_vertices == 0);
  const std::size_t num_cells = new_cell_topology.size() / num_cell_vertices;

  Eigen::Map<const EigenRowArrayXXd> geometry(new_vertex_coordinates.data(),
                                              num_vertices, gdim);
  Eigen::Map<const EigenRowArrayXXi32> topology(new_cell_topology.data(),
                                                num_cells, num_cell_vertices);

  mesh::Mesh mesh(_mesh.mpi_comm(), _mesh.type().cell_type(), geometry,
                  topology);

  return mesh;
}
//-----------------------------------------------------------------------------
mesh::Mesh ParallelRefinement::partition(bool redistribute) const
{
  mesh::LocalMeshData mesh_data(_mesh.mpi_comm());
  mesh_data.topology.dim = _mesh.topology().dim();
  const std::size_t gdim = _mesh.geometry().dim();
  mesh_data.geometry.dim = gdim;

  mesh_data.topology.cell_type = _mesh.type().cell_type();
  mesh_data.topology.num_vertices_per_cell = mesh_data.topology.dim + 1;

  // Copy data to mesh::LocalMeshData structures
  const std::size_t num_local_cells
      = new_cell_topology.size() / mesh_data.topology.num_vertices_per_cell;
  mesh_data.topology.num_global_cells
      = MPI::sum(_mesh.mpi_comm(), num_local_cells);
  mesh_data.topology.global_cell_indices.resize(num_local_cells);
  const std::size_t idx_global_offset
      = MPI::global_offset(_mesh.mpi_comm(), num_local_cells, true);
  for (std::size_t i = 0; i < num_local_cells; i++)
    mesh_data.topology.global_cell_indices[i] = idx_global_offset + i;

  mesh_data.topology.cell_vertices.resize(
      boost::extents[num_local_cells]
                    [mesh_data.topology.num_vertices_per_cell]);
  std::copy(new_cell_topology.begin(), new_cell_topology.end(),
            mesh_data.topology.cell_vertices.data());

  const std::size_t num_local_vertices = new_vertex_coordinates.size() / gdim;
  mesh_data.geometry.num_global_vertices
      = MPI::sum(_mesh.mpi_comm(), num_local_vertices);
  mesh_data.geometry.vertex_coordinates.resize(
      boost::extents[num_local_vertices][gdim]);
  std::copy(new_vertex_coordinates.begin(), new_vertex_coordinates.end(),
            mesh_data.geometry.vertex_coordinates.data());

  mesh_data.geometry.vertex_indices.resize(num_local_vertices);
  const std::size_t vertex_global_offset
      = MPI::global_offset(_mesh.mpi_comm(), num_local_vertices, true);
  for (std::size_t i = 0; i < num_local_vertices; ++i)
    mesh_data.geometry.vertex_indices[i] = vertex_global_offset + i;

  if (!redistribute)
  {
    // FIXME: broken by ghost mesh?
    // Set owning process rank to this process rank
    mesh_data.topology.cell_partition.assign(
        mesh_data.topology.global_cell_indices.size(),
        MPI::rank(_mesh.mpi_comm()));
  }

  return mesh::MeshPartitioning::build_distributed_mesh(mesh_data,
                                                        _mesh.ghost_mode());
}
//-----------------------------------------------------------------------------
void ParallelRefinement::new_cells(const std::vector<std::size_t>& idx)
{
  new_cell_topology.insert(new_cell_topology.end(), idx.begin(), idx.end());
}
//-----------------------------------------------------------------------------
