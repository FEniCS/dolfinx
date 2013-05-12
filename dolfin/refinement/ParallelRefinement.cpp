// Copyright (C) 2013 Chris Richardson
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
//
// First Added: 2013-01-02
// Last Changed: 2013-01-23

#include <vector>
#include <map>
#include <boost/unordered_map.hpp>
#include <boost/multi_array.hpp>

#include <dolfin/common/types.h>
#include <dolfin/common/MPI.h>
#include <dolfin/common/Timer.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/DistributedMeshTools.h>
#include <dolfin/mesh/Edge.h>
#include <dolfin/mesh/LocalMeshData.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshEntityIterator.h>
#include <dolfin/mesh/Vertex.h>

#include "ParallelRefinement.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
ParallelRefinement::ParallelRefinement(const Mesh& mesh) : _mesh(mesh),
  shared_edges(DistributedMeshTools::compute_shared_entities(_mesh, 1)),
  marked_edges(mesh.num_edges(), false)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
ParallelRefinement::~ParallelRefinement()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
bool ParallelRefinement::is_marked(std::size_t edge_index) const
{
  dolfin_assert(edge_index < _mesh.num_edges());
  return marked_edges[edge_index];
}
//-----------------------------------------------------------------------------
void ParallelRefinement::mark(std::size_t edge_index)
{
  dolfin_assert(edge_index < _mesh.num_edges());
  marked_edges[edge_index] = true;
}
//-----------------------------------------------------------------------------
void ParallelRefinement::mark_all()
{
  marked_edges.assign(_mesh.num_edges(), true);
}
//-----------------------------------------------------------------------------
const std::map<std::size_t, std::size_t>&
ParallelRefinement::edge_to_new_vertex() const
{
  return local_edge_to_new_vertex;
}
//-----------------------------------------------------------------------------
void ParallelRefinement::mark(const MeshEntity& cell)
{
  for (EdgeIterator edge(cell); !edge.end(); ++edge)
    marked_edges[edge->index()] = true;
}
//-----------------------------------------------------------------------------
void ParallelRefinement::mark(const MeshFunction<bool>& refinement_marker)
{
  for (MeshEntityIterator cell(_mesh, refinement_marker.dim()); !cell.end();
       ++cell)
  {
    if (refinement_marker[*cell])
    {
      for (EdgeIterator edge(*cell); !edge.end(); ++edge)
        marked_edges[edge->index()] = true;
    }
  }
}
//-----------------------------------------------------------------------------
std::size_t ParallelRefinement::marked_edge_count(const MeshEntity& cell) const
{
  std::size_t n_marked=0;
  for (EdgeIterator edge(cell); !edge.end(); ++edge)
  {
    if (marked_edges[edge->index()])
      n_marked++;
  }
  return n_marked;
}
//-----------------------------------------------------------------------------
void ParallelRefinement::update_logical_edgefunction()
{
  const std::size_t num_processes = MPI::num_processes();

  // Create a list of edges on this process that are 'true' and copy
  // to remote sharing processes
  std::vector<std::vector<std::size_t> > values_to_send(num_processes);
  std::vector<std::vector<std::size_t> > received_values;
  boost::unordered_map<unsigned int, std::vector<
    std::pair<unsigned int, unsigned int> > >::iterator sh_edge;
   for (sh_edge = shared_edges.begin(); sh_edge != shared_edges.end(); sh_edge++)
  {
    const std::size_t local_index = sh_edge->first;
    if (marked_edges[local_index] == true)
    {
      std::vector<std::pair<unsigned int, unsigned int> >::iterator proc_edge;
      for (proc_edge = sh_edge->second.begin();
          proc_edge != sh_edge->second.end(); ++proc_edge)
      {
        values_to_send[proc_edge->first].push_back(proc_edge->second);
      }
    }
  }

  MPI::all_to_all(values_to_send, received_values);

  // Flatten received values and set EdgeFunction true at each index
  // received
  std::vector<std::vector<std::size_t> >::iterator r;
  for (r = received_values.begin(); r != received_values.end(); ++r)
  {
    for (std::vector<std::size_t>::iterator local_index = r->begin();
         local_index != r->end(); ++local_index)
    {
      marked_edges[*local_index] = true;
    }
  }
}
//-----------------------------------------------------------------------------
void ParallelRefinement::create_new_vertices()
{
  // Take marked_edges and use to create new vertices

  const std::size_t num_processes = MPI::num_processes();
  const std::size_t process_number = MPI::process_number();

  // Tally up unshared marked edges, and shared marked edges which are
  // owned on this process.  Index them sequentially from zero.

  const std::size_t gdim = _mesh.geometry().dim();

  std::size_t n = 0;
  for (std::size_t local_i = 0 ; local_i < _mesh.num_edges(); ++local_i)
  {
    if(marked_edges[local_i] == true)
    {
      // Assume this edge is owned locally
      bool owner = true;

      // If shared, check that this is true
      if (shared_edges.count(local_i) != 0)
      {
        // check if any other sharing process has a lower rank
        std::vector<std::pair<unsigned int, unsigned int> >::iterator proc_edge;
        for (proc_edge = shared_edges.find(local_i)->second.begin();
              proc_edge != shared_edges.find(local_i)->second.end(); ++proc_edge)
        {
          if(proc_edge->first < process_number)
            owner = false;
        }
      }

      // If it is still believed to be owned on this process, add to
      // list
      if (owner)
      {
        const Point& midpoint = Edge(_mesh, local_i).midpoint();
        for(std::size_t j = 0; j < gdim; ++j)
          new_vertex_coordinates.push_back(midpoint[j]);

        local_edge_to_new_vertex[local_i] = n++;
      }
    }

  }

  // Calculate global range for new local vertices
  const std::size_t num_new_vertices = n;
  const std::size_t global_offset = MPI::global_offset(num_new_vertices, true)
                                  + _mesh.size_global(0);

  // If they are shared, then the new global vertex index needs to be
  // sent off-process.

  std::vector<std::vector<std::pair<std::size_t, std::size_t> > >
    values_to_send(num_processes);

  // Add offset to map, and collect up any shared new vertices that
  // need to send the new index off-process
  for (std::map<std::size_t, std::size_t>::iterator
       local_edge = local_edge_to_new_vertex.begin();
       local_edge != local_edge_to_new_vertex.end(); ++local_edge)
  {
    // Add global_offset to map, to get new global index of new
    // vertices
    local_edge->second += global_offset;

    const std::size_t local_i = local_edge->first;
    //shared, but locally owned : remote owned are not in list.
    if (shared_edges.count(local_i) != 0)
    {
      for (std::vector<std::pair<unsigned int, unsigned int> >::iterator remote_process_edge
             = shared_edges[local_i].begin();
          remote_process_edge != shared_edges[local_i].end();
          ++remote_process_edge)
      {
        const std::size_t remote_proc_num = remote_process_edge->first;
        // send mapping from remote local edge index to new global vertex index
        values_to_send[remote_proc_num].push_back(std::make_pair(remote_process_edge->second,
                                                                 local_edge->second));
      }
    }
  }

  // send new vertex indices to remote processes and receive
  std::vector<std::vector<std::pair<std::size_t, std::size_t> > >
    received_values(num_processes);

  MPI::all_to_all(values_to_send, received_values);

  // Flatten and add received remote global vertex indices to map
  std::vector<std::vector<std::pair<std::size_t, std::size_t> > >::iterator p;
  for (p = received_values.begin(); p != received_values.end(); ++p)
  {
    std::vector<std::pair<std::size_t, std::size_t> >::iterator q;
    for (q = p->begin(); q != p->end(); ++q)
      local_edge_to_new_vertex[q->first] = q->second;
  }

  //  std::cout << "Process:" << process_number << " "
  //    << num_new_vertices << " new vertices, "
  //            << "Offset = " << global_offset
  //            << std::endl;

  // Now add new vertex coordinates to existing, and index using new
  // global indexing.  Reorder so that MeshPartitioning.cpp can find
  // them. After that, we are done with coordinates, and just need to
  // rebuild the topology.

  new_vertex_coordinates.insert(new_vertex_coordinates.begin(),
                                _mesh.coordinates().begin(),
                                _mesh.coordinates().end());

  std::vector<std::size_t> global_indices(_mesh.topology().global_indices(0));
  for (std::size_t i = 0; i < num_new_vertices; i++)
    global_indices.push_back(i+global_offset);

  reorder_vertices_by_global_indices(new_vertex_coordinates,
                                     _mesh.geometry().dim(), global_indices);
}
//-----------------------------------------------------------------------------
void ParallelRefinement::reorder_vertices_by_global_indices(std::vector<double>& vertex_coords,
                                 const std::size_t gdim,
                                 const std::vector<std::size_t>& global_indices)
{
  // This is needed to interface with MeshPartitioning/LocalMeshData,
  // which expects the vertices in global order
  // This is inefficient, and needs to be addressed in MeshPartitioning.cpp
  // where they are redistributed again.

  Timer t("Parallel Refine: reorder vertices");
  // FIXME: be more efficient with MPI

  dolfin_assert(gdim*global_indices.size() == vertex_coords.size());

  boost::multi_array_ref<double, 2> vertex_array(vertex_coords.data(),
                      boost::extents[vertex_coords.size()/gdim][gdim]);

  // Calculate size of overall global vector by finding max index value
  // anywhere
  const std::size_t global_vector_size
    = MPI::max(*std::max_element(global_indices.begin(), global_indices.end())) + 1;

  // Send unwanted values off process
  const std::size_t num_processes = MPI::num_processes();
  std::vector<std::vector<std::pair<std::size_t, std::vector<double> > > >
    values_to_send(num_processes);

  // Go through local vector and append value to the appropriate list
  // to send to correct process
  for (std::size_t i = 0; i < vertex_array.shape()[0] ; ++i)
  {
    const std::size_t global_i = global_indices[i];
    const std::size_t process_i = MPI::index_owner(global_i, global_vector_size);
    const std::vector<double> v(vertex_array[i].begin(), vertex_array[i].end());
    values_to_send[process_i].push_back(std::make_pair(global_i, v));
  }

  // Redistribute the values to the appropriate process - including
  // self All values are "in the air" at this point, so local vector
  // can be cleared
  std::vector<std::vector<std::pair<std::size_t,std::vector<double> > > >
    received_values;
  MPI::all_to_all(values_to_send, received_values);

  // When receiving, just go through all received values
  // and place them in the local partition of the global vector.
  const std::pair<std::size_t, std::size_t> range
    = MPI::local_range(global_vector_size);
  vertex_coords.resize((range.second - range.first)*gdim);
  boost::multi_array_ref<double, 2> new_vertex_array(vertex_coords.data(),
                     boost::extents[range.second - range.first][gdim]);

  for (std::size_t i = 0; i < received_values.size(); ++i)
  {
    const std::vector<std::pair<std::size_t, std::vector<double> > >&
      received_global_data = received_values[i];
    for (std::size_t j = 0; j < received_global_data.size(); ++j)
    {
      const std::size_t global_i = received_global_data[j].first;
      dolfin_assert(global_i >= range.first && global_i < range.second);
      std::copy(received_global_data[j].second.begin(),
                received_global_data[j].second.end(),
                new_vertex_array[global_i - range.first].begin());
    }
  }
}
//-----------------------------------------------------------------------------
void ParallelRefinement::partition(Mesh& new_mesh, bool redistribute) const
{
  LocalMeshData mesh_data;
  mesh_data.tdim = _mesh.topology().dim();
  const std::size_t gdim = _mesh.geometry().dim();
  mesh_data.gdim = gdim;
  mesh_data.num_vertices_per_cell = mesh_data.tdim + 1;

  // Copy data to LocalMeshData structures

  const std::size_t num_local_cells = new_cell_topology.size()/mesh_data.num_vertices_per_cell;
  mesh_data.num_global_cells = MPI::sum(num_local_cells);
  mesh_data.global_cell_indices.resize(num_local_cells);
  const std::size_t idx_global_offset = MPI::global_offset(num_local_cells, true);
  for (std::size_t i = 0; i < num_local_cells ; i++)
    mesh_data.global_cell_indices[i] = idx_global_offset + i;

  mesh_data.cell_vertices.resize(boost::extents[num_local_cells][mesh_data.num_vertices_per_cell]);
  std::copy(new_cell_topology.begin(), new_cell_topology.end(),
            mesh_data.cell_vertices.data());

  const std::size_t num_local_vertices = new_vertex_coordinates.size()/gdim;
  mesh_data.num_global_vertices = MPI::sum(num_local_vertices);
  mesh_data.vertex_coordinates.resize(boost::extents[num_local_vertices][gdim]);
  std::copy(new_vertex_coordinates.begin(), new_vertex_coordinates.end(),
            mesh_data.vertex_coordinates.data());

  mesh_data.vertex_indices.resize(num_local_vertices);
  const std::size_t vertex_global_offset
    = MPI::global_offset(num_local_vertices, true);
  for (std::size_t i = 0; i < num_local_vertices ; i++)
    mesh_data.vertex_indices[i] = vertex_global_offset + i;

  if (!redistribute)
  {
    // Set owning process rank to this process rank
    mesh_data.cell_partition.assign(mesh_data.global_cell_indices.size(),
                                    MPI::process_number());
  }

  MeshPartitioning::build_distributed_mesh(new_mesh, mesh_data);
}
//-----------------------------------------------------------------------------
void ParallelRefinement::new_cell(const Cell& cell)
{
  for( VertexIterator v(cell); !v.end(); ++v)
    new_cell_topology.push_back(v->global_index());
}
//-----------------------------------------------------------------------------
void ParallelRefinement::new_cell(const std::size_t i0, const std::size_t i1,
                                  const std::size_t i2, const std::size_t i3)
{
  new_cell_topology.push_back(i0);
  new_cell_topology.push_back(i1);
  new_cell_topology.push_back(i2);
  new_cell_topology.push_back(i3);
}
//-----------------------------------------------------------------------------
void ParallelRefinement::new_cell(const std::size_t i0, const std::size_t i1,
                                  const std::size_t i2)
{
  new_cell_topology.push_back(i0);
  new_cell_topology.push_back(i1);
  new_cell_topology.push_back(i2);
}
//-----------------------------------------------------------------------------
