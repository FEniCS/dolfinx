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
// Last Changed: 2013-01-10

#include <vector>
#include <map>
#include <boost/unordered_map.hpp>
#include <boost/multi_array.hpp>

#include <dolfin/common/types.h>
#include <dolfin/common/MPI.h>
#include <dolfin/common/Timer.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Edge.h>
#include <dolfin/mesh/Vertex.h>
#include <dolfin/mesh/LocalMeshData.h>

#include "ParallelRefinement.h"

using namespace dolfin;

ParallelRefinement::ParallelRefinement(const Mesh& mesh):_mesh(mesh)
{
  // work out which edges are shared between processes
  get_shared_edges();
  // maybe take over marked edge assignment - experimenting with this
  marked_edges.assign(_mesh.num_edges(), false);
}
//-----------------------------------------------------------------------------

ParallelRefinement::~ParallelRefinement()
{
  // do nothing
}

//-----------------------------------------------------------------------------

std::map<std::size_t, std::size_t>& ParallelRefinement::edge_to_new_vertex()
{
  return local_edge_to_new_vertex;
}
//-----------------------------------------------------------------------------

std::vector<double>& ParallelRefinement::vertex_coordinates()
{
  return new_vertex_coordinates;
}

//-----------------------------------------------------------------------------

double* ParallelRefinement::vertex_coordinates(std::size_t i)
{
  return new_vertex_coordinates.data() + i*_mesh.geometry().dim();
}

//-----------------------------------------------------------------------------

void ParallelRefinement::get_shared_edges()
{

  const std::size_t edge_dim = 1;

  std::cout << "Calling compute shared entities" << std::endl;
  
  _shared_edges = MeshDistributed::compute_shared_entities(_mesh, edge_dim);
  
  std::cout << "n(shared_edges) = " << _shared_edges.size() << std::endl;

  // uint D = _mesh.topology().dim();

  // // Work out shared edges, and which processes they exist on.
  // // There are special cases where it is more difficult to determine
  // // edge ownership, but these are rare in 2D. 
  // // In 3D, it is always necessary to communicate with MPI
  // // to check the ownership. 
  // // Ultimately, this functionality will be provided inside 
  // // MeshConnectivity or similar
  
  // const std::map<std::size_t, std::set<std::size_t> >& shared_vertices = _mesh.topology().shared_entities(0);

  // // Go through all edges, looking for possible shared edges.
  // // In 2D, it is possible to use the difference between num_global_entities
  // // and num_entities, but this is not working (yet) in 3D.


  // for(EdgeIterator edge(_mesh); !edge.end(); ++edge)
  // {
  //   if(edge->num_entities(D) < edge->num_global_entities(D)
  //      || D == 3 )
  //   {
  //     // Find sharing processes by taking the intersection of the sets of 
  //     // processes of the two attached vertices.
  //     // That does not provide a definitive answer, but it is a start.
  //     VertexIterator v(*edge);
  //     // First, make sure the edge has two shared vertices.
  //     if(shared_vertices.count(v[0].global_index()) != 0 
  //        && shared_vertices.count(v[1].global_index()) != 0)
  //     {
  //       const std::set<std::size_t>& set1 
  //         = shared_vertices.find(v[0].global_index())->second;
  //       const std::set<std::size_t>& set2 
  //         = shared_vertices.find(v[1].global_index())->second;
      
  //       std::vector<std::size_t> result(set1.size() + set2.size());
  //       std::size_t nprocs = std::set_intersection(set1.begin(), set1.end(), 
  //                                                  set2.begin(), set2.end(), 
  //                                                  result.begin()) - result.begin();
        
  //       if(nprocs > 0)
  //       {
  //         // this is probably a shared edge - add an entry to the global->local map
  //         _global_to_local.insert(std::make_pair(edge->global_index(),edge->index()));
  //         std::set<std::size_t> resultant_set(result.data(),result.data() + nprocs);
  //         //          _shared_edges.insert(std::make_pair(edge->global_index(), resultant_set ));
  //       }
  //     }
  //   }
    
  // }
  
  // // Tell remote processes that this process has these shared edges.
  // // When receiving, ignore any edges that this process does not share.

  // std::size_t num_processes = MPI::num_processes();
  // std::vector<uint>destinations(num_processes);
  // std::vector<uint>sources(num_processes);
  // std::vector<std::vector<std::size_t> > values_to_send(num_processes);
  // std::vector<std::vector<std::size_t> > received_values(num_processes);

  // for(std::size_t i = 0; i < num_processes; ++i)
  //   destinations[i] = i;

  // // send a list of global_edge indices to remote processes that probably share with this process
  // for(boost::unordered_map<std::size_t, std::set<std::size_t> >::iterator s = _shared_edges.begin();
  //     s != _shared_edges.end(); ++s)
  // {
  //   for(std::set<std::size_t>::iterator p = s->second.begin(); p != s->second.end(); ++p)
  //     values_to_send[*p].push_back(s->first);
  // }

  // MPI::distribute(values_to_send, destinations, received_values, sources);  

  // // Copy map of probable shared edges, to compare with received shared edges
  // boost::unordered_map<std::size_t, std::set<std::size_t> > original_shared_edges(_shared_edges);
  // _shared_edges.clear();

  // for(std::size_t i = 0; i < sources.size(); ++i)
  // {
  //   std::size_t process = sources[i];
  //   for(std::vector<std::size_t>::iterator recv_edge = received_values[i].begin(); 
  //       recv_edge != received_values[i].end(); ++recv_edge)
  //   {
  //     // only add back if both this process and the remote process believe it is shared
  //     if(original_shared_edges.count(*recv_edge) != 0)
  //       _shared_edges[*recv_edge].insert(process);
  //   }
  // }

  // std::cout << "n(shared_edges) = " << _shared_edges.size() << std::endl;

}

//-----------------------------------------------------------------------------

void ParallelRefinement::mark_edge(std::size_t edge_index)
{
  dolfin_assert(marked_edges.size() == _mesh.num_edges());
  marked_edges[edge_index] = true;
}

//-----------------------------------------------------------------------------

// logical "or" of edgefunction on boundaries
void ParallelRefinement::update_logical_edgefunction(EdgeFunction<bool>& values)
{
  Timer t("update logical edgefunction");
  
  uint num_processes = MPI::num_processes();

  // Create a list of edges on this process that are 'true' and copy to remote sharing processes
  std::vector<uint>destinations(num_processes);
  std::vector<std::vector<std::size_t> > values_to_send(num_processes);
  std::vector<std::vector<std::size_t> > received_values;

  for(uint i = 0; i < num_processes; ++i)
    destinations[i] = i;

  for(boost::unordered_map<std::size_t, std::vector<std::pair<std::size_t, std::size_t> > >::iterator sh_edge = _shared_edges.begin();
      sh_edge != _shared_edges.end(); sh_edge++)
  {
    const std::size_t local_index = sh_edge->first;

    if(values[local_index] == true)
    {
      for(std::vector<std::pair<std::size_t, std::size_t> >::iterator proc_edge = sh_edge->second.begin(); 
          proc_edge != sh_edge->second.end(); ++proc_edge)
      {
        const std::size_t proc_num = proc_edge->first;
        values_to_send[proc_num].push_back(proc_edge->second);
      }
      
    }
  }
  
  MPI::distribute(values_to_send, destinations, received_values);

  // Flatten received values and set EdgeFunction true at each index received
  for(std::vector<std::vector<std::size_t> >::iterator r=received_values.begin();
      r != received_values.end(); ++r)
    for(std::vector<std::size_t>::iterator local_index = r->begin(); 
        local_index != r->end(); ++local_index)
    {
      values[*local_index] = true;
    }
  
}

//-----------------------------------------------------------------------------

void ParallelRefinement::create_new_vertices(const EdgeFunction<bool>& markedEdges)
{
  // Take markedEdges and use to create new vertices
  
  const std::size_t num_processes = MPI::num_processes();
  const std::size_t process_number = MPI::process_number();

  // Tally up unshared marked edges, and shared marked edges which are owned on this process.
  // Index them sequentially from zero.

  const std::size_t gdim = _mesh.geometry().dim();

  std::size_t n=0;
  for(EdgeIterator edge(_mesh); !edge.end(); ++edge)
  {
    if(markedEdges[*edge])
    {
      const std::size_t local_i = edge->index();

      if(_shared_edges.count(local_i) == 0) //local new vertex - not shared
      {
        const Point& midpoint = edge->midpoint();
        for(uint j = 0; j < gdim; ++j)
        {
          new_vertex_coordinates.push_back(midpoint[j]);
        }

        local_edge_to_new_vertex[local_i] = n++;
      }
      else 
      {
        bool owner = true;

        // check if any other sharing process has a lower rank
        for(std::vector<std::pair<std::size_t, std::size_t> >::iterator proc_edge 
              = _shared_edges.find(local_i)->second.begin();
            proc_edge != _shared_edges.find(local_i)->second.end(); ++proc_edge)
        {
          const std::size_t proc_num = proc_edge->first;
          if(proc_num < process_number)
            owner = false;
        }

        if(owner)
        { //local new vertex to be shared with another process
          const Point& midpoint = edge->midpoint();
          for(uint j = 0; j < gdim; ++j)
          {
            new_vertex_coordinates.push_back(midpoint[j]);
          }    
     
          local_edge_to_new_vertex[local_i] = n++;
        } 
      }
      // else new vertex is remotely owned, and we will be informed about it later
    }
  }

  // Calculate global range for new local vertices
  const std::size_t num_new_vertices = n;
  const std::size_t global_offset = MPI::global_offset(num_new_vertices, true) 
                                  + _mesh.size_global(0);

  // If they are shared, then the new global vertex index needs to be sent off-process.

  std::vector<std::vector<std::pair<std::size_t, std::size_t> > > values_to_send(num_processes);
  std::vector<std::size_t> destinations(num_processes);
  // Set up destination vector for communication with remote processes
  for(std::size_t process_j = 0; process_j < num_processes ; ++process_j)
    destinations[process_j] = process_j;

  // Add offset to map, and collect up any shared new vertices that need to send the new index off-process
  for(std::map<std::size_t, std::size_t>::iterator local_edge = local_edge_to_new_vertex.begin();
      local_edge != local_edge_to_new_vertex.end(); ++local_edge)
  {
    // Add global_offset to map, to get new global index of new vertices
    local_edge->second += global_offset; 

    const std::size_t local_i = local_edge->first;
    if(_shared_edges.count(local_i) != 0) //shared, but locally owned. 
    {
      for(std::vector<std::pair<std::size_t, std::size_t> >::iterator remote_process_edge 
            = _shared_edges[local_i].begin(); 
          remote_process_edge != _shared_edges[local_i].end();
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
  std::vector<std::vector<std::pair<std::size_t,std::size_t> > > received_values;
  MPI::distribute(values_to_send, destinations, received_values);

  // Flatten and add received remote global vertex indices to map 
  for(std::vector<std::vector<std::pair<std::size_t, std::size_t> > >::iterator p = received_values.begin();
      p != received_values.end(); ++p)
    for(std::vector<std::pair<std::size_t, std::size_t> >::iterator q = p->begin(); q != p->end(); ++q)
    {
      local_edge_to_new_vertex[q->first] = q->second;
    }
  
  std::cout << "Process:" << process_number << " " << num_new_vertices << " new vertices, "
            << "Offset = " << global_offset
            << std::endl;

  // Now add new vertex coordinates to existing, and index using new global indexing.
  // Reorder so that MeshPartitioning.cpp can find them. After that, we are done with 
  // coordinates, and just need to rebuild the topology.
  
  new_vertex_coordinates.insert(new_vertex_coordinates.begin(),
                                _mesh.coordinates().begin(),
                                _mesh.coordinates().end());

  std::vector<std::size_t> global_indices(_mesh.topology().global_indices(0));
  for(uint i=0; i < num_new_vertices; i++)
  {
    global_indices.push_back(i+global_offset);
  }
  reorder_vertices_by_global_indices(new_vertex_coordinates, _mesh.geometry().dim(), global_indices);
  
  std::cout << "vertices= " << new_vertex_coordinates.size() << std::endl;

}


//-----------------------------------------------------------------------------

// This is needed to interface with MeshPartitioning/LocalMeshData, 
// which expects the vertices in global order 
// This is inefficient, and needs to be addressed in MeshPartitioning.cpp
// where they are redistributed again.

void ParallelRefinement::reorder_vertices_by_global_indices(std::vector<double>& vertex_coords,
                                                            const std::size_t gdim,
                                                            const std::vector<std::size_t>& global_indices)
{
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
  std::vector<std::vector<std::pair<std::size_t, std::vector<double> > > > values_to_send(num_processes);
  std::vector<std::size_t> destinations(num_processes);

  // Set up destination vector for communication with remote processes
  for(std::size_t process_j = 0; process_j < num_processes ; ++process_j)
    destinations[process_j] = process_j;

  // Go through local vector and append value to the appropriate list
  // to send to correct process
  for(std::size_t i = 0; i < vertex_array.shape()[0] ; ++i)
  {
    const std::size_t global_i = global_indices[i];
    const std::size_t process_i = MPI::index_owner(global_i, global_vector_size);
    const std::vector<double> v(vertex_array[i].begin(), vertex_array[i].end());
    values_to_send[process_i].push_back(std::make_pair(global_i, v));
  }

  // Redistribute the values to the appropriate process - including self
  // All values are "in the air" at this point, so local vector can be cleared
  std::vector<std::vector<std::pair<std::size_t,std::vector<double> > > > received_values;
  MPI::distribute(values_to_send, destinations, received_values);

  // When receiving, just go through all received values
  // and place them in the local partition of the global vector.
  const std::pair<std::size_t, std::size_t> range = MPI::local_range(global_vector_size);
  vertex_coords.resize((range.second - range.first)*gdim);
  boost::multi_array_ref<double, 2> new_vertex_array(vertex_coords.data(),
                     boost::extents[range.second - range.first][gdim]);

  for(std::size_t i = 0; i < received_values.size(); ++i)
  {
    const std::vector<std::pair<std::size_t, std::vector<double> > >& received_global_data = received_values[i];
    for(std::size_t j = 0; j < received_global_data.size(); ++j)
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

void ParallelRefinement::new_cell(std::size_t i0, std::size_t i1, std::size_t i2, std::size_t i3)
{
  new_cell_topology.push_back(i0);
  new_cell_topology.push_back(i1);
  new_cell_topology.push_back(i2);
  new_cell_topology.push_back(i3);
}

//-----------------------------------------------------------------------------

void ParallelRefinement::new_cell(std::size_t i0, std::size_t i1, std::size_t i2)
{
  new_cell_topology.push_back(i0);
  new_cell_topology.push_back(i1);
  new_cell_topology.push_back(i2);
}

//-----------------------------------------------------------------------------


std::vector<std::size_t>& ParallelRefinement::cell_topology()
{
  return new_cell_topology;
}

