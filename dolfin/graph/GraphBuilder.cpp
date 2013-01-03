// Copyright (C) 2010 Garth N. Wells
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
// Modified by Chris Richardson, 2012
//
// First added:  2010-02-19
// Last changed: 2012-12-07

#include <algorithm>
#include <numeric>
#include <set>
#include <vector>
#include <boost/unordered_set.hpp>
#include <boost/unordered_map.hpp>
#include <boost/functional/hash.hpp>

#include <boost/container/flat_map.hpp>

#include <dolfin/log/log.h>
#include <dolfin/common/MPI.h>
#include <dolfin/common/Timer.h>
#include <dolfin/common/types.h>
#include <dolfin/fem/GenericDofMap.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/LocalMeshData.h>
#include <dolfin/mesh/MeshEntityIterator.h>
#include <dolfin/mesh/Vertex.h>
#include "GraphBuilder.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
Graph GraphBuilder::local_graph(const Mesh& mesh, const GenericDofMap& dofmap0,
                                                  const GenericDofMap& dofmap1)
{
  // Create empty graph
  const std::size_t n = dofmap0.global_dimension();
  Graph graph(n);

  // Build graph
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    const std::vector<dolfin::la_index>& dofs0 = dofmap0.cell_dofs(cell->index());
    const std::vector<dolfin::la_index>& dofs1 = dofmap1.cell_dofs(cell->index());

    std::vector<dolfin::la_index>::const_iterator node;
    for (node = dofs0.begin(); node != dofs0.end(); ++node)
      graph[*node].insert(dofs1.begin(), dofs1.end());
  }

  return graph;
}
//-----------------------------------------------------------------------------
Graph GraphBuilder::local_graph(const Mesh& mesh,
                                const std::vector<std::size_t>& coloring_type)
{
  // Check coloring type
  dolfin_assert(coloring_type.size() >= 2);
  dolfin_assert(coloring_type.front() == coloring_type.back());

  // Create graph
  const std::size_t num_verticies = mesh.num_entities(coloring_type[0]);
  Graph graph(num_verticies);

  // Build graph
  for (MeshEntityIterator vertex_entity(mesh, coloring_type[0]); !vertex_entity.end(); ++vertex_entity)
  {
    boost::unordered_set<std::size_t> entity_list0;
    boost::unordered_set<std::size_t> entity_list1;
    entity_list0.insert(vertex_entity->index());

    // Build list of entities, moving between levels
    for (std::size_t level = 1; level < coloring_type.size(); ++level)
    {
      for (boost::unordered_set<std::size_t>::const_iterator entity_index
              = entity_list0.begin(); entity_index != entity_list0.end(); ++entity_index)
      {
        const MeshEntity entity(mesh, coloring_type[level -1], *entity_index);
        for (MeshEntityIterator neighbor(entity, coloring_type[level]); !neighbor.end(); ++neighbor)
          entity_list1.insert(neighbor->index());
      }
      entity_list0 = entity_list1;
      entity_list1.clear();
    }

    // Add edges to graph
    const std::size_t vertex_entity_index = vertex_entity->index();
    graph[vertex_entity_index].insert(entity_list0.begin(), entity_list0.end());
  }

  return graph;
}
//-----------------------------------------------------------------------------
Graph GraphBuilder::local_graph(const Mesh& mesh,
                                std::size_t dim0, std::size_t dim1)
{
  // Create graph
  const std::size_t num_verticies = mesh.num_entities(dim0);
  Graph graph(num_verticies);

  // Build graph
  for (MeshEntityIterator colored_entity(mesh, dim0); !colored_entity.end(); ++colored_entity)
  {
    const std::size_t colored_entity_index = colored_entity->index();
    for (MeshEntityIterator entity(*colored_entity, dim1); !entity.end(); ++entity)
    {
      for (MeshEntityIterator neighbor(*entity, dim0); !neighbor.end(); ++neighbor)
        graph[colored_entity_index].insert(neighbor->index());
    }
  }

  return graph;
}
//-----------------------------------------------------------------------------
BoostBidirectionalGraph GraphBuilder::local_boost_graph(const Mesh& mesh,
                                const std::vector<std::size_t>& coloring_type)
{
  // Check coloring type
  dolfin_assert(coloring_type.size() >= 2);
  dolfin_assert(coloring_type.front() == coloring_type.back());

  // Create graph
  const std::size_t num_verticies = mesh.num_entities(coloring_type[0]);
  BoostBidirectionalGraph graph(num_verticies);

  // Build graph
  for (MeshEntityIterator vertex_entity(mesh, coloring_type[0]); !vertex_entity.end(); ++vertex_entity)
  {
    boost::unordered_set<std::size_t> entity_list0;
    boost::unordered_set<std::size_t> entity_list1;
    entity_list0.insert(vertex_entity->index());

    // Build list of entities, moving between levels
    boost::unordered_set<std::size_t>::const_iterator entity_index;
    for (std::size_t level = 1; level < coloring_type.size(); ++level)
    {
      for (entity_index = entity_list0.begin(); entity_index != entity_list0.end(); ++entity_index)
      {
        const MeshEntity entity(mesh, coloring_type[level -1], *entity_index);
        for (MeshEntityIterator neighbor(entity, coloring_type[level]); !neighbor.end(); ++neighbor)
          entity_list1.insert(neighbor->index());
      }
      entity_list0 = entity_list1;
      entity_list1.clear();
    }

    // Add edges to graph
    const std::size_t vertex_entity_index = vertex_entity->index();
    for (entity_index = entity_list0.begin(); entity_index != entity_list0.end(); ++entity_index)
      boost::add_edge(vertex_entity_index, *entity_index, graph);
  }

  return graph;
}
//-----------------------------------------------------------------------------
BoostBidirectionalGraph GraphBuilder::local_boost_graph(const Mesh& mesh,
                                           std::size_t dim0, std::size_t dim1)
{
  // Create graph
  const std::size_t num_verticies = mesh.num_entities(dim0);
  BoostBidirectionalGraph graph(num_verticies);

  // Build graph
  for (MeshEntityIterator colored_entity(mesh, dim0); !colored_entity.end(); ++colored_entity)
  {
    const std::size_t colored_entity_index = colored_entity->index();
    for (MeshEntityIterator entity(*colored_entity, dim1); !entity.end(); ++entity)
    {
      for (MeshEntityIterator neighbor(*entity, dim0); !neighbor.end(); ++neighbor)
        boost::add_edge(colored_entity_index, neighbor->index(), graph);
    }
  }

  return graph;
}
//-----------------------------------------------------------------------------
void GraphBuilder::compute_dual_graph(const LocalMeshData& mesh_data,
                            std::vector<std::set<std::size_t> >& local_graph,
                            std::set<std::size_t>& ghost_vertices)
{
  FacetCellMap facet_cell_map;
  compute_local_dual_graph(mesh_data, local_graph, facet_cell_map);
  compute_nonlocal_dual_graph(mesh_data, local_graph, facet_cell_map, ghost_vertices);

  // Test with Boost hash
  {
    boost::hash<std::vector<std::set<std::size_t> > > uhash;
    const std::size_t local_hash = uhash(local_graph);
    std::vector<std::size_t> all_hashes;
    MPI::gather(local_hash, all_hashes);

    // Hash the received hash keys
    boost::hash<std::vector<size_t> > sizet_hash;
    std::size_t global_hash = sizet_hash(all_hashes);

    // Broadcast hash key
    MPI::broadcast(global_hash);
    if (MPI::process_number() == 0)
      cout << "Local graph hash (old, post parallel): " << global_hash << endl;
  }

  {
    boost::hash<std::set<std::size_t> > uhash;
    const std::size_t local_hash = uhash(ghost_vertices);
    std::vector<std::size_t> all_hashes;
    MPI::gather(local_hash, all_hashes);

    // Hash the received hash keys
    boost::hash<std::vector<size_t> > sizet_hash;
    std::size_t global_hash = sizet_hash(all_hashes);

    // Broadcast hash key
    MPI::broadcast(global_hash);
    if (MPI::process_number() == 0)
      cout << "Ghost graph hash (old): " << global_hash << endl;
  }

  // Testing
  local_graph.clear(); 
  ghost_vertices.clear();
  compute_dual_graph_test(mesh_data, local_graph, ghost_vertices);

  // Test with Boost hash
  {
    boost::hash<std::vector<std::set<std::size_t> > > uhash;
    const std::size_t local_hash = uhash(local_graph);
    std::vector<std::size_t> all_hashes;
    MPI::gather(local_hash, all_hashes);

    // Hash the received hash keys
    boost::hash<std::vector<size_t> > sizet_hash;
    std::size_t global_hash = sizet_hash(all_hashes);

    // Broadcast hash key
    MPI::broadcast(global_hash);
    if (MPI::process_number() == 0)
      cout << "Local graph hash (new, post parallel): " << global_hash << endl;
  }

  {
    boost::hash<std::set<std::size_t> > uhash;
    const std::size_t local_hash = uhash(ghost_vertices);
    std::vector<std::size_t> all_hashes;
    MPI::gather(local_hash, all_hashes);

    // Hash the received hash keys
    boost::hash<std::vector<size_t> > sizet_hash;
    std::size_t global_hash = sizet_hash(all_hashes);

    // Broadcast hash key
    MPI::broadcast(global_hash);
    if (MPI::process_number() == 0)
      cout << "Ghost graph hash (new): " << global_hash << endl;
  }
}
//-----------------------------------------------------------------------------
void GraphBuilder::compute_local_dual_graph(const LocalMeshData& mesh_data,
                            std::vector<std::set<std::size_t> >& local_graph, 
                            FacetCellMap& facet_cell_map)
{
  Timer timer("Compute dual graph");

  // List of cell vertices
  const boost::multi_array<std::size_t, 2>& cell_vertices = mesh_data.cell_vertices;
  const std::size_t num_local_cells = mesh_data.global_cell_indices.size();
  const std::size_t num_vertices_per_cell = mesh_data.num_vertices_per_cell;
  const std::size_t num_vertices_per_facet = num_vertices_per_cell - 1;

  dolfin_assert(num_local_cells == cell_vertices.shape()[0]);
  dolfin_assert(num_vertices_per_cell == cell_vertices.shape()[1]);

  local_graph.resize(num_local_cells);
  facet_cell_map.clear();

  // Compute local edges (cell-cell connections) using global (internal
  // to this function, not the user numbering) numbering

  // Get offset for this process
  const std::size_t cell_offset = MPI::global_offset(num_local_cells, true);

  // Create map from facet (list of vertex indices) to cells
  facet_cell_map.rehash((facet_cell_map.size() + num_local_cells)/facet_cell_map.max_load_factor() + 1);

  // Iterate over all cells
  std::vector<std::size_t> facet(num_vertices_per_facet);
  for (std::size_t i = 0; i < num_local_cells; ++i)
  {
    // Iterate over facets in cell
    for(std::size_t j = 0; j < num_vertices_per_cell; ++j)
    {
      // Build set of vertices that make up a facet (all cell vertices,
      // minus one cell vertex)
      std::size_t pos = 0;
      for (std::size_t k = 0; k < num_vertices_per_cell; ++k)
      {
        if (k != j)
          facet[pos++] = cell_vertices[i][k];
      }

      // Sort into order, so map indexing will be consistent
      std::sort(facet.begin(), facet.end());

      // Look for facet in map
      const FacetCellMap::const_iterator join_cell = facet_cell_map.find(facet);

      // If facet not found in map, insert facet->cell into map
      if(join_cell == facet_cell_map.end())
        facet_cell_map[facet] = i;
      else
      {
        // Already in map. Connect cells and delete facet from map
        local_graph[i].insert(join_cell->second + cell_offset);
        local_graph[join_cell->second].insert(i + cell_offset);
        facet_cell_map.erase(join_cell);
      }
    }
  }
}
//-----------------------------------------------------------------------------
void GraphBuilder::compute_nonlocal_dual_graph(const LocalMeshData& mesh_data,
                            std::vector<std::set<std::size_t> >& local_graph,
                            FacetCellMap& facet_cell_map,
                            std::set<std::size_t>& ghost_vertices)
{
  Timer timer("Compute dual graph");

  // At this stage facet_cell map only contains facets->cells with edge
  // facets either interprocess or external boundaries

  // List of cell vertices
  const boost::multi_array<std::size_t, 2>& cell_vertices = mesh_data.cell_vertices;
  const std::size_t num_local_cells = mesh_data.global_cell_indices.size();
  const std::size_t num_vertices_per_cell = mesh_data.num_vertices_per_cell;
  const std::size_t num_vertices_per_facet = num_vertices_per_cell - 1;

  dolfin_assert(num_local_cells == cell_vertices.shape()[0]);
  dolfin_assert(num_vertices_per_cell == cell_vertices.shape()[1]);

  // Compute local edges (cell-cell connections) using global (internal
  // to this function, not the user numbering) numbering

  double tt = time();

  // Get offset for this processe
  const std::size_t offset = MPI::global_offset(num_local_cells, true);

  // Copy to a new map and re-label cells by adding an offset
  boost::unordered_map<std::vector<std::size_t>, std::size_t> 
      othermap(facet_cell_map.begin(), facet_cell_map.end());
  for(boost::unordered_map<std::vector<std::size_t>, std::size_t>::iterator other_cell = othermap.begin();
        other_cell != othermap.end(); ++other_cell)
  {
    other_cell->second += offset;
  }

  const std::size_t num_processes = MPI::num_processes();
  const std::size_t process_number = MPI::process_number();

  // Clear ghost vertices
  ghost_vertices.clear();

  // Create MPI ring
  const int source = (num_processes + process_number - 1) % num_processes;
  const int dest   = (process_number + 1) % num_processes;

  // Repeat (n-1) times, to go round ring
  std::vector<std::size_t> comm_data_send;
  std::vector<std::size_t> comm_data_recv;

  for(std::size_t i = 0; i < (num_processes - 1); ++i)
  {
    // Pack data in std::vector to send
    comm_data_send.resize((num_vertices_per_facet + 1)*othermap.size());
    boost::unordered_map<std::vector<std::size_t>, std::size_t>::const_iterator it;
    std::size_t k = 0;
    for (it = othermap.begin(); it != othermap.end(); ++it)
    {
      for (std::size_t i = 0; i < num_vertices_per_facet; ++i)
        comm_data_send[k++] = (it->first)[i];
      comm_data_send[k++] = it->second;
    } 

    // Shift data to next process
    MPI::send_recv(comm_data_send, dest, comm_data_recv, source);

    // Unpack data
    std::vector<std::size_t> facet(num_vertices_per_facet);
    othermap.clear();
    othermap.rehash((othermap.size() + comm_data_recv.size()/(num_vertices_per_facet + 1))/othermap.max_load_factor() + 1);
    for (std::size_t i = 0; i < comm_data_recv.size(); i += (num_vertices_per_facet + 1))
    {
      std::size_t j = 0;
      for (j = 0; j < num_vertices_per_facet; ++j)
        facet[j] = comm_data_recv[i + j];
      othermap.insert(othermap.end(), std::make_pair(facet, comm_data_recv[i + j]));
    }

    //const std::size_t mapsize = MPI::sum(othermap.size());
    //if(process_number == 0)
    //{
    //  std::cout << "t = " << (time() - tt) << ", iteration: " << i
    //      << ", average map size = " << mapsize/num_processes << std::endl;
    //}

    // Go through local facets, looking for a matching facet in othermap
    FacetCellMap::iterator fcell;
    for (fcell = facet_cell_map.begin(); fcell != facet_cell_map.end(); ++fcell)
    {
      // Check if maps contains same facet
      boost::unordered_map<std::vector<std::size_t>, std::size_t>::iterator join_cell = othermap.find(fcell->first);
      if (join_cell != othermap.end())
      {
        // Found neighbours, insert into local_graph and delete facets
        // from both maps
        local_graph[fcell->second].insert(join_cell->second);
        ghost_vertices.insert(join_cell->second);
        facet_cell_map.erase(fcell);
        othermap.erase(join_cell);
      }
    }
  }

  tt = time() - tt;
  double tt_max = MPI::max(tt);
  if (process_number == 0)
    info("Time to build connectivity (parallel) map: %g", tt_max);

  // Remaining facets are exterior boundary - could be useful

  const std::size_t n_exterior_facets = MPI::sum(facet_cell_map.size());
  if (process_number == 0)
    std::cout << "n (exterior facets) = " << n_exterior_facets << std::endl;

}
//-----------------------------------------------------------------------------
void GraphBuilder::compute_dual_graph_test(const LocalMeshData& mesh_data,
                            std::vector<std::set<std::size_t> >& local_graph,
                            std::set<std::size_t>& ghost_vertices)
{
  Timer timer("Compute dual graph [experimental]");

  double tt = time();

  // Communicator
  MPICommunicator mpi_comm;
  boost::mpi::communicator comm(*mpi_comm, boost::mpi::comm_attach);

  const std::size_t num_processes = MPI::num_processes();
  const std::size_t process_number = MPI::process_number();

  // List of cell vertices
  const boost::multi_array<std::size_t, 2>& cell_vertices = mesh_data.cell_vertices;
  const std::size_t num_local_cells = mesh_data.global_cell_indices.size();
  const std::size_t num_vertices_per_cell = mesh_data.num_vertices_per_cell;

  dolfin_assert(num_local_cells == cell_vertices.shape()[0]);
  dolfin_assert(num_vertices_per_cell == cell_vertices.shape()[1]);

  local_graph.resize(num_local_cells);

  const std::size_t cell_offset = MPI::global_offset(num_local_cells, true);

  // Vertex-to-cell map type (must be ordered)    
  typedef std::multimap<std::size_t, std::size_t> OrderedVertexCellMultiMap;

  // Build vertex-to-cell map    
  OrderedVertexCellMultiMap meshdata_vertex_to_cell_map;
  for (std::size_t i = 0; i < num_local_cells; ++i)
  {
    // Iterate over cell vertices and add to map 
    for(std::size_t j = 0; j < num_vertices_per_cell; ++j)
      meshdata_vertex_to_cell_map.insert(std::make_pair(cell_vertices[i][j], i + cell_offset));  
  }

  // My local vertex range
  const std::pair<std::size_t, std::size_t> 
    my_vertex_range = MPI::local_range(mesh_data.num_global_vertices); 

  // Ownership ranges for vertices 
  std::vector<std::size_t> ownership;
  for (std::size_t p = 0; p < num_processes; ++p)
    ownership.push_back(MPI::local_range(p, mesh_data.num_global_vertices, num_processes).second);

  // Prepare to send vertex-cell maps to 'owner' process (relies on map
  // being ordered)
  std::vector<std::vector<std::size_t> > send_buffer(num_processes);
  OrderedVertexCellMultiMap::const_iterator vc;
  std::size_t p = 0;
  for (vc = meshdata_vertex_to_cell_map.begin(); vc != meshdata_vertex_to_cell_map.end(); ++vc)
  {
    while (vc->first >= ownership[p])
    {
      // FIXME: Could look here to figure out size of send buffer,
      //        and reserve memory in vectors
      ++p;
    }
    dolfin_assert(p < send_buffer.size()); 
    send_buffer[p].push_back(vc->first);
    send_buffer[p].push_back(vc->second);
  }

  // Receive buffer
  std::vector<std::vector<std::size_t> > recv_buffer(num_processes);

  // Communicate data
  boost::mpi::all_to_all(comm, send_buffer, recv_buffer);

  // Build local vertex-cell map
  std::vector<std::set<std::size_t> > 
      my_vertex_cell_map(my_vertex_range.second - my_vertex_range.first);
  for (std::size_t p = 0; p < num_processes; ++p) 
  {
    const std::vector<std::size_t>& recv_buffer_p = recv_buffer[p];
    for (std::size_t i = 0; i < recv_buffer_p.size(); i += 2) 
    {
      const std::size_t pos = recv_buffer_p[i] - my_vertex_range.first;
      dolfin_assert(pos < my_vertex_cell_map.size());
      my_vertex_cell_map[pos].insert(recv_buffer_p[i +1]);  
    }
  }

  // Build sorted set of my vertices  
  double tt2 = time();
  std::set<std::size_t> my_vertices(cell_vertices.data(), 
                                    cell_vertices.data() + cell_vertices.num_elements());

  // Request vertex-cell map for all vertices that I have from the owning
  // process
  p = 0;
  std::vector<std::vector<std::size_t> > required_vertices(num_processes);
  for (std::set<std::size_t>::const_iterator v = my_vertices.begin();
          v != my_vertices.end(); ++v) 
  {
    while (*v >= ownership[p])
    {
      //cout << "Increment proc: " << p << ", " << vc->first << endl;
      // FIXME: Could look here to figure out size of send buffer,
      //        and reserve memory in vectors
      ++p;
    }
    dolfin_assert(p < send_buffer.size()); 
    required_vertices[p].push_back(*v);
  }

  tt2 = time() - tt2;
  double tt_max2 = MPI::max(tt2);
  if (process_number == 0)
    info("Time to build set to request: %g", tt_max2);
  
  // Send request to procesess that own required vertices
  std::vector<std::vector<std::size_t> > vertices_to_send;
  boost::mpi::all_to_all(comm, required_vertices, vertices_to_send);
  dolfin_assert(vertices_to_send.size() == num_processes);

  // Build list of vertex-cell pairs to send back
  std::vector<std::vector<std::size_t> > remote_vertex_cell_map(num_processes);
  for (std::size_t p = 0; p < num_processes; ++p) 
  {
    const std::vector<std::size_t>& vertices_to_send_p = vertices_to_send[p];
    std::vector<std::size_t>& vertex_cell_map = remote_vertex_cell_map[p];
    for (std::size_t i = 0; i < vertices_to_send_p.size(); ++i) 
    {
      // Sanity check
      dolfin_assert(vertices_to_send_p[i] >= my_vertex_range.first 
          &&  vertices_to_send_p[i] < my_vertex_range.second);

      // Local vertex index
      const std::size_t my_vertex = vertices_to_send_p[i] - my_vertex_range.first;

      dolfin_assert(my_vertex < my_vertex_cell_map.size());
      const std::set<std::size_t>& cell_indices = my_vertex_cell_map[my_vertex]; 

      // Pack data (first number of cells, then cell indices)
      vertex_cell_map.push_back(cell_indices.size());
      vertex_cell_map.insert(vertex_cell_map.end(), cell_indices.begin(), cell_indices.end()); 
    }
  }

  // Send requested vertex-cell maps
  std::vector<std::vector<std::size_t> > local_vertex_cell_map_data;
  boost::mpi::all_to_all(comm, remote_vertex_cell_map, local_vertex_cell_map_data);
  dolfin_assert(local_vertex_cell_map_data.size() == num_processes);

  // Build required vertex-cell map (each vector will be sorted)
  std::vector<std::vector<std::size_t> > vertex_to_cell_map;
  for (std::size_t p = 0; p < num_processes; ++p)
  {
    const std::vector<std::size_t>& data = local_vertex_cell_map_data[p];
    std::size_t i = 0;
    while (i < data.size())
    {
      const std::size_t num_cells = data[i];  
      dolfin_assert(num_cells > 0);
      vertex_to_cell_map.push_back(std::vector<std::size_t>(&data[i+1], &data[i+1] + num_cells));
      i += (num_cells + 1);
    }  
  }

  // Now, the rest is local (no more communication) --------------------------

  const std::size_t num_vertices_per_facet = num_vertices_per_cell - 1;

  // Resize graph
  local_graph.resize(num_local_cells);
  ghost_vertices.clear();

  // Compute local edges (cell-cell connections) using global (internal
  // to this function, not the user numbering) numbering

  // My local cell range
  const std::pair<std::size_t, std::size_t> 
    my_cell_range = MPI::local_range(mesh_data.num_global_cells); 

  // Build renumvering ordering map (maps global to local)
  std::size_t count = 0;
  std::map<std::size_t, std::size_t> reorder;
  for (std::set<std::size_t>::const_iterator v = my_vertices.begin();
        v != my_vertices.end(); ++v)
  {
    reorder[*v] = count++;
  }

  // Renumber vertices to reduce look-ups later
  boost::multi_array<std::size_t, 2> cell_vertices_local = mesh_data.cell_vertices;
  std::size_t* _cell_vertices = cell_vertices_local.data();
  for (std::size_t i = 0; i < cell_vertices_local.num_elements(); ++i)
  {
    std::map<std::size_t, std::size_t>::const_iterator local_v = reorder.find(_cell_vertices[i]); 
    dolfin_assert(local_v != reorder.end());
    _cell_vertices[i] = local_v->second;
  }

  // Iterate over all cells
  std::vector<std::size_t> facet(num_vertices_per_facet);
  std::vector<std::size_t> intersection;
  for (std::size_t i = 0; i < num_local_cells; ++i)
  {
    // Iterate over facets in cell
    for(std::size_t j = 0; j < num_vertices_per_cell; ++j)
    {
      // Build set of vertices that make up a facet (all cell vertices,
      // minus one cell vertex)
      std::size_t pos = 0;
      for (std::size_t k = 0; k < num_vertices_per_cell; ++k)
      {
        if (k != j)
          facet[pos++] = cell_vertices_local[i][k];
      }

      // Sort facet vertices to use set_intersection later
      std::sort(facet.begin(), facet.end());

      // Find intersection of connected cells for each facet vertex  
      intersection = vertex_to_cell_map[facet[0]];
      std::vector<std::size_t>::iterator it = intersection.end();
      for (std::size_t i = 0; i < facet.size(); ++i)
      {
        dolfin_assert(facet[i] < vertex_to_cell_map.size());  
        std::vector<std::size_t>& cells = vertex_to_cell_map[facet[i]];

        // it points to end of constructed range
        it = std::set_intersection(intersection.begin(), it,
                                   cells.begin(), cells.end(),
                                   intersection.begin());         
      }

      // Numbers of cells in intersection
      std::size_t intersection_size = it - intersection.begin();

      // Should have 1 or 2 connections
      dolfin_assert(intersection_size > 0 && intersection_size < 3); 
      if (intersection_size == 2)
      {
        const std::size_t cell0 = *(intersection.begin());
        const std::size_t cell1 = *(intersection.begin() + 1);
        if (cell0 >= my_cell_range.first && cell0 < my_cell_range.second)
        {
          dolfin_assert((cell0 - my_cell_range.first) < local_graph.size());
          local_graph[cell0 - my_cell_range.first].insert(cell1);
        }
        else
          ghost_vertices.insert(cell0);
          
        if (cell1 >= my_cell_range.first && cell1 < my_cell_range.second)
        {
          dolfin_assert((cell1 - my_cell_range.first) < local_graph.size());
          local_graph[cell1 - my_cell_range.first].insert(cell0);
        }
        else
          ghost_vertices.insert(cell1);
      }
    }
  }

  // -----------------------------------------------------
  tt = time() - tt;
  double tt_max = MPI::max(tt);
  if (process_number == 0)
    info("Time to build connectivity map [experimental]: %g", tt_max);

}
//-----------------------------------------------------------------------------

