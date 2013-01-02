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
  Timer timer("Compute dual graph");

  // List of cell vertices
  const boost::multi_array<std::size_t, 2>& cell_vertices = mesh_data.cell_vertices;
  const std::size_t num_local_cells = mesh_data.global_cell_indices.size();
  const std::size_t num_vertices_per_cell = mesh_data.num_vertices_per_cell;
  const std::size_t num_vertices_per_facet = num_vertices_per_cell - 1;

  dolfin_assert(num_local_cells == cell_vertices.shape()[0]);
  dolfin_assert(num_vertices_per_cell == cell_vertices.shape()[1]);

  local_graph.resize(num_local_cells);

  // Compute local edges (cell-cell connections) using global (internal
  // to this function, not the user numbering) numbering

  double tt = time();

  // Get offset for this processe
  const std::size_t offset = MPI::global_offset(num_local_cells, true);

  // Create mapping from facets (list of vertex indices) to cells
  typedef boost::unordered_map<std::vector<std::size_t>, std::size_t> VectorMap;
  VectorMap facet_cell;
  facet_cell.rehash((facet_cell.size() + num_local_cells)/facet_cell.max_load_factor() + 1);

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

      const VectorMap::iterator join_cell = facet_cell.find(facet);

      // If facet not found in map, insert facet->cell into map
      if(join_cell == facet_cell.end())
        facet_cell[facet] = i;
      else
      {
        // Already in map. Connect cells and delete facet from map
        local_graph[i].insert(join_cell->second + offset);
        local_graph[join_cell->second].insert(i + offset);
        facet_cell.erase(join_cell);
      }
    }
  }

  //tt = time() - tt;
  //if (MPI::process_number() == 0)
  //  info("Time to build local connectivity map: %g", tt);

  tt = time();

  // Now facet_cell map only contains facets->cells with edge facets
  // either interprocess or external boundaries

  // FIXME: separate here into two functions

  // From this point relevant in parallel only (deals with ghosts edges)

  // Copy to a new map and re-label cells by adding an offset
  boost::unordered_map<std::vector<std::size_t>, std::size_t> othermap(facet_cell.begin(), facet_cell.end());
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
    VectorMap::iterator fcell;
    for (fcell = facet_cell.begin(); fcell != facet_cell.end(); ++fcell)
    {
      // Check if maps contains same facet
      boost::unordered_map<std::vector<std::size_t>, std::size_t>::iterator join_cell = othermap.find(fcell->first);
      if (join_cell != othermap.end())
      {
        // Found neighbours, insert into local_graph and delete facets
        // from both maps
        local_graph[fcell->second].insert(join_cell->second);
        ghost_vertices.insert(join_cell->second);
        facet_cell.erase(fcell);
        othermap.erase(join_cell);
      }
    }
  }

  tt = time() - tt;
  double tt_max = MPI::max(tt);
  if (process_number == 0)
    info("Time to build connectivity (parallel) map: %g", tt_max);

  // Remaining facets are exterior boundary - could be useful

  const std::size_t n_exterior_facets = MPI::sum(facet_cell.size());
  if (process_number == 0)
    std::cout << "n (exterior facets) = " << n_exterior_facets << std::endl;

  /*
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
    if (process_number == 0)
      cout << "Local graph hash (post parallel): " << global_hash << endl;
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
    if (process_number == 0)
      cout << "Ghost graph hash: " << global_hash << endl;
  }
  */
}
//-----------------------------------------------------------------------------
void GraphBuilder::compute_dual_graph_test(const LocalMeshData& mesh_data,
                            std::vector<std::set<std::size_t> >& local_graph,
                            std::set<std::size_t>& ghost_vertices)
{
  Timer timer("Compute dual graph");

  const std::size_t num_processes = MPI::num_processes();
  const std::size_t process_number = MPI::process_number();

  // List of cell vertices
  const boost::multi_array<std::size_t, 2>& cell_vertices = mesh_data.cell_vertices;
  const std::size_t num_local_cells = mesh_data.global_cell_indices.size();
  const std::size_t num_vertices_per_cell = mesh_data.num_vertices_per_cell;

  dolfin_assert(num_local_cells == cell_vertices.shape()[0]);
  dolfin_assert(num_vertices_per_cell == cell_vertices.shape()[1]);

  local_graph.resize(num_local_cells);

  const std::size_t offset = MPI::global_offset(num_local_cells, true);

  // Build vertex local-to-global map    
  std::map<std::size_t, std::size_t> vertex_global_to_local_map; 
  std::size_t k = 0;
  for (std::size_t i = 0; i < num_local_cells; ++i)
  {
    for(std::size_t j = 0; j < num_vertices_per_cell; ++j)
    {
      if (vertex_global_to_local_map.insert(std::make_pair(cell_vertices[i][j], k)).second)
        k++;
    }
  }
  std::map<std::size_t, std::size_t>::const_iterator tmp_it; 
  std::size_t max_k = 0;
  for (tmp_it = vertex_global_to_local_map.begin(); tmp_it != vertex_global_to_local_map.end(); ++tmp_it)
    max_k = std::max(max_k, tmp_it->second);
  dolfin_assert((max_k + 1) == vertex_global_to_local_map.size());


  // Build vertex-to-cell map    
  std::multimap<std::size_t, std::size_t> vertex_to_cell_map;
  for (std::size_t i = 0; i < num_local_cells; ++i)
  {
    // Iterate over cell vertices and add tp map 
    for(std::size_t j = 0; j < num_vertices_per_cell; ++j)
      vertex_to_cell_map.insert(std::make_pair(cell_vertices[i][j], i + offset));  
  }

  // Ownership range for vertices 
  std::vector<std::size_t> ownership;
  for (std::size_t p = 0; p < num_processes; ++p)
    ownership.push_back(MPI::local_range(p, mesh_data.num_global_vertices, num_processes).second);

  // Prepare to send vertex-cell maps to 'owner'
  std::vector<std::vector<std::size_t> > send_buffer(num_processes);
  std::multimap<std::size_t, std::size_t>::const_iterator vc;
  std::size_t p = 0;
  for (vc = vertex_to_cell_map.begin(); vc != vertex_to_cell_map.end(); ++vc)
  {
    if (vc->first < ownership[p])
      ++p;
    dolfin_assert(p < send_buffer.size()); 
    send_buffer[p].push_back(vc->first);
    send_buffer[p].push_back(vc->second);
  }

  std::vector<std::vector<std::size_t> > recv_buffer;
  MPI::distribute_vector(send_buffer, recv_buffer);

  //  static void distribute(const std::vector<T>& in_values,
  //                         const std::vector<S>& destinations,
  //                         std::vector<T>& out_values)


  // -------

  // Compute local edges (cell-cell connections) using global (internal) numbering

  // -----------

  // Build vertex local-to-global 
  //std::map<std::size_t, std::size_t>::const_iterator it; 
  //std::vector<std::size_t> tmp_vertex_local_to_global(tmp_vertex_global_to_local.size());
  //for (it = tmp_vertex_global_to_local.begin(); it != tmp_vertex_global_to_local.end(); ++it)
  //{
  //  dolfin_assert(it->second < tmp_vertex_local_to_global.size());
  //  tmp_vertex_local_to_global[it->second] = it->first;
  //}

  std::pair<std::size_t, std::size_t> local_range = MPI::local_range(mesh_data.num_global_vertices); 
  cout << "Test offset/range: " << local_range.first << ", " << local_range.second << endl;

  // ---------


  double tt = time();

  // Create mapping from facets (list of vertex indices) to cells
  typedef boost::unordered_map<std::vector<std::size_t>, std::size_t> VectorMap;
  VectorMap facet_cell;

  // Iterate over all cells
  std::vector<std::size_t> facet(num_vertices_per_cell - 1);
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

      const VectorMap::iterator join_cell = facet_cell.find(facet);

      // If facet not found in map, insert facet->cell into map
      if(join_cell == facet_cell.end())
        facet_cell[facet] = i;
      else
      {
        // Already in map. Connect cells and delete from map.
        local_graph[i].insert(join_cell->second + offset);
        local_graph[join_cell->second].insert(i + offset);
        facet_cell.erase(join_cell);
      }
    }
  }

  std::cout << "test time = " << (time() - tt) << std::endl;

  // Now facet_cell map only contains facets->cells with edge facets
  // either interprocess or external boundaries

  // FIXME: separate here into two functions

  // From this point relevant in parallel only (deals with ghosts edges)

  tt = time();

  // Copy to a new map and re-label cells by adding an offset
  std::map<std::vector<std::size_t>, std::size_t> othermap(facet_cell.begin(), facet_cell.end());
  for(std::map<std::vector<std::size_t>, std::size_t>::iterator other_cell = othermap.begin();
        other_cell != othermap.end(); ++other_cell)
  {
    other_cell->second += offset;
  }

  // Clear ghost vertices
  ghost_vertices.clear();

  // Create MPI ring
  const int source = (num_processes + process_number - 1) % num_processes;
  const int dest   = (process_number + 1) % num_processes;

  // FIXME: Find better way to send map data structures via MPI

  // Repeat (n-1) times, to go round ring
  std::vector<std::pair<std::vector<std::size_t>, std::size_t> > comm_data_send;
  std::vector<std::pair<std::vector<std::size_t>, std::size_t> > comm_data_recv;

  for(std::size_t i = 0; i < (num_processes - 1); ++i)
  {
    // FIXME: Improve memory management. Can the maps be serialised/sent directly?

    // Pack data to send
    if(process_number == 0)
      cout << "Map size to send, send to: " << othermap.size() << ", " << dest << endl;
    comm_data_send.resize(othermap.size());
    std::copy(othermap.begin(), othermap.end(), comm_data_send.begin());

    // Shift data to next process
    MPI::send_recv(comm_data_send, dest, comm_data_recv, source);

    // Unpack data
    othermap.clear();
    othermap.insert(comm_data_recv.begin(), comm_data_recv.end());
    //cout << "Map size received, received from: " << othermap.size() << ", " << source << endl;

    const std::size_t mapsize = MPI::sum(othermap.size());
    if(process_number == 0)
    {
      std::cout << "t = " << (time() - tt) << ", iteration: " << i
          << ", average map size = " << mapsize/num_processes << std::endl;
    }

    // Go through local facets, looking for a matching facet in othermap
    VectorMap::iterator fcell;
    for (fcell = facet_cell.begin(); fcell != facet_cell.end(); ++fcell)
    {
      // Check if maps contains same facet
      std::map<std::vector<std::size_t>, std::size_t>::iterator join_cell = othermap.find(fcell->first);
      if (join_cell != othermap.end())
      {
        // Found neighbours, insert into local_graph and delete facets
        // from both maps
        local_graph[fcell->second].insert(join_cell->second);
        ghost_vertices.insert(join_cell->second);
        facet_cell.erase(fcell);
        othermap.erase(join_cell);
      }
    }
    if(process_number == 0)
      cout << "New map size (1), send to: " << othermap.size() << ", " << dest << endl;
  }

  // Remaining facets are exterior boundary - could be useful

  const std::size_t n_exterior_facets = MPI::sum(facet_cell.size());
  if (process_number == 0)
    std::cout << "n (exterior facets) = " << n_exterior_facets << std::endl;

  tt = time() - tt;
  if (process_number == 0)
    info("Time to build connectivity map [new]: %g", tt);
}
//-----------------------------------------------------------------------------

