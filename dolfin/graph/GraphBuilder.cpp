// Copyright (C) 2010-2013 Garth N. Wells
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
// Last changed: 2013-01-31

#include <algorithm>
#include <numeric>
#include <set>
#include <vector>
#include <boost/unordered_map.hpp>
#include <boost/unordered_set.hpp>

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
void GraphBuilder::compute_dual_graph_orig(const LocalMeshData& mesh_data,
                                           std::vector<std::set<std::size_t> >& local_graph,
                                           std::set<std::size_t>& ghost_vertices)
{

  // This function builds the local part of a distributed dual graph
  // by partitioning cell vertices evenly across processes in ascending
  // vertex index order. The 'owner' process holds a
  //
  //  vertex -> [connected cells]
  //
  // map. Each process then requests the map for each vertex that it
  // requires. Since the vertex-cell map is ordered and distributed
  // by index, a process knows which other process owns the required
  // vertex-cells map.
  //
  // Once a processes has requested and received the necessary
  // vertex-cells maps, it builds it's local part of the distributed
  // dual graph without further communication.


  // TODO: 1. Free up memory by clearing large data structures when not
  //          longer used.
  //       2. Look for efficiency gains
  //       3. Check if vectors can be used in instead of maps or sets in
  //          places. Maps and sets can be expensive to destroy.

  #ifdef HAS_MPI

  // Vertex-to-cell map type (must be ordered)
  typedef std::multimap<std::size_t, std::size_t> OrderedVertexCellMultiMap;

  Timer timer("Compute dual graph [experimental]");

  // MPI communicator
  MPICommunicator mpi_comm;
  boost::mpi::communicator comm(*mpi_comm, boost::mpi::comm_attach);

  // Number of MPI processes
  const std::size_t num_processes = MPI::num_processes();

  // List of cell vertices
  const boost::multi_array<std::size_t, 2>& cell_vertices = mesh_data.cell_vertices;
  const std::size_t num_local_cells = mesh_data.global_cell_indices.size();
  const std::size_t num_vertices_per_cell = mesh_data.num_vertices_per_cell;

  // Sanity checks
  dolfin_assert(num_local_cells == cell_vertices.shape()[0]);
  dolfin_assert(num_vertices_per_cell == cell_vertices.shape()[1]);

  // Get cell index offset for this process
  const std::size_t cell_offset = MPI::global_offset(num_local_cells, true);

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

  // Ownership ranges for vertices for all processes
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
    // FIXME: Could look here to figure out size of send buffer,
    //        and reserve memory in vectors
    while (vc->first >= ownership[p])
      ++p;

    dolfin_assert(p < send_buffer.size());
    send_buffer[p].push_back(vc->first);
    send_buffer[p].push_back(vc->second);
  }

  // Create receive buffer
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
  std::set<std::size_t> my_vertices(cell_vertices.data(),
                                    cell_vertices.data() + cell_vertices.num_elements());

  // Request vertex-cell map for all vertices that I have from the owning
  // process
  p = 0;
  std::vector<std::vector<std::size_t> > required_vertices(num_processes);
  for (std::set<std::size_t>::const_iterator v = my_vertices.begin();
          v != my_vertices.end(); ++v)
  {
    // FIXME: Could look here to figure out size of send buffer,
    //        and reserve memory in vectors
    while (*v >= ownership[p])
      ++p;

    dolfin_assert(p < send_buffer.size());
    required_vertices[p].push_back(*v);
  }

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

  // Now, the rest is local (no more communication . . . ,) ----------------

  // Resize graph and clear ghost vertices
  local_graph.resize(num_local_cells);
  ghost_vertices.clear();

  // Number of vertices per facet
  const std::size_t num_vertices_per_facet = num_vertices_per_cell - 1;

  // My local cell range
  const std::pair<std::size_t, std::size_t>
    my_cell_range = MPI::local_range(mesh_data.num_global_cells);
  
  // Build renumbering ordering map (maps global to local)
  std::size_t count = 0;
  boost::unordered_map<std::size_t, std::size_t> reorder;
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
    boost::unordered_map<std::size_t, std::size_t>::const_iterator local_v
      = reorder.find(_cell_vertices[i]);
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

      // Number of cells in intersection
      std::size_t intersection_size = it - intersection.begin();

      // Should have 1 or 2 cell connections, otherwise something is wrong
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

  #else

  // Use algorithm that does not require MPI to be installed
  FacetCellMap facet_cell_map;
  compute_local_dual_graph(mesh_data, local_graph, facet_cell_map);

  #endif

}
//-----------------------------------------------------------------------------
void GraphBuilder::compute_dual_graph(const LocalMeshData& mesh_data,
                            std::vector<std::set<std::size_t> >& local_graph,
                            std::set<std::size_t>& ghost_vertices)
{
  FacetCellMap facet_cell_map;

  #ifdef HAS_MPI

  compute_local_dual_graph(mesh_data, local_graph, facet_cell_map);
  compute_nonlocal_dual_graph(mesh_data, local_graph, facet_cell_map, ghost_vertices);

  #else

  compute_local_dual_graph(mesh_data, local_graph, facet_cell_map);

  #endif

}
//-----------------------------------------------------------------------------
void GraphBuilder::compute_local_dual_graph(const LocalMeshData& mesh_data,
                            std::vector<std::set<std::size_t> >& local_graph,
                            FacetCellMap& facet_cell_map)
{
  Timer timer("Compute local dual graph");

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

  std::vector<std::size_t> cellvtx(num_vertices_per_cell);
  std::vector<std::size_t> facet(num_vertices_per_facet);

  // Iterate over all cells
  for (std::size_t i = 0; i < num_local_cells; ++i)
  {
    // Copy cell vertices and sort into order, taking a subset (minus one vertex) 
    // to form a set of facet vertices
    std::copy(cell_vertices[i].begin(), cell_vertices[i].end(), cellvtx.begin());
    std::sort(cellvtx.begin(), cellvtx.end());
    std::copy(cellvtx.begin() + 1, cellvtx.end(), facet.begin());
    
    // Iterate over facets in cell
    for(std::size_t j = 0; j < num_vertices_per_cell; ++j)
    {
      // Look for facet in map
      const FacetCellMap::const_iterator join_cell = facet_cell_map.find(facet);

      // If facet not found in map, insert [facet => local cell index] into map
      if (join_cell == facet_cell_map.end())
        facet_cell_map[facet] = i;
      else
      {
        // Already in map. Connect cells and delete facet from map
        // Add offset to cell index when inserting into local_graph
        local_graph[i].insert(join_cell->second + cell_offset);
        local_graph[join_cell->second].insert(i + cell_offset);
        // Save memory and search time by erasing
        facet_cell_map.erase(join_cell);
      }

      // Change facet by one entry, cycling from [1,2,3]->[0,2,3]->[0,1,3]->[0,1,2] (for tetrahedron)
      if (j != num_vertices_per_facet)
        facet[j] = cellvtx[j];

    }
  }
}
//-----------------------------------------------------------------------------
void GraphBuilder::compute_nonlocal_dual_graph(const LocalMeshData& mesh_data,
                            std::vector<std::set<std::size_t> >& local_graph,
                            FacetCellMap& facet_cell_map,
                            std::set<std::size_t>& ghost_vertices)
{
  Timer timer("Compute non-local dual graph");

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

  // Get offset for this process
  const std::size_t offset = MPI::global_offset(num_local_cells, true);
  const std::size_t num_processes = MPI::num_processes();

  // Clear ghost vertices
  ghost_vertices.clear();

  // Send facet-cell map to intermediary match-making processes
  std::vector<std::vector<std::size_t> > data_to_send(num_processes);
  std::vector<std::vector<std::size_t> > data_received(num_processes);

  // Pack map data and send to match-maker process
  boost::unordered_map<std::vector<std::size_t>, std::size_t>::const_iterator it;
  for (it = facet_cell_map.begin(); it != facet_cell_map.end(); ++it)
  {
    // Use first vertex of facet to partition into blocks
    // FIXME: could use a better index? 
    // First vertex is slightly skewed towards low values - may not be important
    std::size_t dest_proc = MPI::index_owner((it->first)[0], mesh_data.num_global_vertices);
    // Pack map into vectors to send
    for (std::size_t i = 0; i < num_vertices_per_facet; ++i)
      data_to_send[dest_proc].push_back((it->first)[i]);
    // Add offset to cell numbers sent off process
    data_to_send[dest_proc].push_back(it->second + offset);
  }

  MPI::all_to_all(data_to_send, data_received);
  
  // Clean out send vector for later reuse
  for (std::size_t i = 0; i < num_processes; i++)
    data_to_send[i].clear();

  // Map to connect processes and cells, using facet as key
  boost::unordered_map<std::vector<std::size_t>, std::pair<std::size_t, std::size_t> > matchmap;
  // FIXME: set hash size
    
  std::vector<std::size_t> facet(num_vertices_per_facet);

  for (std::size_t proc_k = 0; proc_k < num_processes; ++proc_k)
  {
    const std::vector<std::size_t>& data_k = data_received[proc_k];
    // Unpack into map
    for (std::size_t i = 0; i < data_k.size(); i += (num_vertices_per_facet + 1))
    {
      std::size_t j = 0;
      for (j = 0; j < num_vertices_per_facet; ++j)
        facet[j] = data_k[i + j];
   
      if (matchmap.find(facet) == matchmap.end())
      {
        matchmap[facet] = std::make_pair(proc_k, data_k[i + j]);
      }
      else
      {
        // Found a match of two facets - send back to owners
        const std::size_t proc1 = matchmap[facet].first;
        const std::size_t cell1 = matchmap[facet].second;
        const std::size_t proc2 = proc_k;
        const std::size_t cell2 = data_k[i + j];
        data_to_send[proc1].push_back(cell1);
        data_to_send[proc1].push_back(cell2);
        data_to_send[proc2].push_back(cell2);
        data_to_send[proc2].push_back(cell1);        
        matchmap.erase(facet); // saves memory and search time
      }
      
    }    
  }

  MPI::all_to_all(data_to_send, data_received);

  // Flatten received data and insert connected cells into local map
  for (std::vector<std::vector<std::size_t> >::iterator r = data_received.begin(); 
       r != data_received.end(); ++r)
  {
    const std::vector<std::size_t>& cell_list = *r;
    for (std::size_t i = 0 ; i < r->size() ; i+=2)
    {
      dolfin_assert(cell_list[i] >= offset);
      dolfin_assert(cell_list[i] - offset < local_graph.size());
      
      local_graph[cell_list[i] - offset].insert(cell_list[i+1]);
      ghost_vertices.insert(cell_list[i+1]);
    }
  }
  
}
//-----------------------------------------------------------------------------
