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
  Timer timer("Compute dual graph [new]");

  // List of cell vertices
  const boost::multi_array<std::size_t, 2>& cell_vertices = mesh_data.cell_vertices;
  const std::size_t num_local_cells = mesh_data.global_cell_indices.size();
  const std::size_t num_vertices_per_cell = mesh_data.num_vertices_per_cell;

  dolfin_assert(num_local_cells == cell_vertices.shape()[0]);
  dolfin_assert(num_vertices_per_cell == cell_vertices.shape()[1]);

  local_graph.resize(num_local_cells);

  const std::size_t offset = MPI::global_offset(num_local_cells, true);

  // Compute local edges (cell-cell connections) using global (internal) numbering

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

  const std::size_t num_processes = MPI::num_processes();
  const std::size_t process_number = MPI::process_number();

  // Clear ghost vertices
  ghost_vertices.clear();

  // Create MPI ring
  const int source = (num_processes + process_number - 1) % num_processes;
  const int dest   = (process_number + 1) % num_processes;

  // FIXME: Find better way to send map data structures via MPI

  // Repeat (n-1) times, to go round ring
  std::vector<std::pair<std::vector<std::size_t>, std::size_t> > comm_data;
  for(std::size_t i = 0; i < (num_processes - 1); ++i)
  {
    // FIXME: Improve memory management. Can the maps be serialised/sent directly?

    // Pack data to send
    comm_data.resize(othermap.size());
    std::copy(othermap.begin(), othermap.end(), comm_data.begin());

    // Shift data to next process
    MPI::send_recv(comm_data, dest, comm_data, source);

    // Unpack data
    othermap.clear();
    othermap.insert(comm_data.begin(), comm_data.end());

    const std::size_t mapsize = MPI::sum(othermap.size());
    if(process_number == 0)
    {
      std::cout << "t = " << (time() - tt) << ", iteration: " << i
          << ", average map size = " << mapsize/num_processes << std::endl;
    }

    // FIXME: The below looks very suspicious - it could screw up the iterators

    // Go through local facets, looking for a matching facet in othermap
    VectorMap::iterator fcell = facet_cell.begin();
    while (fcell != facet_cell.end())
    {
      // Check if maps contains same facet
      std::map<std::vector<std::size_t>, std::size_t>::iterator join_cell = othermap.find(fcell->first);
      if (join_cell != othermap.end())
      {
        // Found neighbours, insert into local_graph and delete facets
        // from both maps
        local_graph[fcell->second].insert(join_cell->second);
        ghost_vertices.insert(join_cell->second);
        facet_cell.erase(fcell++);
        othermap.erase(join_cell);
      }
      else
        ++fcell;
    }
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

