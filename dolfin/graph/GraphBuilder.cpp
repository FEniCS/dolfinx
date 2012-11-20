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
// First added:  2010-02-19
// Last changed:

#include <algorithm>
#include <numeric>
#include <set>
#include <vector>
#include <boost/unordered_set.hpp>
#include <boost/unordered_map.hpp>

#include <dolfin/log/log.h>
#include <dolfin/common/types.h>
#include <dolfin/common/MPI.h>
#include <dolfin/common/Timer.h>
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

  /*
  // Build graph
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    const std::vector<std::size_t>& dofs0 = dofmap0.cell_dofs(cell->index());
    const std::vector<std::size_t>& dofs1 = dofmap1.cell_dofs(cell->index());

    std::vector<std::size_t>::const_iterator node;
    for (node = dofs0.begin(); node != dofs0.end(); ++node)
      graph[*node].insert(dofs1.begin(), dofs1.end());
  }
  */

  // Build graph
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    const std::vector<DolfinIndex>& dofs0 = dofmap0.cell_dofs(cell->index());
    const std::vector<DolfinIndex>& dofs1 = dofmap1.cell_dofs(cell->index());

    std::vector<DolfinIndex>::const_iterator node;
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
      for (boost::unordered_set<std::size_t>::const_iterator entity_index = entity_list0.begin(); entity_index != entity_list0.end(); ++entity_index)
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

  const std::size_t num_mpi_procs = MPI::num_processes();

  // List of cell vertices
  const boost::multi_array<std::size_t, 2>& cell_vertices = mesh_data.cell_vertices;

  const std::size_t num_local_cells    = mesh_data.global_cell_indices.size();
  const std::size_t topological_dim    = mesh_data.tdim;
  const std::size_t num_cell_facets    = topological_dim + 1;
  const std::size_t num_facet_vertices = topological_dim;
  const std::size_t num_cell_vertices  = topological_dim + 1;

  // Resize graph (cell are graph vertices, cell-cell connections are graph edges)
  local_graph.resize(num_local_cells);

  // Get number of cells on each process
  std::vector<std::size_t> cells_per_process;
  MPI::all_gather(num_local_cells, cells_per_process);

  // Compute offset for going from local to (internal) global numbering
  std::vector<std::size_t> process_offsets(num_mpi_procs);
  for (std::size_t i = 0; i < num_mpi_procs; ++i)
    process_offsets[i] = std::accumulate(cells_per_process.begin(), cells_per_process.begin() + i, 0);
  const std::size_t process_offset = process_offsets[MPI::process_number()];

  // Compute local edges (cell-cell connections) using global (internal) numbering
  cout << "Compute local cell-cell connections" << endl;
  compute_connectivity(cell_vertices, num_facet_vertices, process_offset,
                       local_graph);
  cout << "Finished computing local cell-cell connections" << endl;

  //-----------------------------------------------
  // The rest only applies when running in parallel
  //-----------------------------------------------

  // Determine candidate ghost cells (graph ghost vertices)
  info("Preparing data to to send off-process.");
  std::vector<std::size_t> local_boundary_cells;
  for (std::size_t i = 0; i < num_local_cells; ++i)
  {
    dolfin_assert(i < local_graph.size());
    if (local_graph[i].size() != num_cell_facets)
      local_boundary_cells.push_back(i);
  }

  // Get number of possible ghost cells coming from each process
  std::vector<std::size_t> boundary_cells_per_process;
  const std::size_t local_boundary_cells_size = local_boundary_cells.size();
  MPI::all_gather(local_boundary_cells_size, boundary_cells_per_process);

  // Pack local data for candidate ghost cells (global cell index and vertices)
  std::vector<std::size_t> connected_cell_data;
  for (std::size_t i = 0; i < local_boundary_cells.size(); ++i)
  {
    // Global (internal) cell index
    connected_cell_data.push_back(local_boundary_cells[i] + process_offset);

    // Candidate cell vertices
    boost::multi_array<std::size_t, 2>::const_subarray<1>::type vertices = cell_vertices[local_boundary_cells[i]];
    for (std::size_t j = 0; j < num_cell_vertices; ++j)
      connected_cell_data.push_back(vertices[j]);
  }

  // Prepare package to send (do not send data belonging to this process)
  std::vector<std::size_t> destinations;
  std::vector<std::size_t> send_data;
  for (std::size_t i = 0; i < num_mpi_procs; ++i)
  {
    if (i != MPI::process_number())
    {
      send_data.insert(send_data.end(), connected_cell_data.begin(),
                           connected_cell_data.end());
      destinations.insert(destinations.end(), connected_cell_data.size(), i);
    }
  }

  // Set number of candidate ghost cells on this process to zero
  // (not communicated to self)
  boundary_cells_per_process[MPI::process_number()] = 0;

  // FIXME: Make the communication cleverer and more scalable. Send to
  // one process at a time, and remove cells when it is know that all
  // neighbors have been found.

  // Distribute data to all processes
  std::vector<std::size_t> received_data;
  std::vector<std::size_t> sources;
  MPI::distribute(send_data, destinations, received_data, sources);

  // Data structures for unpacking data
  std::vector<std::vector<std::vector<std::size_t> > > candidate_ghost_cell_vertices(num_mpi_procs);
  std::vector<std::vector<std::size_t> > candidate_ghost_cell_global_indices(num_mpi_procs);

  std::size_t _offset = 0;
  for (std::size_t i = 0; i < num_mpi_procs - 1; ++i)
  {
    // Check if there is data to unpack
    if (_offset >= sources.size())
      break;

    const std::size_t p = sources[_offset];
    dolfin_assert(p < boundary_cells_per_process.size());
    const std::size_t data_length = (num_cell_vertices + 1)*boundary_cells_per_process[p];

    std::vector<std::size_t>& _global_cell_indices         = candidate_ghost_cell_global_indices[p];
    std::vector<std::vector<std::size_t> >& _cell_vertices = candidate_ghost_cell_vertices[p];

    // Loop over data for each cell
    for (std::size_t j = _offset; j < _offset + data_length; j += num_cell_vertices + 1)
    {
      dolfin_assert(sources[j] == p);

      // Get cell global index
      _global_cell_indices.push_back(received_data[j]);

      // Get cell vertices
      std::vector<std::size_t> vertices;
      for (std::size_t k = 0; k < num_cell_vertices; ++k)
        vertices.push_back(received_data[(j + 1) + k]);
      _cell_vertices.push_back(vertices);
    }

    // Update offset
    _offset += data_length;
  }

  // Add off-process (ghost) edges (cell-cell) connections to graph
  info("Compute graph ghost edges.");
  std::set<std::size_t> ghost_cell_global_indices;
  for (std::size_t i = 0; i < candidate_ghost_cell_vertices.size(); ++i)
  {
    compute_ghost_connectivity(cell_vertices, local_boundary_cells,
                               candidate_ghost_cell_vertices[i],
                               candidate_ghost_cell_global_indices[i],
                               num_facet_vertices,
                               local_graph, ghost_cell_global_indices);
  }
  ghost_vertices = ghost_cell_global_indices;
  info("Finish compute graph ghost edges.");;
}
//-----------------------------------------------------------------------------
void GraphBuilder::compute_connectivity(const boost::multi_array<std::size_t, 2>& cell_vertices,
                                  std::size_t num_facet_vertices, std::size_t offset,
                                  std::vector<std::set<std::size_t> >& local_graph)
{
  // FIXME: Continue to make this function more efficient

  // Declare iterators
  boost::multi_array<std::size_t, 2>::const_iterator c_vertices;
  boost::multi_array<std::size_t, 2>::const_subarray<1>::type::const_iterator vertex;
  std::vector<std::size_t>::const_iterator c_vertex;
  std::vector<std::size_t>::const_iterator connected_cell;

  boost::unordered_map<std::size_t, std::vector<std::size_t> > vertex_connectivity;
  std::pair<boost::unordered_map<std::size_t, std::vector<std::size_t> >::iterator, bool> ret;

  // Build (global vertex)-(local cell) connectivity
  double tt = time();
  for (c_vertices = cell_vertices.begin(); c_vertices != cell_vertices.end(); ++c_vertices)
  {
    const std::size_t cell_index = c_vertices - cell_vertices.begin();
    for (vertex = c_vertices->begin(); vertex != c_vertices->end(); ++vertex)
    {
      ret = vertex_connectivity.insert(std::pair<std::size_t, std::vector<std::size_t> >(*vertex, std::vector<std::size_t>()) );
      ret.first->second.push_back(cell_index);
    }
  }
  tt = time() - tt;
  info("Time to build vertex-cell connectivity map: %g", tt);

  std::vector<std::size_t>::const_iterator connected_cell0;
  std::vector<std::size_t>::const_iterator connected_cell1;
  boost::multi_array<std::size_t, 2>::const_subarray<1>::type::const_iterator cell_vertex;

  tt = time();
  // Iterate over all vertices
  boost::unordered_map<std::size_t, std::vector<std::size_t> >::const_iterator _vertex;
  for (_vertex = vertex_connectivity.begin(); _vertex != vertex_connectivity.end(); ++_vertex)
  {
    const std::vector<std::size_t>& cell_list = _vertex->second;

    // Iterate over connected cells
    for (connected_cell0 = cell_list.begin() ; connected_cell0 != cell_list.end() -1; ++connected_cell0)
    {
      for (connected_cell1 = connected_cell0 + 1; connected_cell1 != cell_list.end(); ++connected_cell1)
      {
        boost::multi_array<std::size_t, 2>::const_subarray<1>::type cell0_vertices = cell_vertices[*connected_cell0];
        boost::multi_array<std::size_t, 2>::const_subarray<1>::type cell1_vertices = cell_vertices[*connected_cell1];

        std::size_t num_common_vertices = 0;
        for (cell_vertex = cell1_vertices.begin(); cell_vertex != cell1_vertices.end(); ++cell_vertex)
        {
          if (std::find(cell0_vertices.begin(), cell0_vertices.end(), *cell_vertex) != cell0_vertices.end())
            ++num_common_vertices;
          if (num_common_vertices == num_facet_vertices)
          {
            local_graph[*connected_cell0].insert(*connected_cell1 + offset);
            local_graph[*connected_cell1].insert(*connected_cell0 + offset);
          }
        }

      }
    }
  }
  tt = time() - tt;

  info("Time to build local dual graph: : %g", tt);
}
//-----------------------------------------------------------------------------
std::size_t GraphBuilder::compute_ghost_connectivity(const boost::multi_array<std::size_t, 2>& cell_vertices,
                  const std::vector<std::size_t>& local_boundary_cells,
                  const std::vector<std::vector<std::size_t> >& candidate_ghost_vertices,
                  const std::vector<std::size_t>& candidate_ghost_global_indices,
                  std::size_t num_facet_vertices,
                  std::vector<std::set<std::size_t> >& local_graph,
                  std::set<std::size_t>& ghost_cells)
{
  const std::size_t num_ghost_vertices_0 = ghost_cells.size();

  // Declare iterators
  boost::multi_array<std::size_t, 2>::const_iterator c_vertices;
  boost::multi_array<std::size_t, 2>::const_subarray<1>::type::const_iterator vertex;
  boost::multi_array<std::size_t, 2>::const_subarray<1>::type::const_iterator c_vertex;
  std::vector<std::size_t>::const_iterator connected_cell;

  boost::unordered_map<std::size_t, std::pair<std::vector<std::size_t>, std::vector<std::size_t> > > vertex_connectivity;
  std::pair<boost::unordered_map<std::size_t, std::pair<std::vector<std::size_t>, std::vector<std::size_t> > >::iterator, bool> ret;

  // Build boundary (global vertex)-(local cell) connectivity
  double tt = time();
  std::vector<std::size_t>::const_iterator local_cell;
  for (local_cell = local_boundary_cells.begin(); local_cell != local_boundary_cells.end(); ++local_cell)
  {
    boost::multi_array<std::size_t, 2>::const_subarray<1>::type c_vertices = cell_vertices[*local_cell];
    for (vertex = c_vertices.begin(); vertex != c_vertices.end(); ++vertex)
    {
      std::pair<std::vector<std::size_t>, std::vector<std::size_t> > tmp;
      ret = vertex_connectivity.insert(std::pair<std::size_t, std::pair<std::vector<std::size_t>, std::vector<std::size_t> > >(*vertex, tmp) );
      ret.first->second.first.push_back(*local_cell);
    }
  }
  tt = time() - tt;
  info("Time to build local boundary vertex-cell connectivity map: %g", tt);

  // Build off-process boundary (global vertex)-(local cell) connectivity
  tt = time();
  for (std::vector<std::vector<std::size_t> >::const_iterator c_vertices = candidate_ghost_vertices.begin(); c_vertices != candidate_ghost_vertices.end(); ++c_vertices)
  {
    const std::size_t cell_index = c_vertices - candidate_ghost_vertices.begin();
    std::vector<std::size_t>::const_iterator vertex;
    for (vertex = c_vertices->begin(); vertex != c_vertices->end(); ++vertex)
    {
      std::pair<std::vector<std::size_t>, std::vector<std::size_t> > tmp;
      ret = vertex_connectivity.insert(std::pair<std::size_t, std::pair<std::vector<std::size_t>, std::vector<std::size_t> > >(*vertex, tmp) );
      ret.first->second.second.push_back(cell_index);
    }
  }
  tt = time() - tt;
  info("Time to build ghost boundary vertex-cell connectivity map: %g", tt);

  // Iterate over local boundary cells
  tt = time();
  for (local_cell = local_boundary_cells.begin(); local_cell != local_boundary_cells.end(); ++local_cell)
  {
    boost::multi_array<std::size_t, 2>::const_subarray<1>::type c_vertices = cell_vertices[*local_cell];

    // Iterate over local cell vertices
    for (c_vertex = c_vertices.begin(); c_vertex != c_vertices.end(); ++c_vertex)
    {
      // Iterate over ghost cells connected to this vertex
      for (connected_cell = vertex_connectivity[*c_vertex].second.begin();
                     connected_cell != vertex_connectivity[*c_vertex].second.end();
                     ++connected_cell)
      {
        // Vertices of candidate neighbour
        const std::vector<std::size_t>& candidate_vertices = candidate_ghost_vertices[*connected_cell];

        std::size_t num_common_vertices = 0;
        for (vertex = c_vertices.begin(); vertex != c_vertices.end(); ++vertex)
        {
          if (std::find(candidate_vertices.begin(), candidate_vertices.end(), *vertex) != candidate_vertices.end())
            ++num_common_vertices;
          if (num_common_vertices == num_facet_vertices)
          {
            local_graph[*local_cell].insert(candidate_ghost_global_indices[*connected_cell]);
            ghost_cells.insert(candidate_ghost_global_indices[*connected_cell]);
            break;
          }
        }
      }
    }
  }
  tt = time() - tt;
  info("Time to build ghost dual graph: : %g", tt);
  return ghost_cells.size() - num_ghost_vertices_0;
}
//-----------------------------------------------------------------------------
