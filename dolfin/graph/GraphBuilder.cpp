// Copyright (C) 2010 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
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
#include <dolfin/main/MPI.h>
#include <dolfin/mesh/LocalMeshData.h>
#include <dolfin/mesh/Vertex.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/common/Timer.h>
#include "GraphBuilder.h"


#ifdef HAS_SCOTCH
extern "C"
{
#include <ptscotch.h>
}
#endif

using namespace dolfin;

//-----------------------------------------------------------------------------
void GraphBuilder::compute_dual_graph(const LocalMeshData& mesh_data,
                                std::vector<std::set<uint> >& local_graph,
                                std::set<uint>& ghost_vertices)
{
  // List of cell vertices
  const std::vector<std::vector<uint> >& cell_vertices = mesh_data.cell_vertices;

  const uint num_local_cells    = mesh_data.global_cell_indices.size();
  const uint topological_dim    = mesh_data.tdim;
  const uint num_cell_facets    = topological_dim + 1;
  const uint num_facet_vertices = topological_dim;
  const uint num_cell_vertices  = topological_dim + 1;

  // Resize graph (cell are graph vertices, cell-cell connections are graph edges)
  local_graph.resize(num_local_cells);

  // Get number of cells on each process
  std::vector<uint> cells_per_process = MPI::gather(num_local_cells);

  // Compute offset for going from local to (internal) global numbering
  std::vector<uint> process_offsets(MPI::num_processes());
  for (uint i = 0; i < MPI::num_processes(); ++i)
    process_offsets[i] = std::accumulate(cells_per_process.begin(), cells_per_process.begin() + i, 0);
  const uint process_offset = process_offsets[MPI::process_number()];

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
  std::vector<uint> local_boundary_cells;
  for (uint i = 0; i < num_local_cells; ++i)
  {
    assert(i < local_graph.size());
    if (local_graph[i].size() != num_cell_facets)
      local_boundary_cells.push_back(i);
  }
  cout << "Number of possible boundary cells " << local_boundary_cells.size() << endl;

  // Get number of possible ghost cells coming from each process
  std::vector<uint> boundary_cells_per_process = MPI::gather(local_boundary_cells.size());

  // Pack local data for candidate ghost cells (global cell index and vertices)
  std::vector<uint> connected_cell_data;
  for (uint i = 0; i < local_boundary_cells.size(); ++i)
  {
    // Global (internal) cell index
    connected_cell_data.push_back(local_boundary_cells[i] + process_offset);

    // Candidate cell vertices
    const std::vector<uint>& vertices = cell_vertices[local_boundary_cells[i]];
    for (uint j = 0; j < num_cell_vertices; ++j)
      connected_cell_data.push_back(vertices[j]);
  }

  // Prepare package to send (do not send data belonging to this process)
  std::vector<uint> partition;
  std::vector<uint> transmit_data;
  for (uint i = 0; i < MPI::num_processes(); ++i)
  {
    if (i != MPI::process_number())
    {
      transmit_data.insert(transmit_data.end(), connected_cell_data.begin(),
                           connected_cell_data.end());
      partition.insert(partition.end(), connected_cell_data.size(), i);
    }
  }

  // Set number of candidate ghost cells on this process to zero (not communicated to self)
  boundary_cells_per_process[MPI::process_number()] = 0;

  // FIXME: Make the communication cleverer and more scalable. Send to
  // one process at a time, and remove cells when it is know that all
  // neighbors have been found.

  // Distribute data to all processes
  cout << "Send off-process data" << endl;
  MPI::distribute(transmit_data, partition);
  cout << "Finished sending off-process data" << endl;

  // Data structures for unpacking data
  std::vector<std::vector<std::vector<uint> > > candidate_ghost_cell_vertices(MPI::num_processes());
  std::vector<std::vector<uint> > candidate_ghost_cell_global_indices(MPI::num_processes());

  // Unpack data
  uint _offset = 0;
  for (uint i = 0; i < MPI::num_processes() - 1; ++i)
  {
    const uint p = partition[_offset];
    const uint data_length = (num_cell_vertices + 1)*boundary_cells_per_process[p];

    std::vector<uint>& _global_cell_indices         = candidate_ghost_cell_global_indices[p];
    std::vector<std::vector<uint> >& _cell_vertices = candidate_ghost_cell_vertices[p];

    // Loop over data for each cell
    for (uint j = _offset; j < _offset + data_length; j += num_cell_vertices + 1)
    {
      assert(partition[j] == p);

      // Get cell global index
      _global_cell_indices.push_back(transmit_data[j]);

      // Get cell vertices
      std::vector<uint> vertices;
      for (uint k = 0; k < num_cell_vertices; ++k)
        vertices.push_back(transmit_data[(j + 1) +k]);
      _cell_vertices.push_back(vertices);
    }

    // Update offset
    _offset += data_length;
  }

  // Add off-process (ghost) edges (cell-cell) connections to graph
  info("Compute graph ghost edges.");
  std::set<uint> ghost_cell_global_indices;
  for (uint i = 0; i < candidate_ghost_cell_vertices.size(); ++i)
  {
    compute_ghost_connectivity(cell_vertices, local_boundary_cells,
                               candidate_ghost_cell_vertices[i],
                               candidate_ghost_cell_global_indices[i],
                               num_facet_vertices,
                               local_graph, ghost_cell_global_indices);
  }
  info("Finish compute graph ghost edges.");;
}
//-----------------------------------------------------------------------------
void GraphBuilder::compute_connectivity(const std::vector<std::vector<uint> >& cell_vertices,
                                  uint num_facet_vertices, uint offset,
                                  std::vector<std::set<uint> >& local_graph)
{
  // FIXME: Continue to make this function more efficient

  // Declare iterators
  std::vector<std::vector<uint> >::const_iterator c_vertices;
  std::vector<uint>::const_iterator vertex;
  std::vector<uint>::const_iterator c_vertex;
  std::vector<uint>::const_iterator connected_cell;

  boost::unordered_map<uint, std::vector<uint> > vertex_connectivity;
  std::pair<boost::unordered_map<uint, std::vector<uint> >::iterator, bool> ret;

  // Build (global vertex)-(local cell) connectivity
  double tt = time();
  for (c_vertices = cell_vertices.begin(); c_vertices != cell_vertices.end(); ++c_vertices)
  {
    const uint cell_index = c_vertices - cell_vertices.begin();
    for (vertex = c_vertices->begin(); vertex != c_vertices->end(); ++vertex)
    {
      ret = vertex_connectivity.insert(std::pair<uint, std::vector<uint> >(*vertex, std::vector<uint>()) );
      ret.first->second.push_back(cell_index);
    }
  }
  tt = time() - tt;
  info("Time to build vertex-cell connectivity map: %g", tt);

  /*
  tt = time();
  // Iterate over all cells
  for (c_vertices = cell_vertices.begin(); c_vertices != cell_vertices.end(); ++c_vertices)
  {
     const uint index0 = c_vertices - cell_vertices.begin();

    // Iterate over cell vertices
    for (c_vertex = c_vertices->begin(); c_vertex != c_vertices->end(); ++c_vertex)
    {
      // Iterate over cells connected to this vertex
      for (connected_cell = vertex_connectivity[*c_vertex].begin(); connected_cell != vertex_connectivity[*c_vertex].end(); ++connected_cell)
      {
        const uint index1 = *connected_cell;
        if (index0 == index1)
          break;

        // Vertices of candidate neighbour
        const std::vector<uint>& candidate_vertices = cell_vertices[*connected_cell];

        uint num_common_vertices = 0;
        for (vertex = c_vertices->begin(); vertex != c_vertices->end(); ++vertex)
        {
          if (std::find(candidate_vertices.begin(), candidate_vertices.end(), *vertex) != candidate_vertices.end())
            ++num_common_vertices;
          if (num_common_vertices == num_facet_vertices)
          {
            local_graph[index0].insert(index1 + offset);
            local_graph[index1].insert(index0 + offset);
            break;
          }
        }
      }
    }
  }
  tt = time() - tt;
  */

  std::vector<uint>::const_iterator connected_cell0;
  std::vector<uint>::const_iterator connected_cell1;
  std::vector<uint>::const_iterator cell_vertex;

  tt = time();
  // Iterate over all vertices
  boost::unordered_map<uint, std::vector<uint> >::const_iterator _vertex;
  for (_vertex = vertex_connectivity.begin(); _vertex != vertex_connectivity.end(); ++_vertex)
  {
    const std::vector<uint>& cell_list = _vertex->second;

    // Iterate over connected cells
    for (connected_cell0 = cell_list.begin() ; connected_cell0 != cell_list.end() -1; ++connected_cell0)
    {
      for (connected_cell1 = connected_cell0 + 1; connected_cell1 != cell_list.end(); ++connected_cell1)
      {
        const std::vector<uint>& cell0_vertices = cell_vertices[*connected_cell0];
        const std::vector<uint>& cell1_vertices = cell_vertices[*connected_cell1];

        uint num_common_vertices = 0;
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
dolfin::uint GraphBuilder::compute_ghost_connectivity(const std::vector<std::vector<uint> >& cell_vertices,
                                          const std::vector<uint>& local_boundary_cells,
                                          const std::vector<std::vector<uint> >& candidate_ghost_vertices,
                                          const std::vector<uint>& candidate_ghost_global_indices,
                                          uint num_facet_vertices,
                                          std::vector<std::set<uint> >& local_graph,
                                          std::set<uint>& ghost_cells)
{
  const uint num_ghost_vertices_0 = ghost_cells.size();

  // Declare iterators
  std::vector<std::vector<uint> >::const_iterator c_vertices;
  std::vector<uint>::const_iterator vertex;
  std::vector<uint>::const_iterator c_vertex;
  std::vector<uint>::const_iterator connected_cell;

  boost::unordered_map<uint, std::pair<std::vector<uint>, std::vector<uint> > > vertex_connectivity;
  std::pair<boost::unordered_map<uint, std::pair<std::vector<uint>, std::vector<uint> > >::iterator, bool> ret;

  // Build boundary (global vertex)-(local cell) connectivity
  double tt = time();
  std::vector<uint>::const_iterator local_cell;
  for (local_cell = local_boundary_cells.begin(); local_cell != local_boundary_cells.end(); ++local_cell)
  {
    const std::vector<uint>& c_vertices = cell_vertices[*local_cell];
    for (vertex = c_vertices.begin(); vertex != c_vertices.end(); ++vertex)
    {
      std::pair<std::vector<uint>, std::vector<uint> > tmp;
      ret = vertex_connectivity.insert(std::pair<uint, std::pair<std::vector<uint>, std::vector<uint> > >(*vertex, tmp) );
      ret.first->second.first.push_back(*local_cell);
    }
  }
  tt = time() - tt;
  info("Time to build local boundary vertex-cell connectivity map: %g", tt);

  // Build off-process boundary (global vertex)-(local cell) connectivity
  tt = time();
  for (c_vertices = candidate_ghost_vertices.begin(); c_vertices != candidate_ghost_vertices.end(); ++c_vertices)
  {
    const uint cell_index = c_vertices - candidate_ghost_vertices.begin();
    for (vertex = c_vertices->begin(); vertex != c_vertices->end(); ++vertex)
    {
      std::pair<std::vector<uint>, std::vector<uint> > tmp;
      ret = vertex_connectivity.insert(std::pair<uint, std::pair<std::vector<uint>, std::vector<uint> > >(*vertex, tmp) );
      ret.first->second.second.push_back(cell_index);
    }
  }
  tt = time() - tt;
  info("Time to build ghost boundary vertex-cell connectivity map: %g", tt);

  // Iterate over local boundary cells
  tt = time();
  for (local_cell = local_boundary_cells.begin(); local_cell != local_boundary_cells.end(); ++local_cell)
  {
    const std::vector<uint>& c_vertices = cell_vertices[*local_cell];

    // Iterate over local cell vertices
    for (c_vertex = c_vertices.begin(); c_vertex != c_vertices.end(); ++c_vertex)
    {
      // Iterate over ghost cells connected to this vertex
      for (connected_cell = vertex_connectivity[*c_vertex].second.begin();
                     connected_cell != vertex_connectivity[*c_vertex].second.end();
                     ++connected_cell)
      {
        // Vertices of candidate neighbour
        const std::vector<uint>& candidate_vertices = candidate_ghost_vertices[*connected_cell];

        uint num_common_vertices = 0;
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
