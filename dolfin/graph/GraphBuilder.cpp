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
// Modified by Chris Richardson, 2012-2014
//
// First added:  2010-02-19
// Last changed: 2013-01-31

#include <algorithm>
#include <numeric>
#include <set>
#include <utility>
#include <vector>
#include <unordered_map>
#include <unordered_set>

#include <dolfin/log/log.h>
#include <dolfin/common/MPI.h>
#include <dolfin/common/Timer.h>
#include <dolfin/common/types.h>
#include <dolfin/fem/GenericDofMap.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/MeshEntityIterator.h>
#include <dolfin/mesh/Vertex.h>
#include "GraphBuilder.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
Graph GraphBuilder::local_graph(const Mesh& mesh, const GenericDofMap& dofmap0,
                                                  const GenericDofMap& dofmap1)
{
  Timer timer("Build local sparsity graph from dofmaps");

  // Create empty graph
  const std::size_t n = dofmap0.global_dimension();
  Graph graph(n);

  // Build graph
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    const ArrayView<const dolfin::la_index> dofs0
      = dofmap0.cell_dofs(cell->index());
    const ArrayView<const dolfin::la_index> dofs1
      = dofmap1.cell_dofs(cell->index());
    //std::vector<dolfin::la_index>::const_iterator node0, node1;
    for (auto node0 = dofs0.begin(); node0 != dofs0.end(); ++node0)
      for (auto node1 = dofs1.begin(); node1 != dofs1.end(); ++node1)
        if (*node0 != *node1)
          graph[*node0].insert(*node1);
  }

  return graph;
}
//-----------------------------------------------------------------------------
Graph GraphBuilder::local_graph(const Mesh& mesh,
                                const std::vector<std::size_t>& coloring_type)
{
  // Initialise mesh
  for (std::size_t i = 0; i < coloring_type.size(); ++i)
    mesh.init(coloring_type[i]);
  for (std::size_t i = 1; i < coloring_type.size(); ++i)
    mesh.init(coloring_type[i - 1], coloring_type[i]);

  // Check coloring type
  dolfin_assert(coloring_type.size() >= 2);
  dolfin_assert(coloring_type.front() == coloring_type.back());

  // Create graph
  const std::size_t num_vertices = mesh.num_entities(coloring_type[0]);
  Graph graph(num_vertices);

  // Build graph
  for (MeshEntityIterator vertex_entity(mesh, coloring_type[0]);
       !vertex_entity.end(); ++vertex_entity)
  {
    const std::size_t vertex_entity_index = vertex_entity->index();

    std::unordered_set<std::size_t> entity_list0;
    std::unordered_set<std::size_t> entity_list1;
    entity_list0.insert(vertex_entity_index);

    // Build list of entities, moving between levels
    for (std::size_t level = 1; level < coloring_type.size(); ++level)
    {
      for (std::unordered_set<std::size_t>::const_iterator entity_index
             = entity_list0.begin(); entity_index != entity_list0.end();
           ++entity_index)
      {
        const MeshEntity entity(mesh, coloring_type[level -1], *entity_index);
        for (MeshEntityIterator neighbor(entity, coloring_type[level]);
             !neighbor.end(); ++neighbor)
        {
          entity_list1.insert(neighbor->index());
        }
      }
      entity_list0 = entity_list1;
      entity_list1.clear();
    }

    // Add edges to graph
    graph[vertex_entity_index].insert(entity_list0.begin(), entity_list0.end());
  }

  return graph;
}
//-----------------------------------------------------------------------------
Graph GraphBuilder::local_graph(const Mesh& mesh,
                                std::size_t dim0, std::size_t dim1)
{
  mesh.init(dim0);
  mesh.init(dim1);
  mesh.init(dim0, dim1);
  mesh.init(dim1, dim0);

  // Create graph
  const std::size_t num_vertices = mesh.num_entities(dim0);
  Graph graph(num_vertices);

  // Build graph
  for (MeshEntityIterator colored_entity(mesh, dim0); !colored_entity.end();
       ++colored_entity)
  {
    const std::size_t colored_entity_index = colored_entity->index();
    for (MeshEntityIterator entity(*colored_entity, dim1); !entity.end();
         ++entity)
    {
      for (MeshEntityIterator neighbor(*entity, dim0); !neighbor.end();
           ++neighbor)
      {
        if (colored_entity_index != neighbor->index())
          graph[colored_entity_index].insert(neighbor->index());
      }
    }
  }

  return graph;
}
//-----------------------------------------------------------------------------
std::pair<std::int32_t, std::int32_t>
GraphBuilder::compute_dual_graph(const MPI_Comm mpi_comm,
                                 const boost::multi_array<std::int64_t, 2>& cell_vertices,
                                 const CellType& cell_type,
                                 const std::int64_t num_global_vertices,
                                 std::vector<std::vector<std::size_t>>& local_graph,
                                 std::set<std::int64_t>& ghost_vertices)
{
  log(PROGRESS, "Build mesh dual graph");

  // Compute local part of dual graph
  FacetCellMap facet_cell_map;
  std::int32_t num_local_edges = compute_local_dual_graph(mpi_comm, cell_vertices,
                                                          cell_type, local_graph,
                                                          facet_cell_map);

  // Compute nonlocal part
  std::int32_t num_nonlocal_edges
    = compute_nonlocal_dual_graph(mpi_comm, cell_vertices, cell_type,
                                  num_global_vertices, local_graph, facet_cell_map,
                                  ghost_vertices);

  // Shrink to fit
  local_graph.shrink_to_fit();

  return {num_local_edges, num_nonlocal_edges};
}
//-----------------------------------------------------------------------------
std::int32_t GraphBuilder::compute_local_dual_graph(
  const MPI_Comm mpi_comm,
  const boost::multi_array<std::int64_t, 2>& cell_vertices,
  const CellType& cell_type,
  std::vector<std::vector<std::size_t>>& local_graph,
  FacetCellMap& facet_cell_map)
{
  log(PROGRESS, "Build local part of mesh dual graph");

  const std::int8_t tdim = cell_type.dim();
  const std::int8_t num_entity_vertices = cell_type.num_vertices(tdim - 1);
  switch (num_entity_vertices)
  {
  case 1:
    return compute_local_dual_graph_keyed<1>(mpi_comm, cell_vertices,
                                             cell_type, local_graph,
                                             facet_cell_map);
  case 2:
    return compute_local_dual_graph_keyed<2>(mpi_comm, cell_vertices,
                                             cell_type, local_graph,
                                             facet_cell_map);
    break;
  case 3:
    return compute_local_dual_graph_keyed<3>(mpi_comm, cell_vertices,
                                             cell_type, local_graph,
                                             facet_cell_map);
  case 4:
    return compute_local_dual_graph_keyed<4>(mpi_comm, cell_vertices,
                                             cell_type, local_graph,
                                             facet_cell_map);
  default:
    dolfin_error("GraphBuilder.cpp",
                 "compute local part of dual graph",
                 "Entities with %d vertices not supported",
                 num_entity_vertices);
    return 0;
  }
}
//-----------------------------------------------------------------------------
template<int N>
std::int32_t GraphBuilder::compute_local_dual_graph_keyed(
  const MPI_Comm mpi_comm,
  const boost::multi_array<std::int64_t, 2>& cell_vertices,
  const CellType& cell_type,
  std::vector<std::vector<std::size_t>>& local_graph,
  FacetCellMap& facet_cell_map)
{
  Timer timer("Compute local part of mesh dual graph");

  const std::int8_t tdim = cell_type.dim();
  const std::int32_t num_local_cells = cell_vertices.shape()[0];
  const std::int8_t num_vertices_per_cell = cell_type.num_entities(0);
  const std::int8_t num_facets_per_cell = cell_type.num_entities(tdim - 1);
  const std::int8_t num_vertices_per_facet = cell_type.num_vertices(tdim - 1);

  dolfin_assert(N == num_vertices_per_facet);
  dolfin_assert(num_local_cells == (int) cell_vertices.shape()[0]);
  dolfin_assert(num_vertices_per_cell == (int) cell_vertices.shape()[1]);

  local_graph.resize(num_local_cells);
  facet_cell_map.clear();

  // Compute local edges (cell-cell connections) using global
  // (internal to this function, not the user numbering) numbering

  // Get offset for this process
  const std::int64_t cell_offset = MPI::global_offset(mpi_comm, num_local_cells,
                                                      true);

  // Create map from cell vertices to entity vertices
  boost::multi_array<unsigned int, 2>
    facet_vertices(boost::extents[num_facets_per_cell][num_vertices_per_facet]);
  std::vector<unsigned int> v(num_vertices_per_cell);
  std::iota(v.begin(), v.end(), 0);
  cell_type.create_entities(facet_vertices, tdim - 1, v.data());

  // Vector-of-arrays data structure, which is considerably faster than
  // vector-of-vectors.
  std::vector<std::pair<std::array<std::int32_t, N>, std::int32_t>>
    facets(num_facets_per_cell*num_local_cells);

  // Iterate over all cells and build list of all facets (keyed on
  // sorted vertex indices), with cell index attached
  int counter = 0;
  for (std::int32_t i = 0; i < num_local_cells; ++i)
  {
    // Iterate over facets of cell
    for (std::int8_t j = 0; j < num_facets_per_cell; ++j)
    {
      // Get list of facet vertices
      auto& facet = facets[counter].first;
      for (std::int8_t k = 0; k < N; ++k)
        facet[k] = cell_vertices[i][facet_vertices[j][k]];

      // Sort facet vertices
      std::sort(facet.begin(), facet.end());

      // Attach local cell index
      facets[counter].second = i;

      // Increment facet counter
      counter++;
    }
  }

  // Sort facets
  std::sort(facets.begin(), facets.end());

  // Find maching facets by comparing facet i and facet i -1
  std::size_t num_local_edges = 0;
  for (std::size_t i = 1; i < facets.size(); ++i)
  {
    const int ii = i;
    const int jj = i - 1;

    const auto& facet0 = facets[jj].first;
    const auto& facet1 = facets[ii].first;
    const int cell_index0 = facets[jj].second;
    if (std::equal(facet1.begin(), facet1.end(), facet0.begin()))
    {
      // Add edges (directed graph, so add both ways)
      const int cell_index1 = facets[ii].second;
      local_graph[cell_index0].push_back(cell_index1 + cell_offset);
      local_graph[cell_index1].push_back(cell_index0 + cell_offset);

      // Since we've just found a matching pair, the next pair cannot be
      // matching, so advance 1
      ++i;

      // Increment number of local edges found
      ++num_local_edges;
    }
    else
    {
      // No match, so add facet0 to map
      //facet_cell_map.insert(facet_cell_map.end(), {std::vector<std::size_t>(facet0.begin(),
      //    facet0.end()), cell_index0});
      facet_cell_map.push_back({std::vector<std::size_t>(facet0.begin(),
          facet0.end()), cell_index0});
    }
  }

  // Add last facet, as it's not covered by the above loop. We could
  // check it against the preceding facet, but it's easier to just
  // insert it here
  if (!facets.empty())
  {
    const int k = facets.size() - 1;
    const int cell_index = facets[k].second;
    facet_cell_map.push_back({std::vector<std::size_t>(facets[k].first.begin(),
        facets[k].first.end()), cell_index});
  }

  return num_local_edges;
}
//-----------------------------------------------------------------------------
std::int32_t GraphBuilder::compute_nonlocal_dual_graph(
  const MPI_Comm mpi_comm,
  const boost::multi_array<std::int64_t, 2>& cell_vertices,
  const CellType& cell_type,
  const std::int64_t num_global_vertices,
  std::vector<std::vector<std::size_t>>& local_graph,
  FacetCellMap& facet_cell_map,
  std::set<std::int64_t>& ghost_vertices)
{
  log(PROGRESS, "Build nonlocal part of mesh dual graph");
  Timer timer("Compute non-local part of mesh dual graph");

  // Get number of MPI processes, and return if mesh is not distributed
  const int num_processes = MPI::size(mpi_comm);
  if (num_processes == 1)
    return 0;

  // At this stage facet_cell map only contains facets->cells with
  // edge facets either interprocess or external boundaries

  const int tdim = cell_type.dim();

  // List of cell vertices
  const std::int32_t num_local_cells = cell_vertices.shape()[0];
  const std::int8_t num_vertices_per_cell = cell_type.num_entities(0);
  const std::int8_t num_vertices_per_facet = cell_type.num_vertices(tdim - 1);

  dolfin_assert(num_local_cells == (int) cell_vertices.shape()[0]);
  dolfin_assert(num_vertices_per_cell == (int) cell_vertices.shape()[1]);

  // Compute local edges (cell-cell connections) using global
  // (internal to this function, not the user numbering) numbering

  // Get offset for this process
  const std::int64_t offset = MPI::global_offset(mpi_comm, num_local_cells,
                                                 true);

  // Send facet-cell map to intermediary match-making processes
  std::vector<std::vector<std::size_t>> send_buffer(num_processes);
  std::vector<std::vector<std::size_t>> received_buffer(num_processes);

  // Pack map data and send to match-maker process
  for (auto &it : facet_cell_map)
  {
    // FIXME: Could use a better index? First vertex is slightly
    //        skewed towards low values - may not be important

    // Use first vertex of facet to partition into blocks
    const int dest_proc = MPI::index_owner(mpi_comm, (it.first)[0],
                                           num_global_vertices);

    // Pack map into vectors to send
    send_buffer[dest_proc].insert(send_buffer[dest_proc].end(),
                                  it.first.begin(), it.first.end());

    // Add offset to cell numbers sent off process
    send_buffer[dest_proc].push_back(it.second + offset);
  }

  // FIXME: This does not look memory scalable. Switch to 'post-office' model.
  // Send data
  MPI::all_to_all(mpi_comm, send_buffer, received_buffer);

  // Clear send buffer
  send_buffer = std::vector<std::vector<std::size_t>>(num_processes);

  // Map to connect processes and cells, using facet as key
  typedef boost::unordered_map<std::vector<std::size_t>,
              std::pair<std::size_t, std::size_t>> MatchMap;
  MatchMap matchmap;

  // Look for matches to send back to other processes
  std::pair<std::vector<std::size_t>,
            std::pair<std::size_t, std::size_t>> key;
  key.first.resize(num_vertices_per_facet);
  for (int p = 0; p < num_processes; ++p)
  {
    // Unpack into map
    const std::vector<std::size_t>& data_p = received_buffer[p];
    for (auto it = data_p.begin(); it != data_p.end();
         it += (num_vertices_per_facet + 1))
    {
      // Build map key
      std::copy(it, it + num_vertices_per_facet, key.first.begin());
      key.second.first = p;
      key.second.second = *(it + num_vertices_per_facet);

      // Perform map insertion/look-up
      std::pair<MatchMap::iterator, bool> data = matchmap.insert(key);

      // If data is already in the map, extract data and remove from
      // map
      if (!data.second)
      {
        // Found a match of two facets - send back to owners
        const std::size_t proc1 = data.first->second.first;
        const std::size_t proc2 = p;
        const std::size_t cell1 = data.first->second.second;
        const std::size_t cell2 = key.second.second;
        send_buffer[proc1].push_back(cell1);
        send_buffer[proc1].push_back(cell2);
        send_buffer[proc2].push_back(cell2);
        send_buffer[proc2].push_back(cell1);

        // Remove facet - saves memory and search time
        matchmap.erase(data.first);
      }
    }
  }

  // Send matches to other processes
  MPI::all_to_all(mpi_comm, send_buffer, received_buffer);

  // Clear ghost vertices
  ghost_vertices.clear();

  // Flatten received data and insert connected cells into local map
  std::int32_t num_nonlocal_edges = 0;
  for (std::size_t p = 0; p < received_buffer.size(); ++p)
  {
    const std::vector<std::size_t>& cell_list = received_buffer[p];
    for (std::size_t i = 0; i < cell_list.size(); i += 2)
    {
      dolfin_assert((std::int64_t) cell_list[i] >= offset);
      dolfin_assert((std::int64_t)  (cell_list[i] - offset)
                    < (std::int64_t) local_graph.size());

      //local_graph[cell_list[i] - offset].insert(cell_list[i + 1]);
      auto& edges = local_graph[cell_list[i] - offset];
      auto it = std::find(edges.begin(), edges.end(), cell_list[i + 1]);
      if (it == local_graph[cell_list[i] - offset].end())
        edges.push_back(cell_list[i + 1]);

      ghost_vertices.insert(cell_list[i + 1]);
    }

    ++num_nonlocal_edges;
  }

  return num_nonlocal_edges;
}
//-----------------------------------------------------------------------------
