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
#include <dolfin/mesh/LocalMeshData.h>
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
    graph[vertex_entity_index].insert(entity_list0.begin(),
                                      entity_list0.end());
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
void GraphBuilder::compute_dual_graph(
  const MPI_Comm mpi_comm,
  const LocalMeshData& mesh_data,
  std::vector<std::set<std::size_t>>& local_graph,
  std::set<std::size_t>& ghost_vertices)
{
  FacetCellMap facet_cell_map;

#ifdef HAS_MPI
  compute_local_dual_graph(mpi_comm, mesh_data, local_graph, facet_cell_map);
  compute_nonlocal_dual_graph(mpi_comm, mesh_data, local_graph, facet_cell_map,
                              ghost_vertices);
  #else
  compute_local_dual_graph(mpi_comm, mesh_data, local_graph, facet_cell_map);
  #endif
}
//-----------------------------------------------------------------------------
void GraphBuilder::compute_local_dual_graph(const MPI_Comm mpi_comm,
                               const LocalMeshData& mesh_data,
                               std::vector<std::set<std::size_t>>& local_graph,
                               FacetCellMap& facet_cell_map)
{
  std::unique_ptr<CellType> cell_type(CellType::create(mesh_data.cell_type));
  compute_local_dual_graph(mpi_comm, mesh_data.cell_vertices,
                       *cell_type, mesh_data.global_cell_indices, local_graph,
                       facet_cell_map);
}
//-----------------------------------------------------------------------------
void GraphBuilder::compute_local_dual_graph(
  const MPI_Comm mpi_comm,
  const boost::multi_array<std::size_t, 2>& cell_vertices,
  const CellType& cell_type,
  const std::vector<std::size_t>& global_cell_indices,
  std::vector<std::set<std::size_t>>& local_graph,
  FacetCellMap& facet_cell_map)
{
  Timer timer("Compute local part of mesh dual graph");

  const std::size_t tdim = cell_type.dim();
  const std::size_t num_local_cells = global_cell_indices.size();
  const std::size_t num_vertices_per_cell = cell_type.num_entities(0);
  const std::size_t num_facets_per_cell = cell_type.num_entities(tdim - 1);
  const std::size_t num_vertices_per_facet = cell_type.num_vertices(tdim - 1);

  dolfin_assert(num_local_cells == cell_vertices.shape()[0]);
  dolfin_assert(num_vertices_per_cell == cell_vertices.shape()[1]);

  local_graph.resize(num_local_cells);
  facet_cell_map.clear();

  // Compute local edges (cell-cell connections) using global
  // (internal to this function, not the user numbering) numbering

  // Get offset for this process
  const std::size_t cell_offset = MPI::global_offset(mpi_comm, num_local_cells,
                                                     true);

  // Create map from facet (list of vertex indices) to cells
  facet_cell_map.rehash((facet_cell_map.size()
                     + num_local_cells)/facet_cell_map.max_load_factor() + 1);

  boost::multi_array<unsigned int, 2>
    fverts(boost::extents[num_facets_per_cell][num_vertices_per_facet]);

  std::vector<unsigned int> cellvtx(num_vertices_per_cell);
  std::pair<std::vector<std::size_t>, std::size_t> map_entry;
  map_entry.first.resize(num_vertices_per_facet);

  // Iterate over all cells
  for (std::size_t i = 0; i < num_local_cells; ++i)
  {
    map_entry.second = i;

    std::copy(cell_vertices[i].begin(), cell_vertices[i].end(),
              cellvtx.begin());
    cell_type.create_entities(fverts, tdim - 1, cellvtx.data());

    // Iterate over facets in cell
    for (std::size_t j = 0; j < num_facets_per_cell; ++j)
    {
      std::copy(fverts[j].begin(), fverts[j].end(), map_entry.first.begin());
      std::sort(map_entry.first.begin(), map_entry.first.end());

      // Map lookup/insert
      std::pair<FacetCellMap::iterator, bool> map_lookup
        = facet_cell_map.insert(map_entry);

      // If facet was already in the map
      if (!map_lookup.second)
      {
        // Already in map. Connect cells and delete facet from map
        // Add offset to cell index when inserting into local_graph
        local_graph[i].insert(map_lookup.first->second + cell_offset);
        local_graph[map_lookup.first->second].insert(i + cell_offset);

        // Save memory and search time by erasing
        facet_cell_map.erase(map_lookup.first);
      }
    }
  }
}
//-----------------------------------------------------------------------------
void GraphBuilder::compute_nonlocal_dual_graph(
  const MPI_Comm mpi_comm,
  const LocalMeshData& mesh_data,
  std::vector<std::set<std::size_t>>& local_graph,
  FacetCellMap& facet_cell_map,
  std::set<std::size_t>& ghost_vertices)
{
  Timer timer("Compute non-local part of mesh dual graph");

  // At this stage facet_cell map only contains facets->cells with
  // edge facets either interprocess or external boundaries

  const std::size_t tdim = mesh_data.tdim;
  std::unique_ptr<CellType> cell_type(CellType::create(mesh_data.cell_type));

  // List of cell vertices
  const boost::multi_array<std::size_t, 2>& cell_vertices
    = mesh_data.cell_vertices;
  const std::size_t num_local_cells = mesh_data.global_cell_indices.size();
  const std::size_t num_vertices_per_cell = cell_type->num_entities(0);
  const std::size_t num_vertices_per_facet = cell_type->num_vertices(tdim - 1);

  dolfin_assert(num_local_cells == cell_vertices.shape()[0]);
  dolfin_assert(num_vertices_per_cell == cell_vertices.shape()[1]);

  // Compute local edges (cell-cell connections) using global
  // (internal to this function, not the user numbering) numbering

  // Get offset for this process
  const std::size_t offset = MPI::global_offset(mpi_comm, num_local_cells,
                                                true);
  const std::size_t num_processes = MPI::size(mpi_comm);

  // Send facet-cell map to intermediary match-making processes
  std::vector<std::vector<std::size_t>> send_buffer(num_processes);
  std::vector<std::vector<std::size_t>> received_buffer(num_processes);

  // Pack map data and send to match-maker process
  for (auto &it : facet_cell_map)
  {
    // FIXME: Could use a better index? First vertex is slightly
    //        skewed towards low values - may not be important

    // Use first vertex of facet to partition into blocks
    std::size_t dest_proc = MPI::index_owner(mpi_comm, (it.first)[0],
                                             mesh_data.num_global_vertices);

    // Pack map into vectors to send
    send_buffer[dest_proc].insert(send_buffer[dest_proc].end(),
                                  it.first.begin(), it.first.end());

    // Add offset to cell numbers sent off process
    send_buffer[dest_proc].push_back(it.second + offset);
  }

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
  for (std::size_t p = 0; p < num_processes; ++p)
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
  for (std::size_t p = 0; p < received_buffer.size(); ++p)
  {
    const std::vector<std::size_t>& cell_list = received_buffer[p];
    for (std::size_t i = 0; i < cell_list.size(); i += 2)
    {
      dolfin_assert(cell_list[i] >= offset);
      dolfin_assert(cell_list[i] - offset < local_graph.size());

      local_graph[cell_list[i] - offset].insert(cell_list[i + 1]);
      ghost_vertices.insert(cell_list[i + 1]);
    }
  }
}
//-----------------------------------------------------------------------------
