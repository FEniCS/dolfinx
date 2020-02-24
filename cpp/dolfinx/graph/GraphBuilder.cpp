// Copyright (C) 2010-2013 Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "GraphBuilder.h"
#include <algorithm>
#include <boost/unordered_map.hpp>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/MPI.h>
#include <dolfinx/common/Timer.h>
#include <dolfinx/common/log.h>
#include <dolfinx/common/types.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/mesh/MeshIterator.h>
#include <dolfinx/mesh/cell_types.h>
#include <numeric>
#include <set>
#include <unordered_set>
#include <utility>
#include <vector>

using namespace dolfinx;

namespace
{

//-----------------------------------------------------------------------------
// Compute local part of the dual graph, and return return (local_graph,
// facet_cell_map, number of local edges in the graph (undirected)
template <int N>
std::tuple<std::vector<std::vector<std::size_t>>,
           std::vector<std::pair<std::vector<std::size_t>, std::int32_t>>,
           std::int32_t>
compute_local_dual_graph_keyed(
    const MPI_Comm mpi_comm,
    const Eigen::Array<std::int64_t, Eigen::Dynamic, Eigen::Dynamic,
                       Eigen::RowMajor>& cell_vertices,
    const mesh::CellType cell_type)
{
  common::Timer timer("Compute local part of mesh dual graph");

  const std::int8_t tdim = mesh::cell_dim(cell_type);
  const std::int32_t num_local_cells = cell_vertices.rows();
  const std::int8_t num_facets_per_cell
      = mesh::cell_num_entities(cell_type, tdim - 1);
  const std::int8_t num_vertices_per_facet
      = mesh::num_cell_vertices(mesh::cell_entity_type(cell_type, tdim - 1));

  assert(N == num_vertices_per_facet);
  assert(num_local_cells == (int)cell_vertices.rows());
  //  assert(num_vertices_per_cell == (int)cell_vertices.cols());

  std::vector<std::vector<std::size_t>> local_graph(num_local_cells);
  std::vector<std::pair<std::vector<std::size_t>, std::int32_t>> facet_cell_map;

  // Compute local edges (cell-cell connections) using global
  // (internal to this function, not the user numbering) numbering

  // Get offset for this process
  const std::int64_t cell_offset
      = dolfinx::MPI::global_offset(mpi_comm, num_local_cells, true);

  // Create map from cell vertices to entity vertices
  const Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      facet_vertices = mesh::get_entity_vertices(cell_type, tdim - 1);

  // Vector-of-arrays data structure, which is considerably faster than
  // vector-of-vectors
  std::vector<std::pair<std::array<std::int32_t, N>, std::int32_t>> facets(
      num_facets_per_cell * num_local_cells);

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
        facet[k] = cell_vertices(i, facet_vertices(j, k));

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
      facet_cell_map.push_back(
          {std::vector<std::size_t>(facet0.begin(), facet0.end()),
           cell_index0});
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
                                                       facets[k].first.end()),
                              cell_index});
  }

  return std::tuple(std::move(local_graph), std::move(facet_cell_map),
                    num_local_edges);
}
//-----------------------------------------------------------------------------
// Build nonlocal part of dual graph for mesh and return number of
// non-local edges. Note: GraphBuilder::compute_local_dual_graph should
// be called before this function is called. Returns (ghost vertices,
// num_nonlocal_edges)
std::pair<std::int32_t, std::int32_t> compute_nonlocal_dual_graph(
    const MPI_Comm mpi_comm,
    const Eigen::Ref<const Eigen::Array<std::int64_t, Eigen::Dynamic,
                                        Eigen::Dynamic, Eigen::RowMajor>>&
        cell_vertices,
    const mesh::CellType cell_type,
    const graph::GraphBuilder::FacetCellMap& facet_cell_map,
    std::vector<std::vector<std::size_t>>& local_graph)
{
  LOG(INFO) << "Build nonlocal part of mesh dual graph";
  common::Timer timer("Compute non-local part of mesh dual graph");

  // Get number of MPI processes, and return if mesh is not distributed
  const int num_processes = dolfinx::MPI::size(mpi_comm);
  if (num_processes == 1)
    return std::pair(0, 0);

  // At this stage facet_cell map only contains facets->cells with
  // edge facets either interprocess or external boundaries

  const int tdim = mesh::cell_dim(cell_type);

  // List of cell vertices
  const std::int32_t num_local_cells = cell_vertices.rows();
  const std::int8_t num_vertices_per_facet
      = mesh::num_cell_vertices(mesh::cell_entity_type(cell_type, tdim - 1));

  assert(num_local_cells == (int)cell_vertices.rows());
  //  assert(num_vertices_per_cell == (int)cell_vertices.cols());

  // Compute local edges (cell-cell connections) using global
  // (internal to this function, not the user numbering) numbering

  // Get offset for this process
  const std::int64_t offset
      = dolfinx::MPI::global_offset(mpi_comm, num_local_cells, true);

  // Get global range of vertex indices
  const std::int64_t num_global_vertices
      = dolfinx::MPI::max(
            mpi_comm, (cell_vertices.rows() > 0) ? cell_vertices.maxCoeff() : 0)
        + 1;

  // Send facet-cell map to intermediary match-making processes
  std::vector<std::vector<std::size_t>> send_buffer(num_processes);
  std::vector<std::vector<std::size_t>> received_buffer(num_processes);

  // Pack map data and send to match-maker process
  for (auto& it : facet_cell_map)
  {
    // FIXME: Could use a better index? First vertex is slightly
    //        skewed towards low values - may not be important

    // Use first vertex of facet to partition into blocks
    const int dest_proc = dolfinx::MPI::index_owner(mpi_comm, (it.first)[0],
                                                    num_global_vertices);

    // Pack map into vectors to send
    send_buffer[dest_proc].insert(send_buffer[dest_proc].end(),
                                  it.first.begin(), it.first.end());

    // Add offset to cell numbers sent off process
    send_buffer[dest_proc].push_back(it.second + offset);
  }

  // Send data
  dolfinx::MPI::all_to_all(mpi_comm, send_buffer, received_buffer);

  // Clear send buffer
  send_buffer = std::vector<std::vector<std::size_t>>(num_processes);

  // Map to connect processes and cells, using facet as key
  typedef boost::unordered_map<std::vector<std::size_t>,
                               std::pair<std::size_t, std::size_t>>
      MatchMap;
  MatchMap matchmap;

  // Look for matches to send back to other processes
  std::pair<std::vector<std::size_t>, std::pair<std::size_t, std::size_t>> key;
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
  std::vector<std::size_t> cell_list;
  dolfinx::MPI::all_to_all(mpi_comm, send_buffer, cell_list);

  // Ghost nodes
  std::set<std::int64_t> ghost_nodes;

  // Insert connected cells into local map
  std::int32_t num_nonlocal_edges = 0;
  for (std::size_t i = 0; i < cell_list.size(); i += 2)
  {
    assert((std::int64_t)cell_list[i] >= offset);
    assert((std::int64_t)(cell_list[i] - offset)
           < (std::int64_t)local_graph.size());

    auto& edges = local_graph[cell_list[i] - offset];
    auto it = std::find(edges.begin(), edges.end(), cell_list[i + 1]);
    if (it == local_graph[cell_list[i] - offset].end())
    {
      edges.push_back(cell_list[i + 1]);
      ++num_nonlocal_edges;
    }
    ghost_nodes.insert(cell_list[i + 1]);
  }

  return std::pair(ghost_nodes.size(), num_nonlocal_edges);
}
//-----------------------------------------------------------------------------

} // namespace

//-----------------------------------------------------------------------------
dolfinx::graph::Graph
dolfinx::graph::GraphBuilder::local_graph(const mesh::Mesh& mesh,
                                          const fem::DofMap& dofmap0,
                                          const fem::DofMap& dofmap1)
{
  common::Timer timer("Build local sparsity graph from dofmaps");

  assert(dofmap0.element_dof_layout);
  assert(dofmap1.element_dof_layout);
  if (dofmap0.element_dof_layout->is_view()
      or dofmap1.element_dof_layout->is_view())
  {
    throw std::runtime_error("Graph building not support for dofmap views.");
  }

  // Create empty graph
  assert(dofmap0.index_map);
  const std::int32_t n
      = dofmap0.index_map->size_local() + dofmap0.index_map->num_ghosts();
  Graph graph(n);

  // Build graph
  const int tdim = mesh.topology().dim();
  auto map = mesh.topology().index_map(tdim);
  assert(map);
  assert(map->block_size() == 1);
  const int num_cells = map->size_local() + map->num_ghosts();
  for (int c = 0; c < num_cells; ++c)
  {
    auto dofs0 = dofmap0.cell_dofs(c);
    auto dofs1 = dofmap1.cell_dofs(c);
    for (Eigen::Index i = 0; i < dofs0.size(); ++i)
    {
      for (Eigen::Index j = 0; j < dofs1.size(); ++j)
        if (dofs0[i] != dofs1[j])
          graph[dofs0[i]].insert(dofs1[j]);
    }
  }

  return graph;
}
//-----------------------------------------------------------------------------
dolfinx::graph::Graph dolfinx::graph::GraphBuilder::local_graph(
    const mesh::Mesh& mesh, const std::vector<std::size_t>& coloring_type)
{
  // Initialise mesh
  for (std::size_t i = 0; i < coloring_type.size(); ++i)
    mesh.create_entities(coloring_type[i]);
  for (std::size_t i = 1; i < coloring_type.size(); ++i)
    mesh.create_connectivity(coloring_type[i - 1], coloring_type[i]);

  // Check coloring type
  assert(coloring_type.size() >= 2);
  assert(coloring_type.front() == coloring_type.back());

  // Create graph
  const std::size_t num_vertices = mesh.num_entities(coloring_type[0]);
  Graph graph(num_vertices);

  // Build graph
  auto map = mesh.topology().index_map(coloring_type[0]);
  assert(map);
  assert(map->block_size() == 1);
  const int num_entities = map->size_local() + map->num_ghosts();
  for (int e = 0; e < num_entities; ++e)
  {
    std::unordered_set<std::size_t> entity_list0;
    std::unordered_set<std::size_t> entity_list1;
    entity_list0.insert(e);

    // Build list of entities, moving between levels
    for (std::size_t level = 1; level < coloring_type.size(); ++level)
    {
      for (auto entity_index : entity_list0)
      {
        auto connectivity = mesh.topology().connectivity(
            coloring_type[level - 1], coloring_type[level]);
        assert(connectivity);
        auto links = connectivity->links(entity_index);
        for (int neighbor = 0; neighbor < links.rows(); ++neighbor)
          entity_list1.insert(links[neighbor]);
      }
      entity_list0 = entity_list1;
      entity_list1.clear();
    }

    // Add edges to graph
    graph[e].insert(entity_list0.begin(), entity_list0.end());
  }

  return graph;
}
//-----------------------------------------------------------------------------
dolfinx::graph::Graph
dolfinx::graph::GraphBuilder::local_graph(const mesh::Mesh& mesh,
                                          std::size_t dim0, std::size_t dim1)
{
  mesh.create_entities(dim0);
  mesh.create_entities(dim1);
  mesh.create_connectivity(dim0, dim1);
  mesh.create_connectivity(dim1, dim0);

  // Create graph
  const std::size_t num_vertices = mesh.num_entities(dim0);
  Graph graph(num_vertices);

  auto e0_to_e1 = mesh.topology().connectivity(dim0, dim1);
  assert(e0_to_e1);
  auto e1_to_e0 = mesh.topology().connectivity(dim1, dim0);
  assert(e1_to_e0);

  // Build graph
  auto map = mesh.topology().index_map(dim0);
  assert(map);
  assert(map->block_size() == 1);
  const int num_entities = map->size_local() + map->num_ghosts();
  for (int colored_entity = 0; colored_entity < num_entities; ++colored_entity)
  {
    auto links1 = e0_to_e1->links(colored_entity);
    for (int entity = 0; entity < links1.rows(); ++entity)
    {
      auto links0 = e1_to_e0->links(links1(entity));
      for (int neighbor = 0; neighbor < links0.rows(); ++neighbor)
      {
        if (colored_entity != links0(neighbor))
          graph[colored_entity].insert(links0(neighbor));
      }
    }
  }

  return graph;
}
//-----------------------------------------------------------------------------
std::pair<std::vector<std::vector<std::size_t>>, std::array<std::int32_t, 3>>
graph::GraphBuilder::compute_dual_graph(
    const MPI_Comm mpi_comm,
    const Eigen::Ref<const Eigen::Array<std::int64_t, Eigen::Dynamic,
                                        Eigen::Dynamic, Eigen::RowMajor>>&
        cell_vertices,
    const mesh::CellType cell_type)
{
  LOG(INFO) << "Build mesh dual graph";

  // Compute local part of dual graph
  auto [local_graph, facet_cell_map, num_local_edges]
      = graph::GraphBuilder::compute_local_dual_graph(mpi_comm, cell_vertices,
                                                      cell_type);

  // Compute nonlocal part
  const auto [num_ghost_nodes, num_nonlocal_edges]
      = compute_nonlocal_dual_graph(mpi_comm, cell_vertices, cell_type,
                                    facet_cell_map, local_graph);

  // Shrink to fit
  local_graph.shrink_to_fit();

  return {std::move(local_graph),
          {num_ghost_nodes, num_local_edges, num_nonlocal_edges}};
}
//-----------------------------------------------------------------------------
std::tuple<std::vector<std::vector<std::size_t>>,
           std::vector<std::pair<std::vector<std::size_t>, std::int32_t>>,
           std::int32_t>
dolfinx::graph::GraphBuilder::compute_local_dual_graph(
    const MPI_Comm mpi_comm,
    const Eigen::Ref<const Eigen::Array<std::int64_t, Eigen::Dynamic,
                                        Eigen::Dynamic, Eigen::RowMajor>>&
        cell_vertices,
    const mesh::CellType cell_type)
{
  LOG(INFO) << "Build local part of mesh dual graph";

  const std::int8_t tdim = mesh::cell_dim(cell_type);
  const std::int8_t num_entity_vertices
      = mesh::num_cell_vertices(mesh::cell_entity_type(cell_type, tdim - 1));

  switch (num_entity_vertices)
  {
  case 1:
    return compute_local_dual_graph_keyed<1>(mpi_comm, cell_vertices,
                                             cell_type);
  case 2:
    return compute_local_dual_graph_keyed<2>(mpi_comm, cell_vertices,
                                             cell_type);
  case 3:
    return compute_local_dual_graph_keyed<3>(mpi_comm, cell_vertices,
                                             cell_type);
  case 4:
    return compute_local_dual_graph_keyed<4>(mpi_comm, cell_vertices,
                                             cell_type);
  default:
    throw std::runtime_error(
        "Cannot compute local part of dual graph. Entities with "
        + std::to_string(num_entity_vertices) + " vertices not supported");
  }
}
//-----------------------------------------------------------------------------
