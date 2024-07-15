// Copyright (C) 2014-2018 Chris Richardson
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "utils.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <dolfinx/common/MPI.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/Topology.h>
#include <dolfinx/mesh/utils.h>
#include <span>
#include <tuple>
#include <utility>
#include <vector>

#pragma once

/// @brief Plaza mesh refinement.
///
/// Functions for the refinement method described in Plaza and Carey
/// "Local refinement of simplicial grids based on the skeleton",
/// Applied Numerical Mathematics 32 (2000), 195-218.
namespace dolfinx::refinement::plaza
{

/// @brief Options for data to compute during mesh refinement.
enum class Option : int
{
  none = 0, /*!< No extra data */
  parent_cell
  = 1, /*!< Compute list with the parent cell index for each new cell  */
  parent_facet
  = 2, /*!< Compute list of the cell-local facet indices in the parent cell of
          each facet in each new cell (or -1 if no match) */
  parent_cell_and_facet = 3 /*!< Both cell and facet parent data */
};

namespace impl
{
/// Computes the parent-child facet relationship
/// @param simplex_set - index into indices for each child cell
/// @return mapping from child to parent facets, using cell-local index
template <int tdim>
auto compute_parent_facets(std::span<const std::int32_t> simplex_set)
{
  static_assert(tdim == 2 or tdim == 3);
  assert(simplex_set.size() % (tdim + 1) == 0);
  using parent_facet_t
      = std::conditional_t<tdim == 2, std::array<std::int8_t, 12>,
                           std::array<std::int8_t, 32>>;
  parent_facet_t parent_facet;
  parent_facet.fill(-1);
  assert(simplex_set.size() <= parent_facet.size());

  // Index lookups in 'indices' for the child vertices that occur on
  // each parent facet in 2D and 3D. In 2D each edge has 3 child
  // vertices, and in 3D each triangular facet has six child vertices.
  constexpr std::array<std::array<int, 3>, 3> facet_table_2d{
      {{1, 2, 3}, {0, 2, 4}, {0, 1, 5}}};

  constexpr std::array<std::array<int, 6>, 4> facet_table_3d{
      {{1, 2, 3, 4, 5, 6},
       {0, 2, 3, 4, 7, 8},
       {0, 1, 3, 5, 7, 9},
       {0, 1, 2, 6, 8, 9}}};

  const int ncells = simplex_set.size() / (tdim + 1);
  for (int fpi = 0; fpi < (tdim + 1); ++fpi)
  {
    // For each child cell, consider all facets
    for (int cc = 0; cc < ncells; ++cc)
    {
      for (int fci = 0; fci < (tdim + 1); ++fci)
      {
        // Indices of all vertices on child facet, sorted
        std::array<int, tdim> cf, set_output;

        int num_common_vertices;
        if constexpr (tdim == 2)
        {
          for (int j = 0; j < tdim; ++j)
            cf[j] = simplex_set[cc * 3 + facet_table_2d[fci][j]];

          std::ranges::sort(cf);
          auto [last1, last2, it_last] = std::ranges::set_intersection(
              facet_table_2d[fpi], cf, set_output.begin());
          num_common_vertices = std::distance(set_output.begin(), it_last);
        }
        else
        {
          for (int j = 0; j < tdim; ++j)
            cf[j] = simplex_set[cc * 4 + facet_table_3d[fci][j]];

          std::ranges::sort(cf);
          auto [last1, last2, it_last] = std::ranges::set_intersection(
              facet_table_3d[fpi], cf, set_output.begin());
          num_common_vertices = std::distance(set_output.begin(), it_last);
        }

        if (num_common_vertices == tdim)
        {
          assert(parent_facet[cc * (tdim + 1) + fci] == -1);
          // Child facet "fci" of cell cc, lies on parent facet "fpi"
          parent_facet[cc * (tdim + 1) + fci] = fpi;
        }
      }
    }
  }

  return parent_facet;
}

/// Get the subdivision of an original simplex into smaller simplices,
/// for a given set of marked edges, and the longest edge of each facet
/// (cell local indexing). A flag indicates if a uniform subdivision is
/// preferable in 2D.
///
/// @param[in] indices Vector containing the global indices for the original
/// vertices and potential new vertices at each edge. Size (num_vertices +
/// num_edges). If an edge is not refined its corresponding entry is -1
/// @param[in] longest_edge Vector indicating the longest edge for each
/// triangle in the cell. For triangular cells (2D) there is only one value,
/// and for tetrahedra (3D) there are four values, one for each facet. The
/// values give the local edge indices of the cell.
/// @param[in] tdim Topological dimension (2 or 3)
/// @param[in] uniform Make a "uniform" subdivision with all triangles being
/// similar shape
/// @return
std::pair<std::array<std::int32_t, 32>, std::size_t>
get_simplices(std::span<const std::int64_t> indices,
              std::span<const std::int32_t> longest_edge, int tdim,
              bool uniform);

/// Propagate edge markers according to rules (longest edge of each
/// face must be marked, if any edge of face is marked)
void enforce_rules(MPI_Comm comm, const graph::AdjacencyList<int>& shared_edges,
                   std::span<std::int8_t> marked_edges,
                   const mesh::Topology& topology,
                   std::span<const std::int32_t> long_edge);

/// @brief Get the longest edge of each face (using local mesh index)
///
/// @note Edge ratio ok returns an array in 2D, where it checks if the ratio
/// between the shortest and longest edge of a cell is bigger than sqrt(2)/2. In
/// 3D an empty array is returned
///
/// @param[in] mesh The mesh
/// @return A tuple (longest edge, edge ratio ok) where longest edge gives the
/// local index of the longest edge for each face.
template <std::floating_point T>
std::pair<std::vector<std::int32_t>, std::vector<std::int8_t>>
face_long_edge(const mesh::Mesh<T>& mesh)
{
  const int tdim = mesh.topology()->dim();
  // FIXME: cleanup these calls? Some of the happen internally again.
  mesh.topology_mutable()->create_entities(1);
  mesh.topology_mutable()->create_entities(2);
  mesh.topology_mutable()->create_connectivity(2, 1);
  mesh.topology_mutable()->create_connectivity(1, tdim);
  mesh.topology_mutable()->create_connectivity(tdim, 2);

  std::int64_t num_faces = mesh.topology()->index_map(2)->size_local()
                           + mesh.topology()->index_map(2)->num_ghosts();

  // Storage for face-local index of longest edge
  std::vector<std::int32_t> long_edge(num_faces);
  std::vector<std::int8_t> edge_ratio_ok;

  // Check mesh face quality (may be used in 2D to switch to "uniform"
  // refinement)
  const T min_ratio = std::sqrt(2.0) / 2.0;
  if (tdim == 2)
    edge_ratio_ok.resize(num_faces);

  auto x_dofmap = mesh.geometry().dofmap();

  auto c_to_v = mesh.topology()->connectivity(tdim, 0);
  assert(c_to_v);
  auto e_to_c = mesh.topology()->connectivity(1, tdim);
  assert(e_to_c);
  auto e_to_v = mesh.topology()->connectivity(1, 0);
  assert(e_to_v);

  // Store all edge lengths in Mesh to save recalculating for each Face
  auto map_e = mesh.topology()->index_map(1);
  assert(map_e);
  std::vector<T> edge_length(map_e->size_local() + map_e->num_ghosts());
  for (std::size_t e = 0; e < edge_length.size(); ++e)
  {
    // Get first attached cell
    auto cells = e_to_c->links(e);
    assert(!cells.empty());
    auto cell_vertices = c_to_v->links(cells.front());
    auto edge_vertices = e_to_v->links(e);

    // Find local index of edge vertices in the cell geometry map
    auto it0 = std::find(cell_vertices.begin(), cell_vertices.end(),
                         edge_vertices[0]);
    assert(it0 != cell_vertices.end());
    const std::size_t local0 = std::distance(cell_vertices.begin(), it0);
    auto it1 = std::find(cell_vertices.begin(), cell_vertices.end(),
                         edge_vertices[1]);
    assert(it1 != cell_vertices.end());
    const std::size_t local1 = std::distance(cell_vertices.begin(), it1);

    auto x_dofs = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
        x_dofmap, cells.front(), MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
    std::span<const T, 3> x0(mesh.geometry().x().data() + 3 * x_dofs[local0],
                             3);
    std::span<const T, 3> x1(mesh.geometry().x().data() + 3 * x_dofs[local1],
                             3);

    // Compute length of edge between vertex x0 and x1
    edge_length[e] = std::sqrt(std::transform_reduce(
        x0.begin(), x0.end(), x1.begin(), 0.0, std::plus<>(),
        [](auto x0, auto x1) { return (x0 - x1) * (x0 - x1); }));
  }

  // Get longest edge of each face
  auto f_to_v = mesh.topology()->connectivity(2, 0);
  assert(f_to_v);
  auto f_to_e = mesh.topology()->connectivity(2, 1);
  assert(f_to_e);
  const std::vector global_indices
      = mesh.topology()->index_map(0)->global_indices();
  for (int f = 0; f < f_to_v->num_nodes(); ++f)
  {
    auto face_edges = f_to_e->links(f);

    std::int32_t imax = 0;
    T max_len = 0.0;
    T min_len = std::numeric_limits<T>::max();

    for (int i = 0; i < 3; ++i)
    {
      const T e_len = edge_length[face_edges[i]];
      min_len = std::min(e_len, min_len);
      if (e_len > max_len)
      {
        max_len = e_len;
        imax = i;
      }
      else if (tdim == 3 and e_len == max_len)
      {
        // If edges are the same length, compare global index of
        // opposite vertex.  Only important so that tetrahedral faces
        // have a matching refinement pattern across processes.
        auto vertices = f_to_v->links(f);
        const int vmax = vertices[imax];
        const int vi = vertices[i];
        if (global_indices[vi] > global_indices[vmax])
          imax = i;
      }
    }

    // Only save edge ratio in 2D
    if (tdim == 2)
      edge_ratio_ok[f] = (min_len / max_len >= min_ratio);

    long_edge[f] = face_edges[imax];
  }

  return std::pair(std::move(long_edge), std::move(edge_ratio_ok));
}

/// @brief Convenient interface for both uniform and marker refinement
///
/// @note The parent facet map gives you the map from a cell given by parent
/// cell map to the local index (relative to the cell), e.g. the i-th entry of
/// parent facets relates to the local facet index of the i-th entry parent
/// cell.
///
/// @param[in] neighbor_comm Neighbourhood communciator scattering owned edges
/// to processes with ghosts
/// @param[in] marked_edges A marker for all edges on the process (local +
/// ghosts) indicating if an edge should be refined
/// @param[in] shared_edges For each local edge on a process map to ghost
/// processes
/// @param[in] mesh The mesh
/// @param[in] long_edge A map from each face to its longest edge. Index is
/// local to the process.
/// @param[in] edge_ratio_ok For each face in a 2D mesh this error contains a
/// marker indicating if the ratio between smallest and largest edge is bigger
/// than sqrt(2)/2
/// @param[in] option Option to compute additional information relating refined
/// and original mesh entities
/// @return (0) The new mesh topology, (1) the new flattened mesh geometry, (3)
/// Shape of the new geometry_shape, (4) Map from new cells to parent cells
/// and (5) map from refined facets to parent facets.
template <std::floating_point T>
std::tuple<graph::AdjacencyList<std::int64_t>, std::vector<T>,
           std::array<std::size_t, 2>, std::vector<std::int32_t>,
           std::vector<std::int8_t>>
compute_refinement(MPI_Comm neighbor_comm,
                   std::span<const std::int8_t> marked_edges,
                   const graph::AdjacencyList<int>& shared_edges,
                   const mesh::Mesh<T>& mesh,
                   std::span<const std::int32_t> long_edge,
                   std::span<const std::int8_t> edge_ratio_ok,
                   plaza::Option option)
{
  int tdim = mesh.topology()->dim();
  int num_cell_edges = tdim * 3 - 3;
  int num_cell_vertices = tdim + 1;
  bool compute_facets = option == plaza::Option::parent_facet
                        or option == plaza::Option::parent_cell_and_facet;
  bool compute_parent_cell = option == plaza::Option::parent_cell
                             or option == plaza::Option::parent_cell_and_facet;

  // Make new vertices in parallel
  const auto [new_vertex_map, new_vertex_coords, xshape]
      = create_new_vertices(neighbor_comm, shared_edges, mesh, marked_edges);

  std::vector<std::int32_t> parent_cell;
  std::vector<std::int8_t> parent_facet;
  std::vector<std::int64_t> indices(num_cell_vertices + num_cell_edges);
  std::vector<std::int32_t> simplex_set;

  auto map_c = mesh.topology()->index_map(tdim);
  assert(map_c);
  auto c_to_v = mesh.topology()->connectivity(tdim, 0);
  assert(c_to_v);
  auto c_to_e = mesh.topology()->connectivity(tdim, 1);
  assert(c_to_e);
  auto c_to_f = mesh.topology()->connectivity(tdim, 2);
  assert(c_to_f);

  std::int32_t num_new_vertices_local = std::count(
      marked_edges.begin(),
      marked_edges.begin() + mesh.topology()->index_map(1)->size_local(), true);

  std::vector<std::int64_t> global_indices
      = adjust_indices(*mesh.topology()->index_map(0), num_new_vertices_local);

  const std::int32_t num_cells = map_c->size_local();

  // Iterate over all cells, and refine if cell has a marked edge
  std::vector<std::int64_t> cell_topology;
  for (int c = 0; c < num_cells; ++c)
  {
    // Create vector of indices in the order [vertices][edges], 3+3 in
    // 2D, 4+6 in 3D

    // Copy vertices
    auto vertices = c_to_v->links(c);
    for (std::size_t v = 0; v < vertices.size(); ++v)
      indices[v] = global_indices[vertices[v]];

    // Get cell-local indices of marked edges
    auto edges = c_to_e->links(c);
    bool no_edge_marked = true;
    for (std::size_t ei = 0; ei < edges.size(); ++ei)
    {
      if (marked_edges[edges[ei]])
      {
        no_edge_marked = false;
        auto it = new_vertex_map.find(edges[ei]);
        assert(it != new_vertex_map.end());
        indices[num_cell_vertices + ei] = it->second;
      }
      else
        indices[num_cell_vertices + ei] = -1;
    }

    if (no_edge_marked)
    {
      // Copy over existing cell to new topology
      for (auto v : vertices)
        cell_topology.push_back(global_indices[v]);

      if (compute_parent_cell)
        parent_cell.push_back(c);

      if (compute_facets)
      {
        if (tdim == 3)
          parent_facet.insert(parent_facet.end(), {0, 1, 2, 3});
        else
          parent_facet.insert(parent_facet.end(), {0, 1, 2});
      }
    }
    else
    {
      // Need longest edges of each face in cell local indexing. NB in
      // 2D the face is the cell itself, and there is just one entry.
      std::vector<std::int32_t> longest_edge;
      for (auto f : c_to_f->links(c))
        longest_edge.push_back(long_edge[f]);

      // Convert to cell local index
      for (std::int32_t& p : longest_edge)
      {
        for (std::size_t ej = 0; ej < edges.size(); ++ej)
        {
          if (p == edges[ej])
          {
            p = ej;
            break;
          }
        }
      }

      const bool uniform = (tdim == 2) ? edge_ratio_ok[c] : false;
      const auto [simplex_set_b, simplex_set_size]
          = get_simplices(indices, longest_edge, tdim, uniform);
      std::span<const std::int32_t> simplex_set(simplex_set_b.data(),
                                                simplex_set_size);

      // Save parent index
      const std::int32_t ncells = simplex_set.size() / num_cell_vertices;
      if (compute_parent_cell)
      {
        for (std::int32_t i = 0; i < ncells; ++i)
          parent_cell.push_back(c);
      }

      if (compute_facets)
      {
        if (tdim == 3)
        {
          auto npf = compute_parent_facets<3>(simplex_set);
          parent_facet.insert(parent_facet.end(), npf.begin(),
                              std::next(npf.begin(), simplex_set.size()));
        }
        else
        {
          auto npf = compute_parent_facets<2>(simplex_set);
          parent_facet.insert(parent_facet.end(), npf.begin(),
                              std::next(npf.begin(), simplex_set.size()));
        }
      }

      // Convert from cell local index to mesh index and add to cells
      for (std::int32_t v : simplex_set)
        cell_topology.push_back(indices[v]);
    }
  }

  assert(cell_topology.size() % num_cell_vertices == 0);
  std::vector<std::int32_t> offsets(
      cell_topology.size() / num_cell_vertices + 1, 0);
  for (std::size_t i = 0; i < offsets.size() - 1; ++i)
    offsets[i + 1] = offsets[i] + num_cell_vertices;
  graph::AdjacencyList cell_adj(std::move(cell_topology), std::move(offsets));

  return {std::move(cell_adj), std::move(new_vertex_coords), xshape,
          std::move(parent_cell), std::move(parent_facet)};
}
} // namespace impl

/// @brief Uniform refine, optionally redistributing and optionally
/// calculating the parent-child relationships`.
///
/// @param[in] mesh Input mesh to be refined
/// @param[in] redistribute Flag to call the mesh partitioner to
/// redistribute after refinement
/// @param[in] option Control the computation of parent facets, parent
/// cells. If an option is unselected, an empty list is returned.
/// @return Refined mesh and optional parent cell index, parent facet
/// indices
template <std::floating_point T>
std::tuple<mesh::Mesh<T>, std::vector<std::int32_t>, std::vector<std::int8_t>>
refine(const mesh::Mesh<T>& mesh, bool redistribute, Option option)
{
  auto [cell_adj, new_coords, xshape, parent_cell, parent_facet]
      = compute_refinement_data(mesh, option);

  if (dolfinx::MPI::size(mesh.comm()) == 1)
  {
    return {mesh::create_mesh(mesh.comm(), cell_adj.array(),
                              mesh.geometry().cmap(), new_coords, xshape,
                              mesh::GhostMode::none),
            std::move(parent_cell), std::move(parent_facet)};
  }
  else
  {
    std::shared_ptr<const common::IndexMap> map_c
        = mesh.topology()->index_map(mesh.topology()->dim());
    const int num_ghost_cells = map_c->num_ghosts();
    // Check if mesh has ghost cells on any rank
    // FIXME: this is not a robust test. Should be user option.
    int max_ghost_cells = 0;
    MPI_Allreduce(&num_ghost_cells, &max_ghost_cells, 1, MPI_INT, MPI_MAX,
                  mesh.comm());

    // Build mesh
    const mesh::GhostMode ghost_mode = max_ghost_cells == 0
                                           ? mesh::GhostMode::none
                                           : mesh::GhostMode::shared_facet;
    return {partition<T>(mesh, cell_adj, std::span(new_coords), xshape,
                         redistribute, ghost_mode),
            std::move(parent_cell), std::move(parent_facet)};
  }
}

/// @brief Refine with markers, optionally redistributing, and
/// optionally calculating the parent-child relationships.
///
/// @param[in] mesh Input mesh to be refined
/// @param[in] edges Indices of the edges that should be split by this
/// refinement
/// @param[in] redistribute Flag to call the Mesh Partitioner to
/// redistribute after refinement
/// @param[in] option Control the computation of parent facets, parent
/// cells. If an option is unselected, an empty list is returned.
/// @return New Mesh and optional parent cell index, parent facet indices
template <std::floating_point T>
std::tuple<mesh::Mesh<T>, std::vector<std::int32_t>, std::vector<std::int8_t>>
refine(const mesh::Mesh<T>& mesh, std::span<const std::int32_t> edges,
       bool redistribute, Option option)
{
  auto [cell_adj, new_vertex_coords, xshape, parent_cell, parent_facet]
      = compute_refinement_data(mesh, edges, option);

  if (dolfinx::MPI::size(mesh.comm()) == 1)
  {
    return {mesh::create_mesh(mesh.comm(), cell_adj.array(),
                              mesh.geometry().cmap(), new_vertex_coords, xshape,
                              mesh::GhostMode::none),
            std::move(parent_cell), std::move(parent_facet)};
  }
  else
  {
    std::shared_ptr<const common::IndexMap> map_c
        = mesh.topology()->index_map(mesh.topology()->dim());
    const int num_ghost_cells = map_c->num_ghosts();
    // Check if mesh has ghost cells on any rank
    // FIXME: this is not a robust test. Should be user option.
    int max_ghost_cells = 0;
    MPI_Allreduce(&num_ghost_cells, &max_ghost_cells, 1, MPI_INT, MPI_MAX,
                  mesh.comm());

    // Build mesh
    const mesh::GhostMode ghost_mode = max_ghost_cells == 0
                                           ? mesh::GhostMode::none
                                           : mesh::GhostMode::shared_facet;

    return {partition<T>(mesh, cell_adj, new_vertex_coords, xshape,
                         redistribute, ghost_mode),
            std::move(parent_cell), std::move(parent_facet)};
  }
}

/// @brief Refine mesh returning new mesh data.
///
/// @param[in] mesh Input mesh to be refined
/// @param[in] option Control computation of parent facets and parent
/// cells. If an option is unselected, an empty list is returned.
/// @return New mesh data: cell topology, vertex coordinates, vertex
/// coordinates shape, and optional parent cell index, and parent facet
/// indices.
template <std::floating_point T>
std::tuple<graph::AdjacencyList<std::int64_t>, std::vector<T>,
           std::array<std::size_t, 2>, std::vector<std::int32_t>,
           std::vector<std::int8_t>>
compute_refinement_data(const mesh::Mesh<T>& mesh, Option option)
{
  common::Timer t0("PLAZA: refine");
  auto topology = mesh.topology();
  assert(topology);

  if (topology->cell_type() != mesh::CellType::triangle
      and topology->cell_type() != mesh::CellType::tetrahedron)
  {
    throw std::runtime_error("Cell type not supported");
  }

  auto map_e = topology->index_map(1);
  if (!map_e)
    throw std::runtime_error("Edges must be initialised");

  // Get sharing ranks for each edge
  graph::AdjacencyList<int> edge_ranks = map_e->index_to_dest_ranks();

  // Create unique list of ranks that share edges (owners of ghosts
  // plus ranks that ghost owned indices)
  std::vector<int> ranks(edge_ranks.array().begin(), edge_ranks.array().end());
  std::ranges::sort(ranks);
  ranks.erase(std::unique(ranks.begin(), ranks.end()), ranks.end());

  // Convert edge_ranks from global rank to to neighbourhood ranks
  std::ranges::transform(edge_ranks.array(), edge_ranks.array().begin(),
                         [&ranks](auto r)
                         {
                           auto it = std::ranges::lower_bound(ranks, r);
                           assert(it != ranks.end() and *it == r);
                           return std::distance(ranks.begin(), it);
                         });

  MPI_Comm comm;
  MPI_Dist_graph_create_adjacent(mesh.comm(), ranks.size(), ranks.data(),
                                 MPI_UNWEIGHTED, ranks.size(), ranks.data(),
                                 MPI_UNWEIGHTED, MPI_INFO_NULL, false, &comm);

  const auto [long_edge, edge_ratio_ok] = impl::face_long_edge(mesh);
  auto [cell_adj, new_vertex_coords, xshape, parent_cell, parent_facet]
      = impl::compute_refinement(
          comm,
          std::vector<std::int8_t>(map_e->size_local() + map_e->num_ghosts(),
                                   true),
          edge_ranks, mesh, long_edge, edge_ratio_ok, option);
  MPI_Comm_free(&comm);

  return {std::move(cell_adj), std::move(new_vertex_coords), xshape,
          std::move(parent_cell), std::move(parent_facet)};
}

/// Refine with markers returning new mesh data.
///
/// @param[in] mesh Input mesh to be refined
/// @param[in] edges Indices of the edges that should be split by this
/// refinement
/// @param[in] option Control the computation of parent facets, parent
/// cells. If an option is unselected, an empty list is returned.
/// @return New mesh data: cell topology, vertex coordinates and parent
/// cell index, and stored parent facet indices (if requested).
template <std::floating_point T>
std::tuple<graph::AdjacencyList<std::int64_t>, std::vector<T>,
           std::array<std::size_t, 2>, std::vector<std::int32_t>,
           std::vector<std::int8_t>>
compute_refinement_data(const mesh::Mesh<T>& mesh,
                        std::span<const std::int32_t> edges, Option option)
{
  common::Timer t0("PLAZA: refine");
  auto topology = mesh.topology();
  assert(topology);

  if (topology->cell_type() != mesh::CellType::triangle
      and topology->cell_type() != mesh::CellType::tetrahedron)
  {
    throw std::runtime_error("Cell type not supported");
  }

  auto map_e = topology->index_map(1);
  if (!map_e)
    throw std::runtime_error("Edges must be initialised");

  // Get sharing ranks for each edge
  graph::AdjacencyList<int> edge_ranks = map_e->index_to_dest_ranks();

  // Create unique list of ranks that share edges (owners of ghosts plus
  // ranks that ghost owned indices)
  std::vector<int> ranks(edge_ranks.array().begin(), edge_ranks.array().end());
  std::ranges::sort(ranks);
  ranks.erase(std::unique(ranks.begin(), ranks.end()), ranks.end());

  // Convert edge_ranks from global rank to to neighbourhood ranks
  std::ranges::transform(edge_ranks.array(), edge_ranks.array().begin(),
                         [&ranks](auto r)
                         {
                           auto it = std::ranges::lower_bound(ranks, r);
                           assert(it != ranks.end() and *it == r);
                           return std::distance(ranks.begin(), it);
                         });

  // Get number of neighbors
  std::vector<std::int8_t> marked_edges(
      map_e->size_local() + map_e->num_ghosts(), false);
  std::vector<std::vector<std::int32_t>> marked_for_update(ranks.size());
  for (auto edge : edges)
  {
    if (!marked_edges[edge])
    {
      marked_edges[edge] = true;

      // If it is a shared edge, add all sharing neighbors to update set
      for (int rank : edge_ranks.links(edge))
        marked_for_update[rank].push_back(edge);
    }
  }

  MPI_Comm comm;
  MPI_Dist_graph_create_adjacent(mesh.comm(), ranks.size(), ranks.data(),
                                 MPI_UNWEIGHTED, ranks.size(), ranks.data(),
                                 MPI_UNWEIGHTED, MPI_INFO_NULL, false, &comm);

  // Communicate any shared edges
  update_logical_edgefunction(comm, marked_for_update, marked_edges, *map_e);

  // Enforce rules about refinement (i.e. if any edge is marked in a
  // triangle, then the longest edge must also be marked).
  const auto [long_edge, edge_ratio_ok] = impl::face_long_edge(mesh);
  impl::enforce_rules(comm, edge_ranks, marked_edges, *topology, long_edge);

  auto [cell_adj, new_vertex_coords, xshape, parent_cell, parent_facet]
      = impl::compute_refinement(comm, marked_edges, edge_ranks, mesh,
                                 long_edge, edge_ratio_ok, option);
  MPI_Comm_free(&comm);

  return {std::move(cell_adj), std::move(new_vertex_coords), xshape,
          std::move(parent_cell), std::move(parent_facet)};
}
} // namespace dolfinx::refinement::plaza
