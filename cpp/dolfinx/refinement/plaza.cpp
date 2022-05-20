// Copyright (C) 2014-2022 Chris Richardson
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "plaza.h"
#include "utils.h"
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/Timer.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/MeshTags.h>
#include <dolfinx/mesh/Topology.h>
#include <limits>
#include <map>
#include <numeric>
#include <vector>
#include <xtensor/xtensor.hpp>

using namespace dolfinx;
using namespace dolfinx::refinement;

namespace
{

//-----------------------------------------------------------------------------
// Propagate edge markers according to rules (longest edge of each
// face must be marked, if any edge of face is marked)
void enforce_rules(MPI_Comm neighbor_comm,
                   const graph::AdjacencyList<int>& shared_edges,
                   std::vector<std::int8_t>& marked_edges,
                   const mesh::Mesh& mesh,
                   const std::vector<std::int32_t>& long_edge)
{
  common::Timer t0("PLAZA: Enforce rules");

  // Enforce rule, that if any edge of a face is marked, longest edge
  // must also be marked

  auto map_e = mesh.topology().index_map(1);
  assert(map_e);
  auto map_f = mesh.topology().index_map(2);
  assert(map_f);
  const std::int32_t num_faces = map_f->size_local() + map_f->num_ghosts();

  auto f_to_e = mesh.topology().connectivity(2, 1);
  assert(f_to_e);

  // Get number of neighbors
  int indegree(-1), outdegree(-2), weighted(-1);
  MPI_Dist_graph_neighbors_count(neighbor_comm, &indegree, &outdegree,
                                 &weighted);
  assert(indegree == outdegree);
  const int num_neighbors = indegree;
  std::vector<std::vector<std::int32_t>> marked_for_update(num_neighbors);

  std::int32_t update_count = 1;
  while (update_count > 0)
  {
    update_count = 0;
    update_logical_edgefunction(neighbor_comm, marked_for_update, marked_edges,
                                *map_e);
    for (int i = 0; i < num_neighbors; ++i)
      marked_for_update[i].clear();

    for (int f = 0; f < num_faces; ++f)
    {
      const std::int32_t long_e = long_edge[f];
      if (marked_edges[long_e])
        continue;

      bool any_marked = false;
      for (auto edge : f_to_e->links(f))
        any_marked = any_marked or marked_edges[edge];

      if (any_marked)
      {
        if (!marked_edges[long_e])
        {
          marked_edges[long_e] = true;

          // Add sharing neighbors to update set
          for (int rank : shared_edges.links(long_e))
            marked_for_update[rank].push_back(long_e);
        }
        ++update_count;
      }
    }

    const std::int32_t update_count_old = update_count;
    MPI_Allreduce(&update_count_old, &update_count, 1, MPI_INT32_T, MPI_SUM,
                  mesh.comm());
  }
}
//-----------------------------------------------------------------------------
// 2D version of subdivision allowing for uniform subdivision (flag)
std::vector<std::int32_t>
get_triangles(const std::vector<std::int64_t>& indices,
              const std::int32_t longest_edge, bool uniform)
{
  // v0 and v1 are at ends of longest_edge (e2) opposite vertex has same
  // index as longest_edge
  const std::int32_t v0 = (longest_edge + 1) % 3;
  const std::int32_t v1 = (longest_edge + 2) % 3;
  const std::int32_t v2 = longest_edge;
  const std::int32_t e0 = v0 + 3;
  const std::int32_t e1 = v1 + 3;
  const std::int32_t e2 = v2 + 3;

  // Longest edge must be marked
  assert(indices[e2] >= 0);

  // If all edges marked, consider uniform refinement
  if (uniform and indices[e0] >= 0 and indices[e1] >= 0)
    return {e0, e1, v2, e1, e2, v0, e2, e0, v1, e2, e1, e0};

  // Break each half of triangle into one or two sub-triangles
  std::vector<std::int32_t> tri_set;
  if (indices[e0] >= 0)
    tri_set = {e2, v2, e0, e2, e0, v1};
  else
    tri_set = {e2, v2, v1};

  if (indices[e1] >= 0)
  {
    tri_set.insert(tri_set.end(), {e2, v2, e1});
    tri_set.insert(tri_set.end(), {e2, e1, v0});
  }
  else
    tri_set.insert(tri_set.end(), {e2, v2, v0});

  return tri_set;
}
//-----------------------------------------------------------------------------
// 3D version of subdivision
std::vector<std::int32_t>
get_tetrahedra(const std::vector<std::int64_t>& indices,
               const std::vector<std::int32_t>& longest_edge)
{
  // Connectivity matrix for ten possible points (4 vertices + 6 edge
  // midpoints) ordered {v0, v1, v2, v3, e0, e1, e2, e3, e4, e5} Only need
  // upper triangle, but sometimes it is easier just to insert both entries
  // (j,i) and (i,j).
  bool conn[10][10] = {};

  // Edge connectivity to vertices (and by extension facets)
  static const std::int32_t edges[6][2]
      = {{2, 3}, {1, 3}, {1, 2}, {0, 3}, {0, 2}, {0, 1}};

  // Iterate through cell edges
  for (std::int32_t ei = 0; ei < 6; ++ei)
  {
    const std::int32_t v0 = edges[ei][0];
    const std::int32_t v1 = edges[ei][1];
    if (indices[ei + 4] >= 0)
    {
      // Connect edge midpoint to its end vertices

      // Only add upper-triangular connections
      conn[v1][ei + 4] = true;
      conn[v0][ei + 4] = true;

      // Each edge has two attached facets, in the original cell. The
      // numbering of the attached facets is the same as the two
      // vertices which are not in the edge

      // Opposite edge indices sum to 5. Get index of opposite edge.
      const std::int32_t e_opp = 5 - ei;

      // For each facet attached to the edge
      for (std::int32_t j = 0; j < 2; ++j)
      {
        const std::int32_t fj = edges[e_opp][j];
        const std::int32_t le_j = longest_edge[fj];
        if (le_j == ei)
        {
          const std::int32_t fk = edges[e_opp][1 - j];
          const std::int32_t le_k = longest_edge[fk];
          // This is longest edge - connect to opposite vertex

          // Only add upper-triangular connection
          conn[fk][ei + 4] = true;
          if (le_k == ei and indices[e_opp + 4] >= 0)
          {
            // Longest edge of two adjacent facets
            // Join to opposite edge (through centre of tetrahedron)
            // if marked.
            conn[ei + 4][e_opp + 4] = true;
            conn[e_opp + 4][ei + 4] = true;
          }
        }
        else
        {
          // Not longest edge, but marked, so
          // connect back to longest edge of facet
          conn[le_j + 4][ei + 4] = true;
          conn[ei + 4][le_j + 4] = true;
        }
      }
    }
    else
    {
      // No marking on this edge, just connect ends
      conn[v1][v0] = true;
      conn[v0][v1] = true;
    }
  }

  // Iterate through all possible new vertices
  std::vector<std::int32_t> facet_set, tet_set;
  for (std::int32_t i = 0; i < 10; ++i)
  {
    for (std::int32_t j = i + 1; j < 10; ++j)
    {
      if (conn[i][j])
      {
        facet_set.clear();
        for (std::int32_t k = j + 1; k < 10; ++k)
        {
          if (conn[i][k] and conn[j][k])
          {
            // Note that i < j < m < k
            for (const std::int32_t& m : facet_set)
              if (conn[m][k])
                tet_set.insert(tet_set.end(), {i, j, m, k});
            facet_set.push_back(k);
          }
        }
      }
    }
  }

  return tet_set;
}
//-----------------------------------------------------------------------------
/// Get the subdivision of an original simplex into smaller simplices,
/// for a given set of marked edges, and the longest edge of each facet
/// (cell local indexing). A flag indicates if a uniform subdivision is
/// preferable in 2D.
///
/// @param[in] indices Vector indicating which edges are to be
///   split (value >=0)
/// @param[in] longest_edge Vector indicating the longest edge for each
///   triangle. For tdim=2, one entry, for tdim=3, four entries.
/// @param[in] tdim Topological dimension (2 or 3)
/// @param[in] uniform Make a "uniform" subdivision with all triangles
///   being similar shape
/// @return
std::vector<std::int32_t>
get_simplices(const std::vector<std::int64_t>& indices,
              const std::vector<std::int32_t>& longest_edge, std::int32_t tdim,
              bool uniform)
{
  if (tdim == 2)
  {
    assert(longest_edge.size() == 1);
    return get_triangles(indices, longest_edge[0], uniform);
  }
  else if (tdim == 3)
  {
    assert(longest_edge.size() == 4);
    return get_tetrahedra(indices, longest_edge);
  }
  else
    throw std::runtime_error("Topological dimension not supported");
}

// Get the longest edge of each face (using local mesh index)
std::pair<std::vector<std::int32_t>, std::vector<std::int8_t>>
face_long_edge(const mesh::Mesh& mesh)
{
  const int tdim = mesh.topology().dim();
  // FIXME: cleanup these calls? Some of the happen internally again.
  mesh.topology_mutable().create_entities(1);
  mesh.topology_mutable().create_entities(2);
  mesh.topology_mutable().create_connectivity(2, 1);
  mesh.topology_mutable().create_connectivity(1, tdim);
  mesh.topology_mutable().create_connectivity(tdim, 2);

  std::int64_t num_faces = mesh.topology().index_map(2)->size_local()
                           + mesh.topology().index_map(2)->num_ghosts();

  // Storage for face-local index of longest edge
  std::vector<std::int32_t> long_edge(num_faces);
  std::vector<std::int8_t> edge_ratio_ok;

  // Check mesh face quality (may be used in 2D to switch to "uniform"
  // refinement)
  const double min_ratio = sqrt(2.0) / 2.0;
  if (tdim == 2)
    edge_ratio_ok.resize(num_faces);

  const graph::AdjacencyList<std::int32_t>& x_dofmap = mesh.geometry().dofmap();

  auto c_to_v = mesh.topology().connectivity(tdim, 0);
  assert(c_to_v);
  auto e_to_c = mesh.topology().connectivity(1, tdim);
  assert(e_to_c);
  auto e_to_v = mesh.topology().connectivity(1, 0);
  assert(e_to_v);

  // Store all edge lengths in Mesh to save recalculating for each Face
  auto map_e = mesh.topology().index_map(1);
  assert(map_e);
  std::vector<double> edge_length(map_e->size_local() + map_e->num_ghosts());
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

    auto x_dofs = x_dofmap.links(cells.front());
    xtl::span<const double, 3> x0(
        mesh.geometry().x().data() + 3 * x_dofs[local0], 3);
    xtl::span<const double, 3> x1(
        mesh.geometry().x().data() + 3 * x_dofs[local1], 3);

    // Compute length of edge between vertex x0 and x1
    edge_length[e] = std::sqrt(std::transform_reduce(
        x0.begin(), x0.end(), x1.begin(), 0.0, std::plus<>(),
        [](auto x0, auto x1) { return (x0 - x1) * (x0 - x1); }));
  }

  // Get longest edge of each face
  auto f_to_v = mesh.topology().connectivity(2, 0);
  assert(f_to_v);
  auto f_to_e = mesh.topology().connectivity(2, 1);
  assert(f_to_e);
  const std::vector global_indices
      = mesh.topology().index_map(0)->global_indices();
  for (int f = 0; f < f_to_v->num_nodes(); ++f)
  {
    auto face_edges = f_to_e->links(f);

    std::int32_t imax = 0;
    double max_len = 0.0;
    double min_len = std::numeric_limits<double>::max();

    for (int i = 0; i < 3; ++i)
    {
      const double e_len = edge_length[face_edges[i]];
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
//-----------------------------------------------------------------
// Computes the parent-child facet relationship
// @param simplex_set - index into indices for each child cell
// @return mapping from child to parent facets, using cell-local index
template <int tdim>
std::vector<std::int8_t>
compute_parent_facets(const std::vector<std::int32_t>& simplex_set)
{
  assert(simplex_set.size() % (tdim + 1) == 0);

  std::vector<std::int8_t> parent_facet(simplex_set.size(), -1);

  // Index lookups in 'indices' for the child vertices that occur on each parent
  // facet in 2D and 3D. In 2D each edge has 3 child vertices, and in 3D each
  // triangular facet has six child vertices.
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
          std::sort(cf.begin(), cf.end());
          auto it = std::set_intersection(facet_table_2d[fpi].begin(),
                                          facet_table_2d[fpi].end(), cf.begin(),
                                          cf.end(), set_output.begin());
          num_common_vertices = std::distance(set_output.begin(), it);
        }
        else
        {
          for (int j = 0; j < tdim; ++j)
            cf[j] = simplex_set[cc * 4 + facet_table_3d[fci][j]];
          std::sort(cf.begin(), cf.end());
          auto it = std::set_intersection(facet_table_3d[fpi].begin(),
                                          facet_table_3d[fpi].end(), cf.begin(),
                                          cf.end(), set_output.begin());
          num_common_vertices = std::distance(set_output.begin(), it);
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
//-----------------------------------------------------------------------------
// Convenient interface for both uniform and marker refinement
std::tuple<graph::AdjacencyList<std::int64_t>, xt::xtensor<double, 2>,
           std::vector<std::int32_t>, std::vector<std::int8_t>>
compute_refinement(MPI_Comm neighbor_comm,
                   const std::vector<std::int8_t>& marked_edges,
                   const graph::AdjacencyList<int>& shared_edges,
                   const mesh::Mesh& mesh,
                   const std::vector<std::int32_t>& long_edge,
                   const std::vector<std::int8_t>& edge_ratio_ok,
                   plaza::RefinementOptions options)
{
  const std::int32_t tdim = mesh.topology().dim();
  const std::int32_t num_cell_edges = tdim * 3 - 3;
  const std::int32_t num_cell_vertices = tdim + 1;

  bool compute_facets
      = (options == plaza::RefinementOptions::parent_facet
         or options == plaza::RefinementOptions::parent_cell_and_facet);

  bool compute_parent_cell
      = (options == plaza::RefinementOptions::parent_cell
         or options == plaza::RefinementOptions::parent_cell_and_facet);

  // Make new vertices in parallel
  const auto [new_vertex_map, new_vertex_coordinates]
      = create_new_vertices(neighbor_comm, shared_edges, mesh, marked_edges);

  std::vector<std::int32_t> parent_cell;
  std::vector<std::int8_t> parent_facet;
  std::vector<std::int64_t> indices(num_cell_vertices + num_cell_edges);
  std::vector<std::int32_t> simplex_set;

  auto map_c = mesh.topology().index_map(tdim);
  assert(map_c);

  auto c_to_v = mesh.topology().connectivity(tdim, 0);
  assert(c_to_v);
  auto c_to_e = mesh.topology().connectivity(tdim, 1);
  assert(c_to_e);
  auto c_to_f = mesh.topology().connectivity(tdim, 2);
  assert(c_to_f);

  std::int32_t num_new_vertices_local = std::count(
      marked_edges.begin(),
      marked_edges.begin() + mesh.topology().index_map(1)->size_local(), true);

  std::vector<std::int64_t> global_indices
      = adjust_indices(*mesh.topology().index_map(0), num_new_vertices_local);

  const int num_cells = map_c->size_local();

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

      // FIXME: this has an expensive dynamic memory allocation
      simplex_set = get_simplices(indices, longest_edge, tdim, uniform);

      // Save parent index
      const std::int32_t ncells = simplex_set.size() / num_cell_vertices;

      if (compute_parent_cell)
      {
        for (std::int32_t i = 0; i < ncells; ++i)
          parent_cell.push_back(c);
      }
      if (compute_facets)
      {
        std::vector<std::int8_t> npf;
        if (tdim == 3)
          npf = compute_parent_facets<3>(simplex_set);
        else
          npf = compute_parent_facets<2>(simplex_set);
        parent_facet.insert(parent_facet.end(), npf.begin(), npf.end());
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
  graph::AdjacencyList<std::int64_t> cell_adj(std::move(cell_topology),
                                              std::move(offsets));

  return {std::move(cell_adj), std::move(new_vertex_coordinates),
          std::move(parent_cell), std::move(parent_facet)};
}
//-----------------------------------------------------------------------------
} // namespace

//-----------------------------------------------------------------------------
std::tuple<mesh::Mesh, std::vector<std::int32_t>, std::vector<std::int8_t>>
plaza::refine(const mesh::Mesh& mesh, bool redistribute,
              RefinementOptions options)
{

  auto [cell_adj, new_vertex_coordinates, parent_cell, parent_facet]
      = plaza::compute_refinement_data(mesh, options);

  if (dolfinx::MPI::size(mesh.comm()) == 1)
  {
    return {mesh::create_mesh(mesh.comm(), cell_adj, mesh.geometry().cmap(),
                              new_vertex_coordinates, mesh::GhostMode::none),
            std::move(parent_cell), std::move(parent_facet)};
  }

  const std::shared_ptr<const common::IndexMap> map_c
      = mesh.topology().index_map(mesh.topology().dim());
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

  return {partition(mesh, cell_adj, new_vertex_coordinates, redistribute,
                    ghost_mode),
          std::move(parent_cell), std::move(parent_facet)};
}
//-----------------------------------------------------------------------------
std::tuple<mesh::Mesh, std::vector<std::int32_t>, std::vector<std::int8_t>>
plaza::refine(const mesh::Mesh& mesh,
              const xtl::span<const std::int32_t>& edges, bool redistribute,
              RefinementOptions options)
{

  auto [cell_adj, new_vertex_coordinates, parent_cell, parent_facet]
      = plaza::compute_refinement_data(mesh, edges, options);

  if (dolfinx::MPI::size(mesh.comm()) == 1)
  {
    return {mesh::create_mesh(mesh.comm(), cell_adj, mesh.geometry().cmap(),
                              new_vertex_coordinates, mesh::GhostMode::none),
            std::move(parent_cell), std::move(parent_facet)};
  }

  const std::shared_ptr<const common::IndexMap> map_c
      = mesh.topology().index_map(mesh.topology().dim());
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

  return {partition(mesh, cell_adj, new_vertex_coordinates, redistribute,
                    ghost_mode),
          std::move(parent_cell), std::move(parent_facet)};
}
//------------------------------------------------------------------------------
std::tuple<graph::AdjacencyList<std::int64_t>, xt::xtensor<double, 2>,
           std::vector<std::int32_t>, std::vector<std::int8_t>>
plaza::compute_refinement_data(const mesh::Mesh& mesh,
                               RefinementOptions options)
{

  if (mesh.topology().cell_type() != mesh::CellType::triangle
      and mesh.topology().cell_type() != mesh::CellType::tetrahedron)
  {
    throw std::runtime_error("Cell type not supported");
  }

  common::Timer t0("PLAZA: refine");

  auto map_e = mesh.topology().index_map(1);
  if (!map_e)
    throw std::runtime_error("Edges must be initialised");

  // Get sharing ranks for each edge
  graph::AdjacencyList<int> edge_ranks = map_e->index_to_dest_ranks();

  // Create unique list of ranks that share edges (owners of ghosts
  // plus ranks that ghost owned indices)
  std::vector<int> ranks(edge_ranks.array().begin(), edge_ranks.array().end());
  std::sort(ranks.begin(), ranks.end());
  ranks.erase(std::unique(ranks.begin(), ranks.end()), ranks.end());

  // Convert edge_ranks from global rank to to neighbourhood ranks
  std::transform(edge_ranks.array().begin(), edge_ranks.array().end(),
                 edge_ranks.array().begin(),
                 [&ranks](auto r)
                 {
                   auto it = std::lower_bound(ranks.begin(), ranks.end(), r);
                   assert(it != ranks.end() and *it == r);
                   return std::distance(ranks.begin(), it);
                 });

  MPI_Comm comm;
  MPI_Dist_graph_create_adjacent(mesh.comm(), ranks.size(), ranks.data(),
                                 MPI_UNWEIGHTED, ranks.size(), ranks.data(),
                                 MPI_UNWEIGHTED, MPI_INFO_NULL, false, &comm);

  const auto [long_edge, edge_ratio_ok] = face_long_edge(mesh);
  auto [cell_adj, new_vertex_coordinates, parent_cell, parent_facet]
      = compute_refinement(comm,
                           std::vector<std::int8_t>(
                               map_e->size_local() + map_e->num_ghosts(), true),
                           edge_ranks, mesh, long_edge, edge_ratio_ok, options);
  MPI_Comm_free(&comm);

  return {std::move(cell_adj), std::move(new_vertex_coordinates),
          std::move(parent_cell), std::move(parent_facet)};
}
//------------------------------------------------------------------------------
std::tuple<graph::AdjacencyList<std::int64_t>, xt::xtensor<double, 2>,
           std::vector<std::int32_t>, std::vector<std::int8_t>>
plaza::compute_refinement_data(const mesh::Mesh& mesh,
                               const xtl::span<const std::int32_t>& edges,
                               RefinementOptions options)
{
  if (mesh.topology().cell_type() != mesh::CellType::triangle
      and mesh.topology().cell_type() != mesh::CellType::tetrahedron)
  {
    throw std::runtime_error("Cell type not supported");
  }

  common::Timer t0("PLAZA: refine");

  auto map_e = mesh.topology().index_map(1);
  if (!map_e)
    throw std::runtime_error("Edges must be initialised");

  // Get sharing ranks for each edge
  graph::AdjacencyList<int> edge_ranks = map_e->index_to_dest_ranks();

  // Create unique list of ranks that share edges (owners of ghosts plus
  // ranks that ghost owned indices)
  std::vector<int> ranks(edge_ranks.array().begin(), edge_ranks.array().end());
  std::sort(ranks.begin(), ranks.end());
  ranks.erase(std::unique(ranks.begin(), ranks.end()), ranks.end());

  // Convert edge_ranks from global rank to to neighbourhood ranks
  std::transform(edge_ranks.array().begin(), edge_ranks.array().end(),
                 edge_ranks.array().begin(),
                 [&ranks](auto r)
                 {
                   auto it = std::lower_bound(ranks.begin(), ranks.end(), r);
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
  const auto [long_edge, edge_ratio_ok] = face_long_edge(mesh);
  enforce_rules(comm, edge_ranks, marked_edges, mesh, long_edge);

  auto [cell_adj, new_vertex_coordinates, parent_cell, parent_facet]
      = compute_refinement(comm, marked_edges, edge_ranks, mesh, long_edge,
                           edge_ratio_ok, options);
  MPI_Comm_free(&comm);

  return {std::move(cell_adj), std::move(new_vertex_coordinates),
          std::move(parent_cell), std::move(parent_facet)};
}
//-----------------------------------------------------------------------------
