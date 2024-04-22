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
#include <dolfinx/mesh/utils.h>
#include <limits>
#include <numeric>
#include <vector>

using namespace dolfinx;
using namespace dolfinx::refinement;

namespace
{
//-----------------------------------------------------------------------------
/// 2D version of subdivision allowing for uniform subdivision (flag)
/// @param[in] indices Vector containing the
/// global indices for the original vertices and potential new vertices at each
/// edge. If an edge is not refined its corresponding entry is -1. Size
/// `num_vertices + num_edges`
/// @param[in] longest_edge Local index of the longest edge in the triangle.
/// @param[in] uniform If true, the triangle is subdivided into four similar
/// sub-triangles.
/// @returns Local indices for each sub-divided triangle
std::pair<std::array<std::int32_t, 12>, std::size_t>
get_triangles(std::span<const std::int64_t> indices,
              const std::int32_t longest_edge, bool uniform)
{
  // NOTE: The assumption below is based on the UFC ordering of a triangle, i.e.
  // that the N-th edge of a triangle is the edge where the N-th vertex is not
  // part of the set. v0 and v1 are at ends of longest_edge (e2) opposite vertex
  // has same index as longest_edge
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
  {
    return {{e0, e1, v2, e1, e2, v0, e2, e0, v1, e2, e1, e0}, 12};
  }
  else
  {
    // Break each half of triangle into one or two sub-triangles
    std::array<std::int32_t, 12> tri_set;
    tri_set.fill(-1);
    std::size_t tri_set_size = 0;
    if (indices[e0] >= 0)
    {
      std::array<std::int32_t, 6> tri_set0 = {e2, v2, e0, e2, e0, v1};
      std::copy(tri_set0.begin(), tri_set0.end(),
                std::next(tri_set.begin(), tri_set_size));
      tri_set_size += tri_set0.size();
    }
    else
    {
      std::array<std::int32_t, 3> tri_set0 = {e2, v2, v1};
      std::copy(tri_set0.begin(), tri_set0.end(),
                std::next(tri_set.begin(), tri_set_size));
      tri_set_size += tri_set0.size();
    }

    if (indices[e1] >= 0)
    {
      std::array<std::int32_t, 3> tri_set0 = {e2, v2, e1};
      std::copy(tri_set0.begin(), tri_set0.end(),
                std::next(tri_set.begin(), tri_set_size));
      tri_set_size += tri_set0.size();

      std::array<std::int32_t, 3> tri_set1 = {e2, e1, v0};
      std::copy(tri_set1.begin(), tri_set1.end(),
                std::next(tri_set.begin(), tri_set_size));
      tri_set_size += tri_set1.size();
    }
    else
    {
      std::array<std::int32_t, 3> tri_set0 = {e2, v2, v0};
      std::copy(tri_set0.begin(), tri_set0.end(),
                std::next(tri_set.begin(), tri_set_size));
      tri_set_size += tri_set0.size();
    }

    return {tri_set, tri_set_size};
  }
}
//-----------------------------------------------------------------------------
// 3D version of subdivision
std::pair<std::array<std::int32_t, 32>, std::size_t>
get_tetrahedra(std::span<const std::int64_t> indices,
               std::span<const std::int32_t> longest_edge)
{
  // Connectivity matrix for ten possible points (4 vertices + 6 edge
  // midpoints) ordered {v0, v1, v2, v3, e0, e1, e2, e3, e4, e5} Only
  // need upper triangle, but sometimes it is easier just to insert both
  // entries (j,i) and (i,j).
  bool conn[10][10] = {};

  // Edge connectivity to vertices (and by extension facets)
  constexpr std::int32_t edges[6][2]
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
  std::array<std::int32_t, 32> tet_set;
  tet_set.fill(-1);
  std::size_t tet_set_size = 0;
  for (std::int32_t i = 0; i < 10; ++i)
  {
    for (std::int32_t j = i + 1; j < 10; ++j)
    {
      if (conn[i][j])
      {
        std::array<std::int32_t, 10> facet_set_b;
        std::span<std::int32_t> facet_set(facet_set_b.data(), 0);
        for (std::int32_t k = j + 1; k < 10; ++k)
        {
          if (conn[i][k] and conn[j][k])
          {
            // Note that i < j < m < k
            for (std::int32_t m : facet_set)
            {
              if (conn[m][k])
              {
                assert(tet_set_size + 4 < tet_set.size());
                tet_set[tet_set_size++] = i;
                tet_set[tet_set_size++] = j;
                tet_set[tet_set_size++] = m;
                tet_set[tet_set_size++] = k;
              }
            }

            facet_set = std::span(facet_set.data(), facet_set.size() + 1);
            facet_set.back() = k;
          }
        }
      }
    }
  }

  return {std::move(tet_set), tet_set_size};
}
//-----------------------------------------------------------------------------
} // namespace

//-----------------------------------------------------------------------------
void plaza::impl::enforce_rules(MPI_Comm comm,
                                const graph::AdjacencyList<int>& shared_edges,
                                std::span<std::int8_t> marked_edges,
                                const mesh::Topology& topology,
                                std::span<const std::int32_t> long_edge)
{
  common::Timer t0("PLAZA: Enforce rules");

  // Enforce rule, that if any edge of a face is marked, longest edge
  // must also be marked

  auto map_e = topology.index_map(1);
  assert(map_e);
  auto map_f = topology.index_map(2);
  assert(map_f);
  const std::int32_t num_faces = map_f->size_local() + map_f->num_ghosts();

  auto f_to_e = topology.connectivity(2, 1);
  assert(f_to_e);

  // Get number of neighbors
  int indegree(-1), outdegree(-2), weighted(-1);
  MPI_Dist_graph_neighbors_count(comm, &indegree, &outdegree, &weighted);
  assert(indegree == outdegree);
  const int num_neighbors = indegree;
  std::vector<std::vector<std::int32_t>> marked_for_update(num_neighbors);

  std::int32_t update_count = 1;
  while (update_count > 0)
  {
    update_count = 0;
    update_logical_edgefunction(comm, marked_for_update, marked_edges, *map_e);
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
                  comm);
  }
}
//-----------------------------------------------------------------------------
std::pair<std::array<std::int32_t, 32>, std::size_t>
plaza::impl::get_simplices(std::span<const std::int64_t> indices,
                           std::span<const std::int32_t> longest_edge, int tdim,
                           bool uniform)
{
  if (tdim == 2)
  {
    assert(longest_edge.size() == 1);
    auto [_d, size] = get_triangles(indices, longest_edge[0], uniform);
    std::array<std::int32_t, 32> d;
    std::copy_n(_d.begin(), size, d.begin());
    return {d, size};
  }
  else if (tdim == 3)
  {
    assert(longest_edge.size() == 4);
    return get_tetrahedra(indices, longest_edge);
  }
  else
    throw std::runtime_error("Topological dimension not supported");
}
//-----------------------------------------------------------------------------
