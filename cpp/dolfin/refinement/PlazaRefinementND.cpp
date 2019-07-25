// Copyright (C) 2014-2018 Chris Richardson
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "PlazaRefinementND.h"
#include "ParallelRefinement.h"
#include <dolfin/common/Timer.h>
#include <dolfin/mesh/Geometry.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshEntity.h>
#include <dolfin/mesh/MeshIterator.h>
#include <limits>
#include <map>
#include <vector>

using namespace dolfin;
using namespace dolfin::refinement;

namespace
{
// Propagate edge markers according to rules (longest edge of each
// face must be marked, if any edge of face is marked)
void enforce_rules(ParallelRefinement& p_ref, const mesh::Mesh& mesh,
                   const std::vector<std::int32_t>& long_edge)
{
  common::Timer t0("PLAZA: Enforce rules");

  // Enforce rule, that if any edge of a face is marked, longest edge
  // must also be marked

  std::int32_t update_count = 1;
  while (update_count != 0)
  {
    update_count = 0;
    p_ref.update_logical_edgefunction();

    for (const auto& f : mesh::MeshRange<mesh::MeshEntity>(mesh, 2))
    {
      const std::int32_t long_e = long_edge[f.index()];
      if (p_ref.is_marked(long_e))
        continue;
      bool any_marked = false;
      for (const auto& e : mesh::EntityRange<mesh::MeshEntity>(f, 1))
        any_marked |= p_ref.is_marked(e.index());
      if (any_marked)
      {
        p_ref.mark(long_e);
        ++update_count;
      }
    }
    update_count = dolfin::MPI::sum(mesh.mpi_comm(), update_count);
  }
}
//-----------------------------------------------------------------------------
// Convenient interface for both uniform and marker refinement
mesh::Mesh compute_refinement(const mesh::Mesh& mesh, ParallelRefinement& p_ref,
                              const std::vector<std::int32_t>& long_edge,
                              const std::vector<bool>& edge_ratio_ok,
                              bool redistribute)
{
  const std::int32_t tdim = mesh.topology().dim();
  const std::int32_t num_cell_edges = tdim * 3 - 3;
  const std::int32_t num_cell_vertices = tdim + 1;

  // Make new vertices in parallel
  p_ref.create_new_vertices();
  const std::map<std::size_t, std::size_t>& new_vertex_map
      = p_ref.edge_to_new_vertex();

  std::vector<std::size_t> parent_cell;
  std::vector<std::int64_t> indices(num_cell_vertices + num_cell_edges);
  std::vector<std::size_t> marked_edge_list;
  std::vector<std::int32_t> simplex_set;

  const std::vector<std::int64_t>& global_indices
      = mesh.topology().global_indices(0);
  for (const auto& cell : mesh::MeshRange<mesh::MeshEntity>(mesh, tdim))
  {
    // Create vector of indices in the order [vertices][edges], 3+3 in
    // 2D, 4+6 in 3D
    std::int32_t j = 0;
    for (const auto& v : mesh::EntityRange<mesh::MeshEntity>(cell, 0))
      indices[j++] = global_indices[v.index()];

    marked_edge_list = p_ref.marked_edge_list(cell);
    if (marked_edge_list.size() == 0)
    {
      // Copy over existing Cell to new topology
      std::vector<std::int64_t> cell_topology;
      for (const auto& v : mesh::EntityRange<mesh::MeshEntity>(cell, 0))
        cell_topology.push_back(global_indices[v.index()]);
      p_ref.new_cells(cell_topology);
      parent_cell.push_back(cell.index());
    }
    else
    {
      // Get the marked edge indices for new vertices and make bool
      // vector of marked edges
      std::vector<bool> markers(num_cell_edges, false);
      for (auto& p : marked_edge_list)
      {
        markers[p] = true;
        const std::int32_t edge_index = cell.entities(1)[p];

        auto it = new_vertex_map.find(edge_index);
        assert(it != new_vertex_map.end());
        indices[num_cell_vertices + p] = it->second;
      }

      // Need longest edges of each facet in cell local indexing
      std::vector<std::int32_t> longest_edge;
      for (const auto& f : mesh::EntityRange<mesh::MeshEntity>(cell, 2))
        longest_edge.push_back(long_edge[f.index()]);

      // Convert to cell local index
      for (auto& p : longest_edge)
      {
        int i = 0;
        for (const auto& ej : mesh::EntityRange<mesh::MeshEntity>(cell, 1))
        {
          if (p == ej.index())
          {
            p = i;
            break;
          }
          ++i;
        }
      }

      const bool uniform = (tdim == 2) ? edge_ratio_ok[cell.index()] : false;

      // FIXME: this has an expensive dynamic memory allocation
      simplex_set = PlazaRefinementND::get_simplices(markers, longest_edge,
                                                     tdim, uniform);

      // Save parent index
      const std::int32_t ncells = simplex_set.size() / num_cell_vertices;
      for (std::int32_t i = 0; i != ncells; ++i)
        parent_cell.push_back(cell.index());

      // Convert from cell local index to mesh index and add to cells
      std::vector<std::int64_t> simplex_set_global(simplex_set.size());
      for (std::size_t i = 0; i < simplex_set_global.size(); ++i)
        simplex_set_global[i] = indices[simplex_set[i]];
      p_ref.new_cells(simplex_set_global);
    }
  }

  const bool serial = (dolfin::MPI::size(mesh.mpi_comm()) == 1);
  if (serial)
    return p_ref.build_local();
  else
    return p_ref.partition(redistribute);
}
//-----------------------------------------------------------------------------
// 2D version of subdivision allowing for uniform subdivision (flag)
std::vector<std::int32_t> get_triangles(const std::vector<bool>& marked_edges,
                                        const std::int32_t longest_edge,
                                        bool uniform)
{
  // Longest edge must be marked
  assert(marked_edges[longest_edge]);

  // v0 and v1 are at ends of longest_edge (e2) opposite vertex has same
  // index as longest_edge
  const std::int32_t v0 = (longest_edge + 1) % 3;
  const std::int32_t v1 = (longest_edge + 2) % 3;
  const std::int32_t v2 = longest_edge;
  const std::int32_t e0 = v0 + 3;
  const std::int32_t e1 = v1 + 3;
  const std::int32_t e2 = v2 + 3;

  // If all edges marked, consider uniform refinement
  if (uniform and marked_edges[v0] and marked_edges[v1])
    return {e0, e1, v2, e1, e2, v0, e2, e0, v1, e2, e1, e0};

  // Break each half of triangle into one or two sub-triangles
  std::vector<std::int32_t> tri_set;
  if (marked_edges[v0])
    tri_set = {e2, v2, e0, e2, e0, v1};
  else
    tri_set = {e2, v2, v1};

  if (marked_edges[v1])
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
get_tetrahedra(const std::vector<bool>& marked_edges,
               const std::vector<std::int32_t>& longest_edge)
{
  // Connectivity matrix for ten possible points (4 vertices + 6 edge midpoints)
  // ordered {v0, v1, v2, v3, e0, e1, e2, e3, e4, e5}
  // Only need upper triangle, but sometimes it is easier just to insert
  // both entries (j,i) and (i,j).
  bool conn[10][10] = {};

  // Edge connectivity to vertices (and by extension facets)
  static const std::int32_t edges[6][2]
      = {{2, 3}, {1, 3}, {1, 2}, {0, 3}, {0, 2}, {0, 1}};

  // Iterate through cell edges
  for (std::int32_t ei = 0; ei < 6; ++ei)
  {
    const std::int32_t v0 = edges[ei][0];
    const std::int32_t v1 = edges[ei][1];
    if (marked_edges[ei])
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
          if (le_k == ei and marked_edges[e_opp])
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
          if (conn[i][k] && conn[j][k])
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
// Get the longest edge of each face (using local mesh index)
std::pair<std::vector<std::int32_t>, std::vector<bool>>
face_long_edge(const mesh::Mesh& mesh)
{
  const std::int32_t tdim = mesh.topology().dim();
  mesh.create_entities(1);
  mesh.create_entities(2);
  mesh.create_connectivity(2, 1);

  // Storage for face-local index of longest edge
  std::vector<std::int32_t> long_edge(mesh.num_entities(2));
  std::vector<bool> edge_ratio_ok;

  // Check mesh face quality (may be used in 2D to switch to "uniform"
  // refinement)
  const double min_ratio = sqrt(2.0) / 2.0;
  if (tdim == 2)
    edge_ratio_ok.resize(mesh.num_entities(2));

  // Store all edge lengths in Mesh to save recalculating for each Face
  const mesh::Geometry& geometry = mesh.geometry();
  std::vector<double> edge_length(mesh.num_entities(1));
  for (const auto& e : mesh::MeshRange<mesh::MeshEntity>(mesh, 1))
  {
    const std::int32_t* v = e.entities(0);
    edge_length[e.index()] = (geometry.x(v[0]) - geometry.x(v[1])).norm();
  }

  // Get longest edge of each face
  const std::vector<std::int64_t>& global_indices
      = mesh.topology().global_indices(0);
  for (const auto& f : mesh::MeshRange<mesh::MeshEntity>(mesh, 2))
  {
    const std::int32_t* face_edges = f.entities(1);

    std::int32_t imax = 0;
    double max_len = 0.0;
    double min_len = std::numeric_limits<double>::max();

    for (unsigned int i = 0; i < 3; ++i)
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
        const mesh::MeshEntity vmax(mesh, 0, f.entities(0)[imax]);
        const mesh::MeshEntity vi(mesh, 0, f.entities(0)[i]);
        if (global_indices[vi.index()] > global_indices[vmax.index()])
          imax = i;
      }
    }

    // Only save edge ratio in 2D
    if (tdim == 2)
      edge_ratio_ok[f.index()] = (min_len / max_len >= min_ratio);

    long_edge[f.index()] = face_edges[imax];
  }

  return std::make_pair(std::move(long_edge), std::move(edge_ratio_ok));
}
//-----------------------------------------------------------------------------

} // namespace
//-----------------------------------------------------------------------------
mesh::Mesh PlazaRefinementND::refine(const mesh::Mesh& mesh, bool redistribute)
{
  if (mesh.cell_type != mesh::CellType::triangle
      and mesh.cell_type != mesh::CellType::tetrahedron)
  {
    throw std::runtime_error("Cell type not supported");
  }

  common::Timer t0("PLAZA: refine");
  std::vector<std::int32_t> long_edge;
  std::vector<bool> edge_ratio_ok;
  std::tie(long_edge, edge_ratio_ok) = face_long_edge(mesh);

  ParallelRefinement p_ref(mesh);
  p_ref.mark_all();

  return compute_refinement(mesh, p_ref, long_edge, edge_ratio_ok,
                            redistribute);
}
//-----------------------------------------------------------------------------
mesh::Mesh
PlazaRefinementND::refine(const mesh::Mesh& mesh,
                          const mesh::MeshFunction<int>& refinement_marker,
                          bool redistribute)
{
  if (mesh.cell_type != mesh::CellType::triangle
      and mesh.cell_type != mesh::CellType::tetrahedron)
  {
    throw std::runtime_error("Cell type not supported");
  }

  common::Timer t0("PLAZA: refine");
  std::vector<std::int32_t> long_edge;
  std::vector<bool> edge_ratio_ok;
  std::tie(long_edge, edge_ratio_ok) = face_long_edge(mesh);

  ParallelRefinement p_ref(mesh);
  p_ref.mark(refinement_marker);

  enforce_rules(p_ref, mesh, long_edge);

  return compute_refinement(mesh, p_ref, long_edge, edge_ratio_ok,
                            redistribute);
}
//-----------------------------------------------------------------------------
std::vector<std::int32_t>
PlazaRefinementND::get_simplices(const std::vector<bool>& marked_edges,
                                 const std::vector<std::int32_t>& longest_edge,
                                 std::int32_t tdim, bool uniform)
{
  if (tdim == 2)
  {
    assert(longest_edge.size() == 1);
    return get_triangles(marked_edges, longest_edge[0], uniform);
  }
  else if (tdim == 3)
  {
    assert(longest_edge.size() == 4);
    return get_tetrahedra(marked_edges, longest_edge);
  }
  else
  {
    throw std::runtime_error("Topological dimension not supported");
    return std::vector<std::int32_t>();
  }
}
//-----------------------------------------------------------------------------
