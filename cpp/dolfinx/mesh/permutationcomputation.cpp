// Copyright (C) 2020 Matthew Scroggs
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "permutationcomputation.h"
#include "Topology.h"
#include "cell_types.h"
#include <algorithm>
#include <bitset>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/Timer.h>
#include <dolfinx/common/log.h>
#include <dolfinx/graph/AdjacencyList.h>

namespace
{
constexpr int _BITSETSIZE = 32;
} // namespace

using namespace dolfinx;

namespace
{
std::pair<std::int8_t, std::int8_t>
compute_triangle_rot_reflect(const std::vector<std::int32_t>& e_vertices,
                             const std::vector<std::int64_t>& vertices)
{

  // Number of rotations
  std::uint8_t min_v
      = std::distance(e_vertices.begin(), std::ranges::min_element(e_vertices));

  // pre is the (local) number of the next vertex clockwise from the lowest
  // numbered vertex
  const int pre = e_vertices[(min_v + 2) % 3];

  // post is the (local) number of the next vertex anticlockwise from the
  // lowest numbered vertex
  const int post = e_vertices[(min_v + 1) % 3];

  std::uint8_t g_min_v
      = std::distance(vertices.begin(), std::ranges::min_element(vertices));

  // g_pre is the (global) number of the next vertex clockwise from the lowest
  // numbered vertex
  const int g_pre = vertices[(g_min_v + 2) % 3];

  // g_post is the (global) number of the next vertex anticlockwise from the
  // lowest numbered vertex
  const int g_post = vertices[(g_min_v + 1) % 3];

  std::uint8_t rots = 0;
  if (g_post > g_pre)
    rots = (g_min_v + 3 - min_v) % 3;
  else
    rots = (min_v + 3 - g_min_v) % 3;

  return {(post > pre) == (g_post < g_pre), rots};
}
//-----------------------------------------------------------------------------
std::pair<std::int8_t, std::int8_t>
compute_quad_rot_reflect(const std::vector<std::int32_t>& e_vertices,
                         const std::vector<std::int64_t>& vertices)
{
  // Find minimum local cell vertex on facet
  std::uint8_t min_v
      = std::distance(e_vertices.begin(), std::ranges::min_element(e_vertices));

  // Table of next and previous vertices
  // 0 - 2
  // |   |
  // 1 - 3
  const std::array<std::int8_t, 4> prev = {2, 0, 3, 1};

  // pre is the (local) number of the next vertex clockwise from the
  // lowest numbered vertex
  std::int32_t pre = e_vertices[prev[min_v]];

  // post is the (local) number of the next vertex anticlockwise
  // from the lowest numbered vertex
  std::int32_t post = e_vertices[prev[3 - min_v]];

  // If min_v is 2 or 3, swap:
  // 0 - 2       0 - 3
  // |   |       |   |
  // 1 - 3       1 - 2
  // Because of the dolfinx ordering (left), in order to compute the number of
  // anti-clockwise rotations required correctly, min_v is altered to give the
  // ordering on the right.
  if (min_v == 2 or min_v == 3)
    min_v = 5 - min_v;

  // Find minimum global vertex in facet
  std::uint8_t g_min_v
      = std::distance(vertices.begin(), std::ranges::min_element(vertices));

  // rots is the number of rotations to get the lowest numbered
  // vertex to the origin

  // g_pre is the (global) number of the next vertex clockwise from the
  // lowest numbered vertex
  std::int64_t g_pre = vertices[prev[g_min_v]];

  // g_post is the (global) number of the next vertex anticlockwise
  // from the lowest numbered vertex
  std::int64_t g_post = vertices[prev[3 - g_min_v]];

  if (g_min_v == 2 or g_min_v == 3)
    g_min_v = 5 - g_min_v;

  std::uint8_t rots = 0;
  if (g_post > g_pre)
    rots = (g_min_v - min_v + 4) % 4;
  else
    rots = (min_v - g_min_v + 4) % 4;
  return {(post > pre) == (g_post < g_pre), rots};
}
//-----------------------------------------------------------------------------
template <int BITSETSIZE>
std::vector<std::bitset<BITSETSIZE>>
compute_triangle_quad_face_permutations(const mesh::Topology& topology,
                                        int cell_index)
{
  std::vector<mesh::CellType> cell_types = topology.entity_types(3);
  mesh::CellType cell_type = cell_types.at(cell_index);

  // Get face types of the cell and mesh
  std::vector<mesh::CellType> mesh_face_types = topology.entity_types(2);
  std::vector<mesh::CellType> cell_face_types(
      mesh::cell_num_entities(cell_type, 2));
  for (std::size_t i = 0; i < cell_face_types.size(); ++i)
    cell_face_types[i] = mesh::cell_facet_type(cell_type, i);

  // Connectivity for each face type
  std::vector<std::shared_ptr<const graph::AdjacencyList<std::int32_t>>> c_to_f;
  std::vector<std::shared_ptr<const graph::AdjacencyList<std::int32_t>>> f_to_v;

  // Create mapping for each face type to cell-local face index
  int tdim = topology.dim();
  std::vector<std::vector<int>> face_type_indices(mesh_face_types.size());
  for (std::size_t i = 0; i < mesh_face_types.size(); ++i)
  {
    for (std::size_t j = 0; j < cell_face_types.size(); ++j)
    {
      if (mesh_face_types[i] == cell_face_types[j])
        face_type_indices[i].push_back(j);
    }
    c_to_f.push_back(topology.connectivity({tdim, cell_index}, {2, i}));
    f_to_v.push_back(topology.connectivity({2, i}, {0, 0}));
  }

  auto c_to_v = topology.connectivity({tdim, cell_index}, {0, 0});
  assert(c_to_v);

  const std::int32_t num_cells = c_to_v->num_nodes();
  std::vector<std::bitset<BITSETSIZE>> face_perm(num_cells, 0);
  std::vector<std::int64_t> cell_vertices, vertices;
  std::vector<std::int32_t> e_vertices;
  auto im = topology.index_map(0);

  for (std::size_t t = 0; t < face_type_indices.size(); ++t)
  {
    spdlog::info("Computing permutations for face type {}", t);
    if (!face_type_indices[t].empty())
    {
      auto compute_refl_rots = (mesh_face_types[t] == mesh::CellType::triangle)
                                   ? compute_triangle_rot_reflect
                                   : compute_quad_rot_reflect;
      for (int c = 0; c < num_cells; ++c)
      {
        cell_vertices.resize(c_to_v->links(c).size());
        im->local_to_global(c_to_v->links(c), cell_vertices);

        auto cell_faces = c_to_f[t]->links(c);
        for (std::size_t i = 0; i < cell_faces.size(); ++i)
        {
          // Get the face
          const int face = cell_faces[i];
          e_vertices.resize(f_to_v[t]->num_links(face));
          vertices.resize(f_to_v[t]->num_links(face));
          im->local_to_global(f_to_v[t]->links(face), vertices);

          // Orient that triangle or quadrilateral so the lowest numbered
          // vertex is the origin, and the next vertex anticlockwise from
          // the lowest has a lower number than the next vertex clockwise.
          // Find the index of the lowest numbered vertex.

          // Find iterators pointing to cell vertex given a vertex on facet
          for (std::size_t j = 0; j < vertices.size(); ++j)
          {
            auto it = std::find(cell_vertices.begin(), cell_vertices.end(),
                                vertices[j]);
            // Get the actual local vertex indices
            e_vertices[j] = std::distance(cell_vertices.begin(), it);
          }

          // Compute reflections and rotations for this face type
          auto [refl, rots] = compute_refl_rots(e_vertices, vertices);

          // Store bits for this face
          int fi = face_type_indices[t][i];
          face_perm[c][3 * fi] = refl;
          face_perm[c][3 * fi + 1] = rots % 2;
          face_perm[c][3 * fi + 2] = rots / 2;
        }
      }
    }
  }

  return face_perm;
}
//-----------------------------------------------------------------------------
template <int BITSETSIZE>
std::vector<std::bitset<BITSETSIZE>>
compute_edge_reflections(const mesh::Topology& topology)
{
  mesh::CellType cell_type = topology.cell_type();
  const int tdim = topology.dim();
  const int edges_per_cell = cell_num_entities(cell_type, 1);

  const std::int32_t num_cells = topology.connectivity(tdim, 0)->num_nodes();

  auto c_to_v = topology.connectivity(tdim, 0);
  assert(c_to_v);
  auto c_to_e = topology.connectivity(tdim, 1);
  assert(c_to_e);
  auto e_to_v = topology.connectivity(1, 0);
  assert(e_to_v);

  auto im = topology.index_map(0);
  assert(im);

  std::vector<std::bitset<BITSETSIZE>> edge_perm(num_cells, 0);
  std::vector<std::int64_t> cell_vertices, vertices;
  for (int c = 0; c < c_to_v->num_nodes(); ++c)
  {
    cell_vertices.resize(c_to_v->num_links(c));
    im->local_to_global(c_to_v->links(c), cell_vertices);
    auto cell_edges = c_to_e->links(c);
    for (int i = 0; i < edges_per_cell; ++i)
    {
      vertices.resize(e_to_v->links(cell_edges[i]).size());
      im->local_to_global(e_to_v->links(cell_edges[i]), vertices);

      // If the entity is an interval, it should be oriented pointing
      // from the lowest numbered vertex to the highest numbered vertex.

      // Find iterators pointing to cell vertex given a vertex on facet
      auto it0
          = std::find(cell_vertices.begin(), cell_vertices.end(), vertices[0]);
      auto it1
          = std::find(cell_vertices.begin(), cell_vertices.end(), vertices[1]);

      // The number of reflections. Comparing iterators directly instead
      // of values they point to is sufficient here.
      edge_perm[c][i] = (it1 < it0) == (vertices[1] > vertices[0]);
    }
  }

  return edge_perm;
}
//-----------------------------------------------------------------------------
template <int BITSETSIZE>
std::vector<std::bitset<BITSETSIZE>>
compute_face_permutations(const mesh::Topology& topology)
{
  if (topology.entity_types(3).size() > 1)
  {
    throw std::runtime_error(
        "Cannot compute permutations for mixed topology mesh.");
  }

  const int tdim = topology.dim();
  assert(tdim > 2);
  if (!topology.index_map(2))
    throw std::runtime_error("Faces have not been computed.");

  // Compute face permutations for first cell type in the topology
  return compute_triangle_quad_face_permutations<BITSETSIZE>(topology, 0);
}
//-----------------------------------------------------------------------------
} // namespace

//-----------------------------------------------------------------------------
std::pair<std::vector<std::uint8_t>, std::vector<std::uint32_t>>
mesh::compute_entity_permutations(const mesh::Topology& topology)
{
  common::Timer t_perm("Compute entity permutations");
  const int tdim = topology.dim();
  CellType cell_type = topology.cell_type();
  const std::int32_t num_cells = topology.connectivity(tdim, 0)->num_nodes();
  const int facets_per_cell = cell_num_entities(cell_type, tdim - 1);

  std::vector<std::uint32_t> cell_permutation_info(num_cells, 0);
  std::vector<std::uint8_t> facet_permutations(num_cells * facets_per_cell);
  std::int32_t used_bits = 0;
  if (tdim > 2)
  {
    spdlog::info("Compute face permutations");
    const int faces_per_cell = cell_num_entities(cell_type, 2);
    const auto face_perm = compute_face_permutations<_BITSETSIZE>(topology);
    for (int c = 0; c < num_cells; ++c)
      cell_permutation_info[c] = face_perm[c].to_ulong();

    // Currently, 3 bits are used for each face. If faces with more than
    // 4 sides are implemented, this will need to be increased.
    used_bits += faces_per_cell * 3;
    assert(tdim == 3);
    for (int c = 0; c < num_cells; ++c)
    {
      for (int i = 0; i < facets_per_cell; ++i)
      {
        facet_permutations[c * facets_per_cell + i]
            = (cell_permutation_info[c] >> (3 * i)) & 7;
      }
    }
  }

  if (tdim > 1)
  {
    spdlog::info("Compute edge permutations");
    const int edges_per_cell = cell_num_entities(cell_type, 1);
    const auto edge_perm = compute_edge_reflections<_BITSETSIZE>(topology);
    for (int c = 0; c < num_cells; ++c)
      cell_permutation_info[c] |= edge_perm[c].to_ulong() << used_bits;

    used_bits += edges_per_cell;
    if (tdim == 2)
    {
      for (int c = 0; c < num_cells; ++c)
      {
        for (int i = 0; i < facets_per_cell; ++i)
          facet_permutations[c * facets_per_cell + i] = edge_perm[c][i];
      }
    }
  }
  assert(used_bits < _BITSETSIZE);

  return {std::move(facet_permutations), std::move(cell_permutation_info)};
}
//-----------------------------------------------------------------------------
