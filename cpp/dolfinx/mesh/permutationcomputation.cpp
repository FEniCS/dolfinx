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
#include <dolfinx/common/Scatterer.h>
#include <dolfinx/graph/AdjacencyList.h>

namespace
{
constexpr int _BITSETSIZE = 32;
} // namespace

using namespace dolfinx;

namespace
{
//-----------------------------------------------------------------------------
template <int BITSETSIZE>
std::vector<std::bitset<BITSETSIZE>> compute_face_permutations_simplex(
    const graph::AdjacencyList<std::int32_t>& c_to_v,
    const graph::AdjacencyList<std::int32_t>& c_to_f,
    const graph::AdjacencyList<std::int32_t>& f_to_v, int faces_per_cell,
    const common::IndexMap& im)
{
  const std::int32_t num_cells = c_to_v.num_nodes();
  std::vector<std::bitset<BITSETSIZE>> face_perm(num_cells, 0);
  std::vector<std::int64_t> cell_vertices, vertices;
  for (int c = 0; c < num_cells; ++c)
  {
    cell_vertices.resize(c_to_v.links(c).size());
    im.local_to_global(c_to_v.links(c), cell_vertices);
    auto cell_faces = c_to_f.links(c);
    for (int i = 0; i < faces_per_cell; ++i)
    {
      // Get the face
      const int face = cell_faces[i];
      vertices.resize(f_to_v.links(face).size());
      im.local_to_global(f_to_v.links(face), vertices);

      // Orient that triangle so the lowest numbered vertex is the
      // origin, and the next vertex anticlockwise from the lowest has a
      // lower number than the next vertex clockwise. Find the index of
      // the lowest numbered vertex

      // Store local vertex indices here
      std::array<std::int32_t, 3> e_vertices;

      // Find iterators pointing to cell vertex given a vertex on facet
      for (int j = 0; j < 3; ++j)
      {
        auto it = std::find(cell_vertices.begin(), cell_vertices.end(),
                            vertices[j]);
        // Get the actual local vertex indices
        e_vertices[j] = std::distance(cell_vertices.begin(), it);
      }

      // Number of rotations
      std::uint8_t min_v = 0;
      for (int v = 1; v < 3; ++v)
        if (e_vertices[v] < e_vertices[min_v])
          min_v = v;

      // pre is the number of the next vertex clockwise from the lowest
      // numbered vertex
      const int pre = min_v == 0 ? e_vertices[3 - 1] : e_vertices[min_v - 1];

      // post is the number of the next vertex anticlockwise from the
      // lowest numbered vertex
      const int post = min_v == 3 - 1 ? e_vertices[0] : e_vertices[min_v + 1];

      std::uint8_t g_min_v = 0;
      for (int v = 1; v < 3; ++v)
        if (vertices[v] < vertices[g_min_v])
          g_min_v = v;

      // pre is the number of the next vertex clockwise from the lowest
      // numbered vertex
      const int g_pre = g_min_v == 0 ? vertices[3 - 1] : vertices[g_min_v - 1];

      // post is the number of the next vertex anticlockwise from the
      // lowest numbered vertex
      const int g_post = g_min_v == 3 - 1 ? vertices[0] : vertices[g_min_v + 1];

      std::uint8_t rots = 0;
      if (g_post > g_pre)
        rots = min_v <= g_min_v ? g_min_v - min_v : g_min_v + 3 - min_v;
      else
        rots = g_min_v <= min_v ? min_v - g_min_v : min_v + 3 - g_min_v;

      face_perm[c][3 * i] = (post > pre) == (g_post < g_pre);
      face_perm[c][3 * i + 1] = rots % 2;
      face_perm[c][3 * i + 2] = rots / 2;
    }
  }

  return face_perm;
}
//-----------------------------------------------------------------------------
template <int BITSETSIZE>
std::vector<std::bitset<BITSETSIZE>>
compute_face_permutations_tp(const graph::AdjacencyList<std::int32_t>& c_to_v,
                             const graph::AdjacencyList<std::int32_t>& c_to_f,
                             const graph::AdjacencyList<std::int32_t>& f_to_v,
                             int faces_per_cell, const common::IndexMap& im)
{
  const std::int32_t num_cells = c_to_v.num_nodes();
  std::vector<std::bitset<BITSETSIZE>> face_perm(num_cells, 0);
  std::vector<std::int64_t> cell_vertices, vertices;
  for (int c = 0; c < num_cells; ++c)
  {
    cell_vertices.resize(c_to_v.links(c).size());
    im.local_to_global(c_to_v.links(c), cell_vertices);

    auto cell_faces = c_to_f.links(c);
    for (int i = 0; i < faces_per_cell; ++i)
    {
      // Get the face
      const int face = cell_faces[i];
      vertices.resize(f_to_v.links(face).size());
      im.local_to_global(f_to_v.links(face), vertices);

      // Orient that triangle so the lowest numbered vertex is the
      // origin, and the next vertex anticlockwise from the lowest has a
      // lower number than the next vertex clockwise. Find the index of
      // the lowest numbered vertex

      // Store local vertex indices here
      std::array<std::int32_t, 4> e_vertices;

      // Find iterators pointing to cell vertex given a vertex on facet
      for (int j = 0; j < 4; ++j)
      {
        auto it = std::find(cell_vertices.begin(), cell_vertices.end(),
                            vertices[j]);
        // Get the actual local vertex indices
        e_vertices[j] = std::distance(cell_vertices.begin(), it);
      }

      // Number of rotations
      std::uint8_t min_v = 0;
      for (int v = 1; v < 4; ++v)
        if (e_vertices[v] < e_vertices[min_v])
          min_v = v;

      // pre is the (local) number of the next vertex clockwise from the
      // lowest numbered vertex
      int pre = 2;

      // post is the (local) number of the next vertex anticlockwise
      // from the lowest numbered vertex
      int post = 1;

      assert(min_v < 4);
      switch (min_v)
      {
      case 1:
        pre = 0;
        post = 3;
        break;
      case 2:
        pre = 3;
        post = 0;
        min_v = 3;
        break;
      case 3:
        pre = 1;
        post = 2;
        min_v = 2;
        break;
      }

      std::uint8_t g_min_v = 0;
      for (int v = 1; v < 4; ++v)
        if (vertices[v] < vertices[g_min_v])
          g_min_v = v;

      // rots is the number of rotations to get the lowest numbered
      // vertex to the origin
      // pre is the (local) number of the next vertex clockwise from the
      // lowest numbered vertex
      int g_pre = 2;

      // post is the (local) number of the next vertex anticlockwise
      // from the lowest numbered vertex
      int g_post = 1;

      assert(g_min_v < 4);
      switch (g_min_v)
      {
      case 1:
        g_pre = 0;
        g_post = 3;
        break;
      case 2:
        g_pre = 3;
        g_post = 0;
        g_min_v = 3;
        break;
      case 3:
        g_pre = 1;
        g_post = 2;
        g_min_v = 2;
        break;
      }

      std::uint8_t rots = 0;
      if (vertices[g_post] > vertices[g_pre])
        rots = min_v <= g_min_v ? g_min_v - min_v : g_min_v + 4 - min_v;
      else
        rots = g_min_v <= min_v ? min_v - g_min_v : min_v + 4 - g_min_v;

      face_perm[c][3 * i] = (e_vertices[post] > e_vertices[pre])
                            == (vertices[g_post] < vertices[g_pre]);
      face_perm[c][3 * i + 1] = rots % 2;
      face_perm[c][3 * i + 2] = rots / 2;
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
    cell_vertices.resize(c_to_v->links(c).size());
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
  const int tdim = topology.dim();
  assert(tdim > 2);
  if (!topology.index_map(2))
    throw std::runtime_error("Faces have not been computed.");

  // If faces have been computed, the below should exist
  auto c_to_v = topology.connectivity(tdim, 0);
  assert(c_to_v);
  auto c_to_f = topology.connectivity(tdim, 2);
  assert(c_to_f);
  auto f_to_v = topology.connectivity(2, 0);
  assert(f_to_v);

  auto im = topology.index_map(0);
  assert(im);

  mesh::CellType cell_type = topology.cell_type();
  const int faces_per_cell = cell_num_entities(cell_type, 2);
  if (cell_type == mesh::CellType::triangle
      or cell_type == mesh::CellType::tetrahedron)
  {
    return compute_face_permutations_simplex<BITSETSIZE>(
        *c_to_v, *c_to_f, *f_to_v, faces_per_cell, *im);
  }
  else
  {
    return compute_face_permutations_tp<BITSETSIZE>(*c_to_v, *c_to_f, *f_to_v,
                                                    faces_per_cell, *im);
  }
}
//-----------------------------------------------------------------------------
} // namespace

//-----------------------------------------------------------------------------
std::pair<std::vector<std::uint8_t>, std::vector<std::uint32_t>>
mesh::compute_entity_permutations(const mesh::Topology& topology)
{
  const int tdim = topology.dim();
  CellType cell_type = topology.cell_type();
  const std::int32_t num_cells = topology.connectivity(tdim, 0)->num_nodes();
  const int facets_per_cell = cell_num_entities(cell_type, tdim - 1);

  std::vector<std::uint32_t> cell_permutation_info(num_cells, 0);
  std::vector<std::uint8_t> facet_permutations(num_cells * facets_per_cell);
  std::int32_t used_bits = 0;
  if (tdim > 2)
  {
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
std::vector<std::uint8_t>
mesh::compute_cell_permutations(const mesh::Topology& topology)
{
  // TODO See if above functions can be called instead
  const int tdim = topology.dim();

  if (tdim == 3)
    throw std::runtime_error("Cannot compute cell permutations of a 3D mesh.");

  const std::int32_t num_cells = topology.index_map(tdim)->size_local()
                                 + topology.index_map(tdim)->num_ghosts();
  std::vector<std::uint8_t> cell_permutations(num_cells);

  if (tdim == 0)
    return cell_permutations;

  if (tdim == 2)
  {
    const auto perms = compute_face_permutations<_BITSETSIZE>(topology);
    for (int c = 0; c < num_cells; ++c)
      cell_permutations[c] = perms[c].to_ulong() & 7;
  }
  else if (tdim == 1)
  {
    const auto perms = compute_edge_reflections<_BITSETSIZE>(topology);
    for (int c = 0; c < num_cells; ++c)
      cell_permutations[c] = perms[c].to_ulong() & 1;
  }

  // TODO Move?
  // FIXME Can this be avoided?
  auto imap = topology.index_map(tdim);
  assert(imap);
  common::Scatterer scatterer(*imap, 1);
  scatterer.scatter_fwd(
      std::span<const std::uint8_t>(cell_permutations.begin(),
                                    cell_permutations.begin()
                                        + imap->size_local()),
      std::span<std::uint8_t>(cell_permutations.begin()
                                  + imap->size_local(),
                              cell_permutations.end()));

  return cell_permutations;
}
//-----------------------------------------------------------------------------
