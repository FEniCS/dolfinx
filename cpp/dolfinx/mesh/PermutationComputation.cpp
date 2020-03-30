// Copyright (C) 2020 Matthew Scroggs
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "PermutationComputation.h"
#include <bitset>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/mesh/Topology.h>

#define BITSETSIZE 32

using namespace dolfinx;
using namespace dolfinx::mesh;

namespace
{
Eigen::Array<std::bitset<BITSETSIZE>, Eigen::Dynamic, 1>
compute_face_permutations_simplex(
    const graph::AdjacencyList<std::int32_t>& c_to_v,
    const graph::AdjacencyList<std::int32_t>& c_to_f,
    const graph::AdjacencyList<std::int32_t>& f_to_v, int faces_per_cell)
{
  const std::int32_t num_cells = c_to_v.num_nodes();
  Eigen::Array<std::bitset<BITSETSIZE>, Eigen::Dynamic, 1> face_perm(num_cells);
  face_perm.fill(0);

  for (int c = 0; c < num_cells; ++c)
  {
    auto cell_vertices = c_to_v.links(c);
    auto cell_faces = c_to_f.links(c);
    for (int i = 0; i < faces_per_cell; ++i)
    {
      // Get the face
      const int face = cell_faces[i];
      auto vertices = f_to_v.links(face);

      // Orient that triangle so the the lowest numbered vertex is the
      // origin, and the next vertex anticlockwise from the lowest has a
      // lower number than the next vertex clockwise. Find the index of
      // the lowest numbered vertex

      // Store local vertex indices here
      std::array<std::int32_t, 3> e_vertices;

      // Find iterators pointing to cell vertex given a vertex on facet
      for (int j = 0; j < 3; ++j)
      {
        const auto *const it = std::find(cell_vertices.data(),
                                  cell_vertices.data() + cell_vertices.size(),
                                  vertices[j]);
        // Get the actual local vertex indices
        e_vertices[j] = it - cell_vertices.data();
      }

      // Number of rotations
      std::uint8_t rots = 0;
      for (int v = 1; v < 3; ++v)
        if (e_vertices[v] < e_vertices[rots])
          rots = v;

      // pre is the number of the next vertex clockwise from the lowest
      // numbered vertex
      const int pre = rots == 0 ? e_vertices[3 - 1] : e_vertices[rots - 1];

      // post is the number of the next vertex anticlockwise from the
      // lowest numbered vertex
      const int post = rots == 3 - 1 ? e_vertices[0] : e_vertices[rots + 1];

      face_perm[c][3 * i] = (post > pre);
      face_perm[c][3 * i + 1] = rots % 2;
      face_perm[c][3 * i + 2] = rots / 2;
    }
  }
  return face_perm;
}
//-----------------------------------------------------------------------------
Eigen::Array<std::bitset<BITSETSIZE>, Eigen::Dynamic, 1>
compute_face_permutations_tp(const graph::AdjacencyList<std::int32_t>& c_to_v,
                             const graph::AdjacencyList<std::int32_t>& c_to_f,
                             const graph::AdjacencyList<std::int32_t>& f_to_v,
                             int faces_per_cell)
{
  const std::int32_t num_cells = c_to_v.num_nodes();
  Eigen::Array<std::bitset<BITSETSIZE>, Eigen::Dynamic, 1> face_perm(num_cells);
  face_perm.fill(0);
  for (int c = 0; c < num_cells; ++c)
  {
    auto cell_vertices = c_to_v.links(c);
    auto cell_faces = c_to_f.links(c);
    for (int i = 0; i < faces_per_cell; ++i)
    {
      const int face = cell_faces[i];
      auto vertices = f_to_v.links(face);

      // quadrilateral
      // Orient that quad so the the lowest numbered vertex is the
      // origin, and the next vertex anticlockwise from the lowest has a
      // lower number than the next vertex clockwise. Find the index of
      // the lowest numbered vertex
      int num_min = -1;

      // Store local vertex indices here
      std::array<std::int32_t, 4> e_vertices;

      // Find iterators pointing to cell vertex given a vertex on facet
      for (int j = 0; j < 4; ++j)
      {
        const auto *const it = std::find(cell_vertices.data(),
                                  cell_vertices.data() + cell_vertices.size(),
                                  vertices[j]);
        // Get the actual local vertex indices
        e_vertices[j] = it - cell_vertices.data();
      }

      for (int v = 0; v < 4; ++v)
      {
        if (num_min == -1 or e_vertices[v] < e_vertices[num_min])
          num_min = v;
      }

      // rots is the number of rotations to get the lowest numbered
      // vertex to the origin
      std::uint8_t rots = num_min;

      // pre is the (local) number of the next vertex clockwise from the
      // lowest numbered vertex
      int pre = 2;

      // post is the (local) number of the next vertex anticlockwise
      // from the lowest numbered vertex
      int post = 1;

      // The tensor product ordering of quads must be taken into account
      assert(num_min < 4);
      switch (num_min)
      {
      case 1:
        pre = 0;
        post = 3;
        break;
      case 2:
        pre = 3;
        post = 0;
        rots = 3;
        break;
      case 3:
        pre = 1;
        post = 2;
        rots = 2;
        break;
      }

      face_perm[c][3 * i] = (post > pre);
      face_perm[c][3 * i + 1] = rots % 2;
      face_perm[c][3 * i + 2] = rots / 2;
    }
  }
  return face_perm;
}
//-----------------------------------------------------------------------------
Eigen::Array<std::bitset<BITSETSIZE>, Eigen::Dynamic, 1>
compute_edge_reflections(const mesh::Topology& topology)
{
  const int tdim = topology.dim();
  const CellType cell_type = topology.cell_type();
  const int edges_per_cell = cell_num_entities(cell_type, 1);

  const std::int32_t num_cells = topology.connectivity(tdim, 0)->num_nodes();

  Eigen::Array<std::bitset<BITSETSIZE>, Eigen::Dynamic, 1> edge_perm(num_cells);
  edge_perm.fill(0);

  auto c_to_v = topology.connectivity(tdim, 0);
  assert(c_to_v);
  auto c_to_e = topology.connectivity(tdim, 1);
  assert(c_to_e);
  auto e_to_v = topology.connectivity(1, 0);
  assert(e_to_v);

  Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> reflections(
      edges_per_cell, c_to_v->num_nodes());
  for (int c = 0; c < c_to_v->num_nodes(); ++c)
  {
    auto cell_vertices = c_to_v->links(c);
    auto cell_edges = c_to_e->links(c);
    for (int i = 0; i < edges_per_cell; ++i)
    {
      auto vertices = e_to_v->links(cell_edges[i]);

      // If the entity is an interval, it should be oriented pointing
      // from the lowest numbered vertex to the highest numbered vertex.

      // Find iterators pointing to cell vertex given a vertex on facet
      const auto *const it0
          = std::find(cell_vertices.data(),
                      cell_vertices.data() + cell_vertices.size(), vertices[0]);
      const auto *const it1
          = std::find(cell_vertices.data(),
                      cell_vertices.data() + cell_vertices.size(), vertices[1]);

      // The number of reflections. Comparing iterators directly instead
      // of values they point to is sufficient here.
      edge_perm[c][i] = (it1 < it0);
    }
  }
  return edge_perm;
}
//-----------------------------------------------------------------------------
Eigen::Array<std::bitset<BITSETSIZE>, Eigen::Dynamic, 1>
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

  const CellType cell_type = topology.cell_type();
  const int faces_per_cell = cell_num_entities(cell_type, 2);
  if (cell_type == CellType::triangle or cell_type == CellType::tetrahedron)
  {
    return compute_face_permutations_simplex(*c_to_v, *c_to_f, *f_to_v,
                                             faces_per_cell);
  }
  else
  {
    return compute_face_permutations_tp(*c_to_v, *c_to_f, *f_to_v,
                                        faces_per_cell);
  }
}
//-----------------------------------------------------------------------------
} // namespace

//-----------------------------------------------------------------------------
std::pair<Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>,
          Eigen::Array<std::uint32_t, Eigen::Dynamic, 1>>
PermutationComputation::compute_entity_permutations(
    const mesh::Topology& topology)
{
  const int tdim = topology.dim();
  const CellType cell_type = topology.cell_type();
  assert(topology.connectivity(tdim, 0));
  const std::int32_t num_cells = topology.connectivity(tdim, 0)->num_nodes();
  const int facets_per_cell = cell_num_entities(cell_type, tdim - 1);

  Eigen::Array<std::uint32_t, Eigen::Dynamic, 1> cell_permutation_info
      = Eigen::Array<std::uint32_t, Eigen::Dynamic, 1>::Zero(num_cells);
  Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic> facet_permutations(
      facets_per_cell, num_cells);

  std::int32_t used_bits = 0;
  if (tdim > 2)
  {
    const int faces_per_cell = cell_num_entities(cell_type, 2);
    Eigen::Array<std::bitset<BITSETSIZE>, Eigen::Dynamic, 1> face_perm
        = compute_face_permutations(topology);
    for (int c = 0; c < num_cells; ++c)
      cell_permutation_info[c] = face_perm[c].to_ulong();

    // Currently, 3 bits are used for each face. If faces with more than
    // 4 sides are implemented, this will need to be increased.
    used_bits += faces_per_cell * 3;
    assert(tdim == 3);
    for (int c = 0; c < num_cells; ++c)
    {
      for (int i = 0; i < facets_per_cell; ++i)
        facet_permutations(i, c) = (cell_permutation_info[c] >> (3 * i)) & 7;
    }
  }

  if (tdim > 1)
  {
    const int edges_per_cell = cell_num_entities(cell_type, 1);
    Eigen::Array<std::bitset<BITSETSIZE>, Eigen::Dynamic, 1> edge_perm
        = compute_edge_reflections(topology);
    for (int c = 0; c < num_cells; ++c)
      cell_permutation_info[c] |= edge_perm[c].to_ulong() << used_bits;

    used_bits += edges_per_cell;
    if (tdim == 2)
    {
      for (int c = 0; c < num_cells; ++c)
        for (int i = 0; i < facets_per_cell; ++i)
          facet_permutations(i, c) = edge_perm[c][i];
    }
  }

  assert(used_bits < BITSETSIZE);

  return {std::move(facet_permutations), std::move(cell_permutation_info)};
}
//-----------------------------------------------------------------------------
