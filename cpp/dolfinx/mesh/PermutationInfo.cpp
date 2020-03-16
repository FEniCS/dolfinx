// Copyright (C) 2020 Matthew Scroggs
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "PermutationInfo.h"
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/graph/Partitioning.h>
#include <dolfinx/mesh/Topology.h>

using namespace dolfinx;
using namespace dolfinx::mesh;

namespace
{
std::pair<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic>,
          Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>>
compute_face_permutations_simplex(
    const graph::AdjacencyList<std::int32_t>& c_to_v,
    const graph::AdjacencyList<std::int32_t>& c_to_f,
    const graph::AdjacencyList<std::int32_t>& f_to_v, int faces_per_cell)
{
  Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> reflections(
      faces_per_cell, c_to_v.num_nodes());
  Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic> rotations(
      faces_per_cell, c_to_v.num_nodes());
  for (int c = 0; c < c_to_v.num_nodes(); ++c)
  {
    auto cell_vertices = c_to_v.links(c);
    auto cell_faces = c_to_f.links(c);
    for (int i = 0; i < faces_per_cell; ++i)
    {
      // Get the face
      const int face = cell_faces[i];
      auto vertices = f_to_v.links(face);

      // Orient that triangle so the the lowest numbered vertex is
      // the origin, and the next vertex anticlockwise from the
      // lowest has a lower number than the next vertex clockwise.
      // Find the index of the lowest numbered vertex

      // Store local vertex indices here
      std::array<std::int32_t, 3> e_vertices;

      // Find iterators pointing to cell vertex given a vertex on
      // facet
      for (int j = 0; j < 3; ++j)
      {
        const auto it = std::find(cell_vertices.data(),
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

      // pre is the number of the next vertex clockwise from the
      // lowest numbered vertex
      const int pre = rots == 0 ? e_vertices[3 - 1] : e_vertices[rots - 1];

      // post is the number of the next vertex anticlockwise from
      // the lowest numbered vertex
      const int post = rots == 3 - 1 ? e_vertices[0] : e_vertices[rots + 1];

      // The number of reflections
      const std::uint8_t refs = post > pre;

      reflections(i, c) = refs;
      rotations(i, c) = rots;
    }
  }

  return {std::move(reflections), std::move(rotations)};
}
//-----------------------------------------------------------------------------
std::pair<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic>,
          Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>>
compute_face_permutations_tp(const graph::AdjacencyList<std::int32_t>& c_to_v,
                             const graph::AdjacencyList<std::int32_t>& c_to_f,
                             const graph::AdjacencyList<std::int32_t>& f_to_v,
                             int faces_per_cell)
{
  Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> reflections(
      faces_per_cell, c_to_v.num_nodes());
  Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic> rotations(
      faces_per_cell, c_to_v.num_nodes());
  for (int c = 0; c < c_to_v.num_nodes(); ++c)
  {
    auto cell_vertices = c_to_v.links(c);
    auto cell_faces = c_to_f.links(c);
    for (int i = 0; i < faces_per_cell; ++i)
    {
      const int face = cell_faces[i];
      auto vertices = f_to_v.links(face);

      // quadrilateral
      // Orient that quad so the the lowest numbered vertex is the origin,
      // and the next vertex anticlockwise from the lowest has a lower
      // number than the next vertex clockwise. Find the index of the
      // lowest numbered vertex
      int num_min = -1;

      // Store local vertex indices here
      std::array<std::int32_t, 4> e_vertices;
      // Find iterators pointing to cell vertex given a vertex on facet
      for (int j = 0; j < 4; ++j)
      {
        const auto it = std::find(cell_vertices.data(),
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

      // rots is the number of rotations to get the lowest numbered vertex
      // to the origin
      std::uint8_t rots = num_min;

      // pre is the (local) number of the next vertex clockwise from the
      // lowest numbered vertex
      int pre = 2;

      // post is the (local) number of the next vertex anticlockwise from
      // the lowest numbered vertex
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

      // The number of reflections
      const std::uint8_t refs = (e_vertices[post] > e_vertices[pre]);

      reflections(i, c) = refs;
      rotations(i, c) = rots;
    }
  }

  return {std::move(reflections), std::move(rotations)};
}
//-----------------------------------------------------------------------------
} // namespace

//-----------------------------------------------------------------------------
Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic>
mesh::compute_edge_reflections(const Topology& topology)
{
  const int tdim = topology.dim();
  const CellType cell_type = topology.cell_type();
  const int edges_per_cell = cell_num_entities(cell_type, 1);

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
      const auto it0
          = std::find(cell_vertices.data(),
                      cell_vertices.data() + cell_vertices.size(), vertices[0]);
      const auto it1
          = std::find(cell_vertices.data(),
                      cell_vertices.data() + cell_vertices.size(), vertices[1]);

      // The number of reflections. Comparing iterators directly instead
      // of values they point to is sufficient here.
      reflections(i, c) = (it1 < it0);
    }
  }

  return reflections;
}
//-----------------------------------------------------------------------------
std::pair<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic>,
          Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>>
mesh::compute_face_permutations(const Topology& topology)
{
  const int tdim = topology.dim();
  assert(tdim > 2);
  if (!topology.index_map(2))
    throw std::runtime_error("Faces have not been computed");

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

//-----------------------------------------------------------------------------
PermutationInfo::PermutationInfo()
    : _edge_reflections(0, 0), _face_reflections(0, 0), _face_rotations(0, 0),
      _facet_permutations(0, 0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
const Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic>&
PermutationInfo::get_edge_reflections() const
{
  return _edge_reflections;
}
//-----------------------------------------------------------------------------
const Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic>&
PermutationInfo::get_face_reflections() const
{
  return _face_reflections;
}
//-----------------------------------------------------------------------------
const Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>&
PermutationInfo::get_face_rotations() const
{
  return _face_rotations;
}
//-----------------------------------------------------------------------------
const Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>&
PermutationInfo::get_facet_permutations() const
{
  return _facet_permutations;
}
//-----------------------------------------------------------------------------
void PermutationInfo::create_entity_permutations(mesh::Topology& topology)
{
  if (_edge_reflections.rows() > 0)
  {
    // assert(_face_reflections.size() != 0);
    // assert(_face_rotations.size() != 0);
    // assert(_facet_permutations.size() != 0);
    return;
  }

  const int tdim = topology.dim();
  const CellType cell_type = topology.cell_type();
  assert(topology.connectivity(tdim, 0));
  const std::int32_t num_cells = topology.connectivity(tdim, 0)->num_nodes();
  const int faces_per_cell = cell_num_entities(cell_type, 2);

  // FIXME: Avoid create 'identity' reflections/rotations

  if (tdim <= 1)
  {
    const int edges_per_cell = cell_num_entities(cell_type, 1);
    const int facets_per_cell = cell_num_entities(cell_type, tdim - 1);
    _edge_reflections.resize(edges_per_cell, num_cells);
    _edge_reflections = false;

    _face_reflections.resize(faces_per_cell, num_cells);
    _face_reflections = false;

    _face_rotations.resize(faces_per_cell, num_cells);
    _face_rotations = 0;
    _facet_permutations.resize(facets_per_cell, num_cells);
    _facet_permutations = 0;
  }

  if (tdim > 1)
  {
    _face_reflections.resize(faces_per_cell, num_cells);
    _face_reflections = false;
    _face_rotations.resize(faces_per_cell, num_cells);
    _face_rotations = 0;

    Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> reflections
        = mesh::compute_edge_reflections(topology);
    _edge_reflections = reflections;
    if (tdim == 2)
      _facet_permutations = reflections.cast<std::uint8_t>();
  }

  if (tdim > 2)
  {
    auto [reflections, rotations] = mesh::compute_face_permutations(topology);
    _face_reflections = reflections;
    _face_rotations = rotations;
    if (tdim == 3)
      _facet_permutations = 2 * rotations + reflections.cast<std::uint8_t>();
  }
}
//-----------------------------------------------------------------------------
