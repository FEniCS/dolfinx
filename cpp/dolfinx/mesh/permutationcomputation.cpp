// Copyright (C) 2020-2026 Matthew Scroggs, Jørgen S. Dokken and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "permutationcomputation.h"
#include "Topology.h"
#include "cell_types.h"
#include <algorithm>
#include <array>
#include <bitset>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/Timer.h>
#include <dolfinx/common/local_range.h>
#include <dolfinx/common/log.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <format>
#include <functional>
#include <memory>
#include <ranges>
#include <span>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace
{
constexpr int bitset_size = 32;
} // namespace

using namespace dolfinx;

namespace
{
std::pair<std::int8_t, std::int8_t>
compute_triangle_rot_reflect(std::span<const int> e_vertices,
                             std::span<const std::int64_t> vertices)
{

  // Number of rotations
  std::uint8_t min_v = std::ranges::distance(
      e_vertices.begin(), std::ranges::min_element(e_vertices));

  // pre is the (local) number of the next vertex clockwise from the lowest
  // numbered vertex
  const int pre = e_vertices[(min_v + 2) % 3];

  // post is the (local) number of the next vertex anticlockwise from the
  // lowest numbered vertex
  const int post = e_vertices[(min_v + 1) % 3];

  std::uint8_t g_min_v = std::ranges::distance(
      vertices.begin(), std::ranges::min_element(vertices));

  // g_pre is the (global) number of the next vertex clockwise from the lowest
  // numbered vertex
  const std::int64_t g_pre = vertices[(g_min_v + 2) % 3];

  // g_post is the (global) number of the next vertex anticlockwise from the
  // lowest numbered vertex
  const std::int64_t g_post = vertices[(g_min_v + 1) % 3];

  std::uint8_t rots = 0;
  if (g_post > g_pre)
    rots = (g_min_v + 3 - min_v) % 3;
  else
    rots = (min_v + 3 - g_min_v) % 3;

  return {(post > pre) == (g_post < g_pre), rots};
}
//-----------------------------------------------------------------------------
std::pair<std::int8_t, std::int8_t>
compute_quad_rot_reflect(std::span<const int> e_vertices,
                         std::span<const std::int64_t> vertices)
{
  // Find minimum local cell vertex on facet
  std::uint8_t min_v = std::ranges::distance(
      e_vertices.begin(), std::ranges::min_element(e_vertices));

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
  std::uint8_t g_min_v = std::ranges::distance(
      vertices.begin(), std::ranges::min_element(vertices));

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
                                        int cell_index, int num_threads)
{
  common::Timer t_perm("* Compute triangle/quad face permutations");
  const std::vector<mesh::CellType>& cell_types = topology.entity_types(3);
  mesh::CellType cell_type = cell_types.at(cell_index);

  // Get face types of the cell and mesh
  const std::vector<mesh::CellType>& mesh_face_types = topology.entity_types(2);
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
    c_to_f.push_back(topology.connectivity({tdim, cell_index}, {2, int(i)}));
    f_to_v.push_back(topology.connectivity({2, int(i)}, {0, 0}));
  }

  auto c_to_v = topology.connectivity({tdim, cell_index}, {0, 0});
  assert(c_to_v);

  const std::int32_t num_cells = c_to_v->num_nodes();
  std::vector<std::bitset<BITSETSIZE>> face_perm(num_cells, 0);
  auto im = topology.index_map(0);

  auto process_thread
      = [](std::array<std::int64_t, 2> range, auto&& im, auto&& face_perm,
           auto&& face_type_indices, auto&& c_to_v, auto&& f_to_v,
           auto&& c_to_f, auto&& compute_refl_rots)
  {
    std::vector<std::int64_t> cell_vertices, vertices;
    std::vector<std::int32_t> e_vertices;
    for (std::int64_t c = range[0]; c < range[1]; ++c)
    {
      cell_vertices.resize(c_to_v->links(c).size());
      im->local_to_global(c_to_v->links(c), cell_vertices);
      auto cell_faces = c_to_f->links(c);
      for (std::size_t j = 0; j < cell_faces.size(); ++j)
      {
        // Get the face
        const int face = cell_faces[j];
        e_vertices.resize(f_to_v->num_links(face));
        vertices.resize(f_to_v->num_links(face));
        im->local_to_global(f_to_v->links(face), vertices);

        // Orient that triangle or quadrilateral so the lowest
        // numbered vertex is the origin, and the next vertex
        // anticlockwise from the lowest has a lower number than the
        // next vertex clockwise. Find the index of the lowest
        // numbered vertex.

        // Find iterators pointing to cell vertex given a vertex on
        // facet
        for (std::size_t k = 0; k < vertices.size(); ++k)
        {
          auto it = std::ranges::find(cell_vertices, vertices[k]);
          assert(it != cell_vertices.end());

          // Get the actual local vertex indices
          e_vertices[k] = std::ranges::distance(cell_vertices.begin(), it);
        }

        // Compute reflections and rotations for this face type
        auto [refl, rots] = compute_refl_rots(e_vertices, vertices);

        // Store bits for this face
        int fi = face_type_indices.get()[j];
        face_perm.get()[c][3 * fi] = refl;
        face_perm.get()[c][3 * fi + 1] = rots % 2;
        face_perm.get()[c][3 * fi + 2] = rots / 2;
      }
    }
  };

  for (std::size_t t = 0; t < face_type_indices.size(); ++t)
  {
    spdlog::info("Computing permutations for face type {}", t);
    if (!face_type_indices[t].empty())
    {
      auto compute_refl_rots = (mesh_face_types[t] == mesh::CellType::triangle)
                                   ? compute_triangle_rot_reflect
                                   : compute_quad_rot_reflect;
      assert(num_threads > 0);
      std::vector<std::jthread> threads;
      for (int i : std::ranges::iota_view(1, num_threads))
      {
        std::array range = common::local_range(i, num_cells, num_threads);
        threads.emplace_back(process_thread, range, im, std::ref(face_perm),
                             std::cref(face_type_indices[t]), c_to_v, f_to_v[t],
                             c_to_f[t], compute_refl_rots);
      }
      std::array range = common::local_range(0, num_cells, num_threads);
      process_thread(range, im, std::ref(face_perm),
                     std::cref(face_type_indices[t]), c_to_v, f_to_v[t],
                     c_to_f[t], compute_refl_rots);
    }
  }

  return face_perm;
}
//-----------------------------------------------------------------------------
template <int BITSETSIZE>
std::vector<std::bitset<BITSETSIZE>>
compute_edge_reflections(const mesh::Topology& topology, int num_threads)
{
  common::Timer t_perm("* Compute edge reflections");

  mesh::CellType cell_type = topology.cell_type();
  const int tdim = topology.dim();
  const int edges_per_cell = cell_num_entities(cell_type, 1);

  const std::int32_t num_cells = topology.connectivity(tdim, 0)->num_nodes();

  auto c_to_v = topology.connectivity(tdim, 0);
  assert(c_to_v);
  auto c_to_e = topology.connectivity(tdim, 1);
  if (!c_to_e)
    throw std::runtime_error("Edges have not been computed.");
  auto e_to_v = topology.connectivity(1, 0);
  if (!e_to_v)
  {
    throw std::runtime_error(
        "Edge-to-vertex connectivity has not been computed.");
  }

  auto im = topology.index_map(0);
  assert(im);

  std::vector<std::bitset<bitset_size>> edge_perm(num_cells, 0);
  auto process_thread
      = [](std::array<std::int64_t, 2> range, auto&& im, auto&& edge_perm,
           auto&& c_to_v, auto&& e_to_v, auto&& c_to_e, int num_edges)
  {
    std::vector<std::int64_t> cell_vertices;
    std::vector<std::int64_t> vertices;
    for (int c = range[0]; c < range[1]; ++c)
    {
      cell_vertices.resize(c_to_v->num_links(c));
      im->local_to_global(c_to_v->links(c), cell_vertices);
      auto cell_edges = c_to_e->links(c);
      for (int edge = 0; edge < num_edges; ++edge)
      {
        vertices.resize(e_to_v->links(cell_edges[edge]).size());
        im->local_to_global(e_to_v->links(cell_edges[edge]), vertices);

        // If the entity is an interval, it should be oriented pointing
        // from the lowest numbered vertex to the highest numbered vertex.

        // Find iterators pointing to cell vertex given a vertex on facet
        auto it0 = std::ranges::find(cell_vertices, vertices[0]);
        auto it1 = std::ranges::find(cell_vertices, vertices[1]);

        // The number of reflections. Comparing iterators directly instead
        // of values they point to is sufficient here.
        edge_perm.get()[c][edge] = (it1 < it0) == (vertices[1] > vertices[0]);
      }
    }
  };

  // Launch threads for computing edge reflections. The first thread is run in
  // the main task.

  std::vector<std::jthread> threads;
  assert(num_threads > 0);
  for (int i : std::ranges::iota_view(1, num_threads))
  {
    std::array<std::int64_t, 2> range
        = common::local_range(i, c_to_v->num_nodes(), num_threads);
    threads.emplace_back(process_thread, range, im, std::ref(edge_perm), c_to_v,
                         e_to_v, c_to_e, edges_per_cell);
  }
  std::array<std::int64_t, 2> range
      = common::local_range(0, c_to_v->num_nodes(), num_threads);
  process_thread(range, im, std::ref(edge_perm), c_to_v, e_to_v, c_to_e,
                 edges_per_cell);

  return edge_perm;
}
//-----------------------------------------------------------------------------
template <int BITSETSIZE>
std::vector<std::bitset<BITSETSIZE>>
compute_face_permutations(const mesh::Topology& topology, int num_threads)
{
  if (topology.entity_types(3).size() > 1)
  {
    throw std::runtime_error(
        "Cannot compute permutations for mixed topology mesh.");
  }

  [[maybe_unused]] const int tdim = topology.dim();
  assert(tdim > 2);
  if (!topology.index_map(2))
    throw std::runtime_error("Faces have not been computed.");

  // Compute face permutations for first cell type in the topology
  return compute_triangle_quad_face_permutations<BITSETSIZE>(topology, 0,
                                                             num_threads);
}
//-----------------------------------------------------------------------------
void compute_cell_permutations_range(
    std::array<std::int64_t, 2> range,
    const graph::AdjacencyList<std::int32_t>& c_to_v,
    const common::IndexMap& vertex_map, const graph::AdjacencyList<int>& edges,
    const graph::AdjacencyList<int>& faces,
    std::span<std::uint32_t> cell_permutation_info)
{
  const int edge_offset = 3 * faces.num_nodes();
  std::array<std::int64_t, 8> cell_vertices;
  std::array<std::int64_t, 4> face_vertices;
  for (std::int32_t c = range[0]; c < range[1]; ++c)
  {
    const std::span<const std::int32_t> vertices = c_to_v.links(c);
    assert(vertices.size() <= cell_vertices.size());
    vertex_map.local_to_global(vertices,
                               std::span(cell_vertices).first(vertices.size()));
    std::uint32_t info = 0;
    for (int f = 0; f < faces.num_nodes(); ++f)
    {
      const std::span<const int> e_vertices = faces.links(f);
      assert(e_vertices.size() <= face_vertices.size());
      for (std::size_t i = 0; i < e_vertices.size(); ++i)
        face_vertices[i] = cell_vertices[e_vertices[i]];
      const std::span<const std::int64_t> global_vertices
          = std::span(face_vertices).first(e_vertices.size());
      const auto [refl, rots]
          = e_vertices.size() == 3
                ? compute_triangle_rot_reflect(e_vertices, global_vertices)
                : compute_quad_rot_reflect(e_vertices, global_vertices);
      info |= std::uint32_t(refl + 2 * rots) << (3 * f);
    }
    for (int e = 0; e < edges.num_nodes(); ++e)
    {
      const std::span<const int> e_vertices = edges.links(e);
      const bool reflected
          = (e_vertices[1] < e_vertices[0])
            == (cell_vertices[e_vertices[1]] > cell_vertices[e_vertices[0]]);
      info |= std::uint32_t(reflected) << (edge_offset + e);
    }
    cell_permutation_info[c] = info;
  }
}
//-----------------------------------------------------------------------------
} // namespace

//-----------------------------------------------------------------------------
std::vector<std::uint8_t>
mesh::compute_entity_permutations(const mesh::Topology& topology, int dim,
                                  int num_threads)
{
  if (num_threads < 1)
    throw std::invalid_argument("num_threads must be >= 1.");

  const int tdim = topology.dim();
  if (dim < 0 or dim >= tdim)
  {
    throw std::invalid_argument(
        std::format("Cannot compute permutations for dimension {} entities of "
                    "a topology of dimension {}.",
                    dim, tdim));
  }

  // A vertex has no orientation, so there is nothing to permute.
  if (dim == 0)
    return {};

  common::Timer t_perm("Compute entity permutations");

  CellType cell_type = topology.cell_type();
  const std::int32_t num_cells = topology.connectivity(tdim, 0)->num_nodes();
  const int entities_per_cell = cell_num_entities(cell_type, dim);
  std::vector<std::uint8_t> perms(num_cells * entities_per_cell, 0);

  switch (dim)
  {
  case 1:
  {
    spdlog::info("Compute edge permutations");
    const std::vector<std::bitset<bitset_size>> edge_perm
        = compute_edge_reflections<bitset_size>(topology, num_threads);
    for (std::int32_t c = 0; c < num_cells; ++c)
      for (int i = 0; i < entities_per_cell; ++i)
        perms[c * entities_per_cell + i] = edge_perm[c][i];
    break;
  }
  case 2:
  {
    spdlog::info("Compute face permutations");
    const std::vector<std::bitset<bitset_size>> face_perm
        = compute_face_permutations<bitset_size>(topology, num_threads);
    // Three bits encode each face: one reflection bit and two rotation
    // bits.
    for (std::int32_t c = 0; c < num_cells; ++c)
    {
      for (int i = 0; i < entities_per_cell; ++i)
      {
        perms[c * entities_per_cell + i]
            = (face_perm[c].to_ulong() >> (3 * i)) & 7;
      }
    }
    break;
  }
  default:
    throw std::invalid_argument(std::format(
        "Permutations of dimension {} entities are not supported.", dim));
  }

  return perms;
}
//-----------------------------------------------------------------------------

std::vector<std::uint32_t>
mesh::compute_cell_permutations(const mesh::Topology& topology, int num_threads)
{
  if (num_threads < 1)
    throw std::invalid_argument("num_threads must be >= 1.");

  common::Timer t_perm("Compute cell permutations");

  const int tdim = topology.dim();
  const CellType cell_type = topology.cell_type();
  const auto c_to_v = topology.connectivity(tdim, 0);
  assert(c_to_v);
  const auto vertex_map = topology.index_map(0);
  assert(vertex_map);
  const std::int32_t num_cells = c_to_v->num_nodes();

  // Reference-cell entities identify vertices without constructing mesh
  // edges or faces and their distributed index maps.
  const graph::AdjacencyList<int> edges
      = tdim > 1 ? get_entity_vertices(cell_type, 1)
                 : graph::AdjacencyList<int>(0);
  const graph::AdjacencyList<int> faces
      = tdim > 2 ? get_entity_vertices(cell_type, 2)
                 : graph::AdjacencyList<int>(0);
  assert(3 * faces.num_nodes() + edges.num_nodes() < bitset_size);
  std::vector<std::uint32_t> cell_permutation_info(num_cells, 0);

  {
    std::vector<std::jthread> threads;
    for (int i = 1; i < num_threads; ++i)
      threads.emplace_back(compute_cell_permutations_range,
                           common::local_range(i, num_cells, num_threads),
                           std::cref(*c_to_v), std::cref(*vertex_map),
                           std::cref(edges), std::cref(faces),
                           std::span(cell_permutation_info));
    compute_cell_permutations_range(
        common::local_range(0, num_cells, num_threads), *c_to_v, *vertex_map,
        edges, faces, cell_permutation_info);
  }

  return cell_permutation_info;
}
//-----------------------------------------------------------------------------
