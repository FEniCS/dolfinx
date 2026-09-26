// Copyright (C) 2020-2026 Matthew Scroggs and Jørgen S. Dokken
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
#include <cstddef>
#include <cstdint>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/MPI.h>
#include <dolfinx/common/Scatterer.h>
#include <dolfinx/common/Timer.h>
#include <dolfinx/common/local_range.h>
#include <dolfinx/common/log.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <format>
#include <functional>
#include <iterator>
#include <memory>
#include <mpi.h>
#include <numeric>
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
compute_triangle_rot_reflect(const std::vector<std::int32_t>& e_vertices,
                             const std::vector<std::int64_t>& vertices)
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
  CellType cell_type = topology.cell_type();
  const std::int32_t num_cells = topology.connectivity(tdim, 0)->num_nodes();

  std::vector<std::uint32_t> cell_permutation_info(num_cells, 0);
  std::int32_t used_bits = 0;
  if (tdim > 2)
  {
    // Each face occupies 3 bits: one reflection and two rotations. This
    // will need increasing if faces with more than 4 sides are added.
    spdlog::info("Compute face permutations");
    const std::vector<std::bitset<bitset_size>> face_perm
        = compute_face_permutations<bitset_size>(topology, num_threads);
    for (std::int32_t c = 0; c < num_cells; ++c)
      cell_permutation_info[c] = face_perm[c].to_ulong();

    used_bits += cell_num_entities(cell_type, 2) * 3;
  }

  if (tdim > 1)
  {
    spdlog::info("Compute edge permutations");
    const std::vector<std::bitset<bitset_size>> edge_perm
        = compute_edge_reflections<bitset_size>(topology, num_threads);
    for (std::int32_t c = 0; c < num_cells; ++c)
      cell_permutation_info[c] |= edge_perm[c].to_ulong() << used_bits;

    used_bits += cell_num_entities(cell_type, 1);
  }
  // At most bits 0 to 30 are taken, which leaves reversed_cell_bit free
  assert(used_bits < bitset_size);

  return cell_permutation_info;
}
//-----------------------------------------------------------------------------
namespace
{
/// Send rows of `width` values over a neighbourhood communicator,
/// `rows_per_dest[i]` of them to its i-th destination. Returns the rows
/// received, grouped by source, and the number from each source.
std::pair<std::vector<std::int64_t>, std::vector<std::int32_t>>
exchange_rows(MPI_Comm graph, std::vector<std::int64_t> send,
              std::span<const std::int32_t> rows_per_dest, std::size_t num_src,
              int width)
{
  std::vector<int> send_sizes(rows_per_dest.size());
  std::ranges::transform(rows_per_dest, send_sizes.begin(),
                         [width](std::int32_t n) { return n * width; });
  std::vector<int> recv_sizes(num_src);
  send_sizes.reserve(1);
  recv_sizes.reserve(1);
  MPI_Neighbor_alltoall(send_sizes.data(), 1, MPI_INT, recv_sizes.data(), 1,
                        MPI_INT, graph);

  std::vector<int> send_disp(send_sizes.size() + 1, 0);
  std::partial_sum(send_sizes.begin(), send_sizes.end(),
                   std::next(send_disp.begin()));
  std::vector<int> recv_disp(recv_sizes.size() + 1, 0);
  std::partial_sum(recv_sizes.begin(), recv_sizes.end(),
                   std::next(recv_disp.begin()));
  std::vector<std::int64_t> recv(recv_disp.back());
  send.reserve(1);
  recv.reserve(1);
  MPI_Neighbor_alltoallv(send.data(), send_sizes.data(), send_disp.data(),
                         MPI_INT64_T, recv.data(), recv_sizes.data(),
                         recv_disp.data(), MPI_INT64_T, graph);

  std::vector<std::int32_t> rows_per_src(num_src);
  std::ranges::transform(recv_sizes, rows_per_src.begin(),
                         [width](int n) { return n / width; });
  return {std::move(recv), std::move(rows_per_src)};
}

// Why a consistent orientation could not be found, combined over ranks
constexpr int not_orientable = 1;
constexpr int not_manifold = 2;
} // namespace

//-----------------------------------------------------------------------------
std::vector<std::int8_t>
mesh::compute_cell_orientations(const mesh::Topology& topology)
{
  common::Timer timer("Compute cell orientations");

  if (topology.dim() != 2)
  {
    throw std::invalid_argument(
        std::format("Cell orientations need a surface mesh (topological "
                    "dimension 2), not dimension {}.",
                    topology.dim()));
  }
  // Positions of the vertices in the order that goes round the cell. A
  // 2D cell is a triangle or a quadrilateral.
  const std::vector<int> cycle
      = topology.cell_type() == mesh::CellType::triangle
            ? std::vector{0, 1, 2}
            : std::vector{0, 1, 3, 2};

  auto c_to_v = topology.connectivity(2, 0);
  auto c_to_e = topology.connectivity(2, 1);
  auto e_to_v = topology.connectivity(1, 0);
  auto e_to_c = topology.connectivity(1, 2);
  if (!c_to_v or !c_to_e or !e_to_v or !e_to_c)
  {
    throw std::runtime_error("Edges and the connectivity between edges and "
                             "cells must be created first.");
  }

  MPI_Comm comm = topology.comm();
  const int comm_size = dolfinx::MPI::size(comm);
  std::shared_ptr<const common::IndexMap> cell_map = topology.index_map(2);
  std::shared_ptr<const common::IndexMap> edge_map = topology.index_map(1);
  std::shared_ptr<const common::IndexMap> vertex_map = topology.index_map(0);
  const std::int32_t num_owned = cell_map->size_local();
  const std::int32_t num_cells = num_owned + cell_map->num_ghosts();

  // Global indices of the vertices, to compare edge directions across
  // ranks
  std::vector<std::int64_t> global_vertex(vertex_map->size_local()
                                          + vertex_map->num_ghosts());
  std::iota(global_vertex.begin(),
            std::next(global_vertex.begin(), vertex_map->size_local()),
            vertex_map->local_range()[0]);
  std::ranges::copy(vertex_map->ghosts(),
                    std::next(global_vertex.begin(), vertex_map->size_local()));

  // Whether going round cell c in the order of its vertices runs edge e
  // from its lower to its higher global vertex
  auto runs_up = [&e_to_v, &c_to_v, &global_vertex,
                  &cycle](std::int32_t c, std::int32_t e) -> bool
  {
    std::span<const std::int32_t> ev = e_to_v->links(e);
    const auto [lo, hi] = global_vertex[ev[0]] < global_vertex[ev[1]]
                              ? std::pair(ev[0], ev[1])
                              : std::pair(ev[1], ev[0]);
    std::span<const std::int32_t> cv = c_to_v->links(c);
    for (std::size_t i = 0; i < cycle.size(); ++i)
    {
      const std::int32_t a = cv[cycle[i]];
      const std::int32_t b = cv[cycle[(i + 1) % cycle.size()]];
      if (a == lo and b == hi)
        return true;
      if (a == hi and b == lo)
        return false;
    }
    throw std::runtime_error("Edge is not an edge of the cell.");
  };

  // Throw on every rank if any rank found a problem. Collective.
  auto check_problems = [&comm](int problems)
  {
    MPI_Allreduce(MPI_IN_PLACE, &problems, 1, MPI_INT, MPI_BOR, comm);
    if (problems & not_manifold)
    {
      throw std::runtime_error("Cannot orient the cells: an edge is shared "
                               "by more than two cells.");
    }
    if (problems & not_orientable)
    {
      throw std::runtime_error("Cannot orient the cells: the surface is not "
                               "orientable (e.g. a Möbius strip).");
    }
  };

  // 1. Depth-first walk of each "part", a set of owned cells connected
  //    through shared edges (cells meeting only at a vertex are not),
  //    from its "root", the first owned cell not yet in a part. sign[c]
  //    is the orientation of c relative to the root. The walk stops at
  //    a cell reached along two routes with opposite signs, or at an
  //    edge with more than two cells.
  int problems = 0;
  std::vector<std::int8_t> sign(num_owned, 0); // Relative to the root
  std::vector<std::int32_t> part(num_owned, -1);
  std::int32_t num_parts = 0;
  {
    // Each owned cell is pushed at most once and ghosts never
    std::vector<std::int32_t> stack;
    stack.reserve(num_owned);
    for (std::int32_t root = 0; root < num_owned and problems == 0; ++root)
    {
      if (part[root] >= 0)
        continue;
      const std::int32_t p = num_parts++;
      part[root] = p;
      sign[root] = 1;
      stack.push_back(root);
      while (!stack.empty() and problems == 0)
      {
        const std::int32_t c = stack.back();
        stack.pop_back();
        for (std::int32_t e : c_to_e->links(c))
        {
          std::span<const std::int32_t> cells = e_to_c->links(e);
          if (cells.size() > 2)
          {
            problems |= not_manifold;
            break;
          }
          for (std::int32_t d : cells)
          {
            // Ghost cells are handled across ranks, in steps 2 and 3
            if (d == c or d >= num_owned)
              continue;
            // Running the shared edge the same way means they disagree
            const std::int8_t wanted
                = runs_up(c, e) == runs_up(d, e) ? -sign[c] : sign[c];
            if (part[d] < 0)
            {
              part[d] = p;
              sign[d] = wanted;
              stack.push_back(d);
            }
            else if (sign[d] != wanted) // Reached before, other sign
              problems |= not_orientable;
          }
        }
      }
    }
  }

  // Stop before the collective steps if any rank found a problem
  check_problems(problems);

  // 2. Pair the sides of each edge on a rank boundary, i.e. a ghost edge
  //    or an owned edge ghosted by another rank. Every owned cell on such
  //    an edge gives a "half" of it, sent to the "post office" of the
  //    edge, the rank that its global index maps to, which thus sees all
  //    cells of the edge. For each half: its owned cell, whether that
  //    cell runs the edge up, and the edge's global index.
  std::vector<std::int8_t> on_rank_boundary(
      edge_map->size_local() + edge_map->num_ghosts(), 0);
  for (std::int32_t e : edge_map->shared_indices())
    on_rank_boundary[e] = 1;
  std::fill(std::next(on_rank_boundary.begin(), edge_map->size_local()),
            on_rank_boundary.end(), 1);
  std::vector<std::int32_t> half_cell;
  std::vector<std::int8_t> half_up;
  std::vector<std::int32_t> half_edge_local;
  for (std::int32_t c = 0; c < num_owned; ++c)
  {
    for (std::int32_t e : c_to_e->links(c))
    {
      if (on_rank_boundary[e])
      {
        half_cell.push_back(c);
        half_up.push_back(runs_up(c, e));
        half_edge_local.push_back(e);
      }
    }
  }
  const std::size_t num_halves = half_cell.size();
  std::vector<std::int64_t> half_edge(num_halves);
  edge_map->local_to_global(half_edge_local, half_edge);

  // Rank, half index and direction of the other side of each half, or
  // -1 as rank if it has none
  std::vector<std::int32_t> partner_rank(num_halves, -1);
  std::vector<std::int64_t> partner_half(num_halves, -1);
  std::vector<std::int8_t> partner_up(num_halves, 0);
  {
    // Send (edge, direction, half index) to the post office of the edge,
    // grouped by office
    const std::int64_t num_edges = edge_map->size_global();
    std::vector<int> office(num_halves);
    for (std::size_t h = 0; h < num_halves; ++h)
      office[h] = dolfinx::MPI::index_owner(comm_size, half_edge[h], num_edges);
    std::vector<std::int32_t> perm(num_halves);
    std::iota(perm.begin(), perm.end(), 0);
    std::ranges::stable_sort(perm, {},
                             [&office](std::int32_t h) { return office[h]; });
    std::vector<int> dest;
    std::vector<std::int32_t> rows_per_dest;
    std::vector<std::int64_t> send;
    send.reserve(3 * num_halves);
    for (std::int32_t h : perm)
    {
      if (dest.empty() or dest.back() != office[h])
      {
        dest.push_back(office[h]);
        rows_per_dest.push_back(0);
      }
      ++rows_per_dest.back();
      send.insert(send.end(), {half_edge[h], half_up[h], h});
    }
    std::vector<int> src = dolfinx::MPI::compute_graph_edges_nbx(comm, dest);
    src.reserve(1);
    dest.reserve(1);
    MPI_Comm to_office;
    int ierr = MPI_Dist_graph_create_adjacent(
        comm, src.size(), src.data(), MPI_UNWEIGHTED, dest.size(), dest.data(),
        MPI_UNWEIGHTED, MPI_INFO_NULL, false, &to_office);
    dolfinx::MPI::check_error(comm, ierr);
    auto [recv, rows_per_src] = exchange_rows(to_office, std::move(send),
                                              rows_per_dest, src.size(), 3);
    MPI_Comm_free(&to_office);

    // Sort the received halves by edge, so the halves of an edge are
    // adjacent. Each entry is (edge, position of the sender in src, row
    // in recv).
    std::vector<std::array<std::int64_t, 3>> rows;
    for (std::size_t s = 0, r = 0; s < src.size(); ++s)
      for (std::int32_t i = 0; i < rows_per_src[s]; ++i, ++r)
        rows.push_back({recv[3 * r], static_cast<std::int64_t>(s),
                        static_cast<std::int64_t>(r)});
    std::ranges::sort(rows);

    // An edge with two halves from different ranks pairs them: reply to
    // each with (its half index, partner rank, partner half index,
    // partner direction), grouped by its rank. Two halves from the same
    // rank were joined by the walk, one half is on the boundary of the
    // surface, and more than two mean too many cells share the edge.
    std::vector<std::vector<std::int64_t>> replies(src.size());
    for (auto it = rows.begin(); it != rows.end();)
    {
      auto next = std::find_if(
          it, rows.end(), [e = (*it)[0]](const std::array<std::int64_t, 3>& row)
          { return row[0] != e; });
      if (std::distance(it, next) > 2)
        problems |= not_manifold;
      else if (std::distance(it, next) == 2 and (*it)[1] != (*(it + 1))[1])
      {
        for (int side = 0; side < 2; ++side)
        {
          const auto& self = *(it + side);
          const auto& other = *(it + 1 - side);
          replies[self[1]].insert(replies[self[1]].end(),
                                  {recv[3 * self[2] + 2], src[other[1]],
                                   recv[3 * other[2] + 2],
                                   recv[3 * other[2] + 1]});
        }
      }
      it = next;
    }
    std::vector<std::int64_t> reply;
    std::vector<std::int32_t> rows_per_src_reply;
    for (const std::vector<std::int64_t>& r : replies)
    {
      reply.insert(reply.end(), r.begin(), r.end());
      rows_per_src_reply.push_back(r.size() / 4);
    }
    MPI_Comm from_office;
    ierr = MPI_Dist_graph_create_adjacent(
        comm, dest.size(), dest.data(), MPI_UNWEIGHTED, src.size(), src.data(),
        MPI_UNWEIGHTED, MPI_INFO_NULL, false, &from_office);
    dolfinx::MPI::check_error(comm, ierr);
    auto [answers, rows_per_office] = exchange_rows(
        from_office, std::move(reply), rows_per_src_reply, dest.size(), 4);
    MPI_Comm_free(&from_office);
    for (std::size_t r = 0; r < answers.size() / 4; ++r)
    {
      const std::int64_t h = answers[4 * r];
      partner_rank[h] = answers[4 * r + 1];
      partner_half[h] = answers[4 * r + 2];
      partner_up[h] = answers[4 * r + 3];
    }
  }

  // 3. Merge the parts joined by paired halves into connected surfaces.
  //    Each part carries a "label", initially the global index of its
  //    root, and part_sign, initially +1, with the orientation of cell c
  //    being part_sign[part[c]] * sign[c]. Parts are numbered in the
  //    order of their roots, which are their first cells.
  std::vector<std::int64_t> label(num_parts);
  {
    const std::int64_t offset = cell_map->local_range()[0];
    for (std::int32_t c = 0, p = 0; p < num_parts; ++c)
    {
      if (part[c] == p)
        label[p++] = offset + c;
    }
  }
  std::vector<std::int8_t> part_sign(num_parts, 1);

  // Halves with a partner, grouped by partner rank, which are the
  // neighbours in the merge
  std::vector<std::int32_t> paired;
  for (std::size_t h = 0; h < num_halves; ++h)
    if (partner_rank[h] >= 0)
      paired.push_back(h);
  std::ranges::stable_sort(paired, {}, [&partner_rank](std::int32_t h)
                           { return partner_rank[h]; });
  std::vector<int> neighbours;
  std::vector<std::int32_t> rows_per_neighbour;
  for (std::int32_t h : paired)
  {
    if (neighbours.empty() or neighbours.back() != partner_rank[h])
    {
      neighbours.push_back(partner_rank[h]);
      rows_per_neighbour.push_back(0);
    }
    ++rows_per_neighbour.back();
  }
  neighbours.reserve(1);
  MPI_Comm graph;
  int ierr = MPI_Dist_graph_create_adjacent(
      comm, neighbours.size(), neighbours.data(), MPI_UNWEIGHTED,
      neighbours.size(), neighbours.data(), MPI_UNWEIGHTED, MPI_INFO_NULL,
      false, &graph);
  dolfinx::MPI::check_error(comm, ierr);

  // Send (partner half, label, orientation) for every paired half and
  // return what the partners sent
  auto exchange_state
      = [&paired, &half_cell, &partner_half, &label, &part, &part_sign, &sign,
         graph, &rows_per_neighbour, &neighbours]()
  {
    std::vector<std::int64_t> send;
    send.reserve(3 * paired.size());
    for (std::int32_t h : paired)
    {
      const std::int32_t c = half_cell[h];
      send.insert(send.end(), {partner_half[h], label[part[c]],
                               part_sign[part[c]] * sign[c]});
    }
    return exchange_rows(graph, std::move(send), rows_per_neighbour,
                         neighbours.size(), 3)
        .first;
  };

  // The orientation of the cell of half h that its partner's
  // orientation `o` implies
  auto implied
      = [&half_up, &partner_up](std::int64_t h, std::int64_t o) -> std::int8_t
  { return half_up[h] == partner_up[h] ? -o : o; };

  // Each round, every part takes the smallest label that its partners
  // hold, if smaller than its own, and the part_sign that makes it agree
  // with that partner. Once no label changes, each connected surface is
  // oriented relative to its cell with the lowest global index. The
  // number of rounds is bounded by the number of parts a surface spans.
  while (true)
  {
    std::vector<std::int64_t> recv = exchange_state();
    std::vector<std::int64_t> best_label = label;
    std::vector<std::int8_t> best_sign = part_sign;
    for (std::size_t r = 0; r < recv.size() / 3; ++r)
    {
      const std::int64_t h = recv[3 * r];
      const std::int32_t c = half_cell[h];
      const std::int32_t p = part[c];
      if (recv[3 * r + 1] < best_label[p])
      {
        best_label[p] = recv[3 * r + 1];
        best_sign[p] = implied(h, recv[3 * r + 2]) * sign[c];
      }
    }
    int changed = best_label != label;
    label = std::move(best_label);
    part_sign = std::move(best_sign);
    MPI_Allreduce(MPI_IN_PLACE, &changed, 1, MPI_INT, MPI_LOR, comm);
    if (!changed)
      break;
  }

  // Check every paired edge. A part joined to the surface along two
  // routes that imply opposite orientations means that the surface is
  // not orientable.
  {
    std::vector<std::int64_t> recv = exchange_state();
    for (std::size_t r = 0; r < recv.size() / 3; ++r)
    {
      const std::int64_t h = recv[3 * r];
      const std::int32_t c = half_cell[h];
      if (part_sign[part[c]] * sign[c] != implied(h, recv[3 * r + 2]))
        problems |= not_orientable;
    }
  }
  MPI_Comm_free(&graph);
  check_problems(problems);

  // 4. Orientations of the owned cells, then of the ghost cells from
  //    their owners
  std::vector<std::int8_t> orientations(num_cells);
  for (std::int32_t c = 0; c < num_owned; ++c)
    orientations[c] = part_sign[part[c]] * sign[c];
  common::Scatterer<> sc(*cell_map);
  const std::vector<std::int32_t>& local = sc.local_indices_block();
  const std::vector<std::int32_t>& remote = sc.remote_indices_block();
  std::vector<std::int8_t> send_buffer(local.size());
  std::ranges::transform(local, send_buffer.begin(),
                         [&orientations](std::int32_t c)
                         { return orientations[c]; });
  std::vector<std::int8_t> recv_buffer(remote.size());
  MPI_Request request = MPI_REQUEST_NULL;
  sc.scatter_fwd_begin(send_buffer.data(), recv_buffer.data(), 1, request);
  sc.scatter_fwd_end(request);
  for (std::size_t i = 0; i < remote.size(); ++i)
    orientations[num_owned + remote[i]] = recv_buffer[i];

  return orientations;
}
//-----------------------------------------------------------------------------
