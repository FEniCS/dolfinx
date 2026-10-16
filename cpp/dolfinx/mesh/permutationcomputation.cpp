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
#include <dolfinx/common/Scatterer.h>
#include <dolfinx/common/Timer.h>
#include <dolfinx/common/local_range.h>
#include <dolfinx/common/log.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <format>
#include <functional>
#include <iterator>
#include <limits>
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
  if (topology.index_maps(2).size() != topology.entity_types(2).size())
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
  // A hexahedron takes bits 0 to 29, overlapping the facet sign bits 27
  // to 30. Those are used only for tdim == 2, which takes one bit per
  // edge.
  assert(used_bits < bitset_size);
  assert(tdim != 2 or used_bits <= 27);

  return cell_permutation_info;
}
//-----------------------------------------------------------------------------
namespace
{
// Whether walking round a cell in its vertex order runs Basix edge
// i = [p, q] from p to q. Round a triangle (0-1-2) the walk runs its
// edges [1, 2], [0, 2], [0, 1] forward, back, forward, and round a
// quadrilateral (0-1-3-2) its edges [0, 1], [0, 2], [1, 3], [2, 3]
// forward, back, forward, back. Valid only for these two cell types, so
// a new cell type of dimension 2 needs its own table.
constexpr std::array<bool, 4> edge_runs_forward = {true, false, true, false};
} // namespace

std::vector<std::uint32_t>
mesh::compute_facet_orientations(const mesh::Topology& topology)
{
  common::Timer timer("Compute facet orientations");

  assert(topology.dim() == 2);
  std::shared_ptr<const graph::AdjacencyList<std::int32_t>> c_to_e
      = topology.connectivity(2, 1);
  std::shared_ptr<const common::IndexMap> edge_map = topology.index_map(1);
  if (!c_to_e or !edge_map)
    throw std::runtime_error("Edges must be created first.");

  std::shared_ptr<const common::IndexMap> cell_map = topology.index_map(2);
  const std::int32_t num_owned = cell_map->size_local();
  const std::int32_t num_cells = num_owned + cell_map->num_ghosts();
  const std::int32_t num_owned_edges = edge_map->size_local();
  const std::int32_t num_edges = num_owned_edges + edge_map->num_ghosts();
  const int edges_per_cell = mesh::cell_num_entities(topology.cell_type(), 1);

  // Whether walking round cell c in its vertex order runs its local
  // edge i from the lower to the higher global vertex
  const std::vector<std::uint8_t>& edge_perms
      = topology.get_entity_permutations(1);
  assert(std::cmp_less_equal(edges_per_cell, edge_runs_forward.size()));
  auto runs_up
      = [&edge_perms, edges_per_cell](std::int32_t c, std::size_t i) -> bool
  {
    const bool reflected = edge_perms[c * edges_per_cell + i] % 2;
    return edge_runs_forward[i] != reflected;
  };

  // Code of cell c and its local edge i: 2 * (global index of c) +
  // runs_up(c, i). It packs the cell and its direction along the edge
  // into one integer that is the same on every process. Codes order as
  // global cell indices do, so the larger code is the cell with the
  // larger global index, and code % 2 is its direction.
  std::vector<std::int32_t> local(num_cells);
  std::iota(local.begin(), local.end(), 0);
  std::vector<std::int64_t> global(num_cells);
  cell_map->local_to_global(local, global);
  auto code = [&global, &runs_up](std::int32_t c, std::size_t i)
  { return 2 * global[c] + runs_up(c, i); };

  // Per edge, over its owned cells on all processes, edge_data holds
  // values_per_edge = 3 values (the block size of the scatters below):
  //   [0] the largest code,
  //   [1] the negated smallest code, so that both reduce with max,
  //   [2] the number of cells.
  // Each cell is owned once, so this counts it once, and with two cells
  // the largest and smallest code are the two cells. Combined by the
  // owner of the edge, then sent back to every process holding it. No
  // ghost cells are needed.
  constexpr int values_per_edge = 3;
  constexpr std::int64_t none = std::numeric_limits<std::int64_t>::min();
  std::vector<std::int64_t> edge_data(values_per_edge * num_edges, 0);
  for (std::int32_t e = 0; e < num_edges; ++e)
  {
    edge_data[values_per_edge * e] = none;
    edge_data[values_per_edge * e + 1] = none;
  }
  auto combine = [&edge_data](std::int32_t e, std::int64_t hi,
                              std::int64_t neg_lo, std::int64_t count)
  {
    std::int64_t* d = edge_data.data() + values_per_edge * e;
    d[0] = std::max(d[0], hi);
    d[1] = std::max(d[1], neg_lo);
    d[2] += count;
  };
  for (std::int32_t c = 0; c < num_owned; ++c)
  {
    std::span<const std::int32_t> edges = c_to_e->links(c);
    for (std::size_t i = 0; i < edges.size(); ++i)
      combine(edges[i], code(c, i), -code(c, i), 1);
  }

  common::Scatterer<> sc(*edge_map);
  const std::vector<std::int32_t>& owned_idx = sc.local_indices_block();
  const std::vector<std::int32_t>& ghost_idx = sc.remote_indices_block();
  {
    std::vector<std::int64_t> send(values_per_edge * ghost_idx.size());
    for (std::size_t j = 0; j < ghost_idx.size(); ++j)
    {
      std::ranges::copy_n(
          std::next(edge_data.begin(),
                    values_per_edge * (num_owned_edges + ghost_idx[j])),
          values_per_edge, std::next(send.begin(), values_per_edge * j));
    }
    std::vector<std::int64_t> recv(values_per_edge * owned_idx.size());
    MPI_Request request = MPI_REQUEST_NULL;
    sc.scatter_rev_begin(send.data(), recv.data(), values_per_edge, request);
    sc.scatter_rev_end(request);
    for (std::size_t j = 0; j < owned_idx.size(); ++j)
    {
      combine(owned_idx[j], recv[values_per_edge * j],
              recv[values_per_edge * j + 1], recv[values_per_edge * j + 2]);
    }
  }
  {
    std::vector<std::int64_t> send(values_per_edge * owned_idx.size());
    for (std::size_t j = 0; j < owned_idx.size(); ++j)
    {
      std::ranges::copy_n(
          std::next(edge_data.begin(), values_per_edge * owned_idx[j]),
          values_per_edge, std::next(send.begin(), values_per_edge * j));
    }
    std::vector<std::int64_t> recv(values_per_edge * ghost_idx.size());
    MPI_Request request = MPI_REQUEST_NULL;
    sc.scatter_fwd_begin(send.data(), recv.data(), values_per_edge, request);
    sc.scatter_fwd_end(request);
    for (std::size_t j = 0; j < ghost_idx.size(); ++j)
    {
      std::ranges::copy_n(
          std::next(recv.begin(), values_per_edge * j), values_per_edge,
          std::next(edge_data.begin(),
                    values_per_edge * (num_owned_edges + ghost_idx[j])));
    }
  }

  // A facet is signed when its two cells run it the same way, i.e. give
  // it opposite normals (codes of equal parity), on the cell with the
  // larger global index (the larger code). Each process holds every edge
  // of its ghost cells, so ghosts are signed as by their owner. Errors
  // are raised only after the collective calls above, so that no
  // process leaves while its peers wait in them.
  bool non_manifold = false;
  std::vector<std::uint32_t> flags(num_cells, 0);
  for (std::int32_t c = 0; c < num_cells; ++c)
  {
    std::span<const std::int32_t> edges = c_to_e->links(c);
    for (std::size_t i = 0; i < edges.size(); ++i)
    {
      const std::int64_t* d = edge_data.data() + values_per_edge * edges[i];
      if (d[2] > 2)
        non_manifold = true;
      else if (d[2] == 2)
      {
        const std::int64_t hi = d[0];
        const std::int64_t lo = -d[1];
        if (code(c, i) == hi and (hi - lo) % 2 == 0)
          flags[c] |= reversed_facet_bit(i);
      }
    }
  }

#ifndef NDEBUG
  // Make every process throw, or none
  int error = non_manifold;
  MPI_Allreduce(MPI_IN_PLACE, &error, 1, MPI_INT, MPI_LOR, topology.comm());
  non_manifold = error;
#endif
  if (non_manifold)
  {
    throw std::runtime_error("Cannot sign the facets: an edge is shared by "
                             "more than two cells.");
  }

  return flags;
}
//-----------------------------------------------------------------------------
