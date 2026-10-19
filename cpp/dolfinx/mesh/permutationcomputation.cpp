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
#include <dolfinx/common/MPI.h>
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
#include <tuple>
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
    logger().info("Computing permutations for face type {}", t);
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
    logger().info("Compute edge permutations");
    const std::vector<std::bitset<bitset_size>> edge_perm
        = compute_edge_reflections<bitset_size>(topology, num_threads);
    for (std::int32_t c = 0; c < num_cells; ++c)
      for (int i = 0; i < entities_per_cell; ++i)
        perms[c * entities_per_cell + i] = edge_perm[c][i];
    break;
  }
  case 2:
  {
    logger().info("Compute face permutations");
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
    logger().info("Compute face permutations");
    const std::vector<std::bitset<bitset_size>> face_perm
        = compute_face_permutations<bitset_size>(topology, num_threads);
    for (std::int32_t c = 0; c < num_cells; ++c)
      cell_permutation_info[c] = face_perm[c].to_ulong();

    used_bits += cell_num_entities(cell_type, 2) * 3;
  }

  if (tdim > 1)
  {
    logger().info("Compute edge permutations");
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

/// Whether walking round cell `c` in its vertex order runs its local
/// edge `i` from the lower to the higher global vertex. The global vertex
/// numbering is shared by all cells and processes, so the result can be
/// compared between cells. `edge_perms` holds `edges_per_cell` edge
/// permutations per cell; bit 0 is whether edge `i` runs from the higher
/// to the lower global vertex.
bool runs_up(std::span<const std::uint8_t> edge_perms, int edges_per_cell,
             std::int32_t c, std::size_t i)
{
  assert(i < edge_runs_forward.size());
  const bool reflected = edge_perms[c * edges_per_cell + i] % 2;
  return edge_runs_forward[i] != reflected;
}

/// Values per edge reduced by reduce_edge_codes: the largest code, the
/// negated smallest code, so that both reduce with max, and the number
/// of cells
constexpr int codes_per_edge = 3;

/// The `bs` values of edge `e` in `data`, which holds `bs` per edge
template <int bs>
std::span<std::int64_t, bs> edge_row(std::span<std::int64_t> data,
                                     std::int32_t e)
{
  return data.subspan(bs * e).first<bs>();
}

/// Fold a `contribution` to an edge, i.e. the values of one owned cell
/// on the edge or another process's row for it, into the edge's row
/// `total`
void fold_codes(std::span<std::int64_t, codes_per_edge> total,
                std::span<const std::int64_t, codes_per_edge> contribution)
{
  total[0] = std::max(total[0], contribution[0]);
  total[1] = std::max(total[1], contribution[1]);
  total[2] += contribution[2];
}

/// For each owned and ghost edge, reduce over its owned cells on all
/// processes the codes `code(c, i)` of cell `c` on its local edge `i`
/// (see ::codes_per_edge). Each cell is owned once, so it is counted
/// once, and with two cells the largest and smallest code are the two
/// cells. No ghost cells are needed: the owner of an edge combines the
/// contributions, then sends the result to every process holding the
/// edge. Collective.
template <typename Code>
std::vector<std::int64_t>
reduce_edge_codes(const graph::AdjacencyList<std::int32_t>& c_to_e,
                  std::int32_t num_owned_cells,
                  const common::IndexMap& edge_map, Code code)
{
  const std::int32_t num_owned_edges = edge_map.size_local();
  const std::int32_t num_edges = num_owned_edges + edge_map.num_ghosts();
  constexpr std::int64_t none = std::numeric_limits<std::int64_t>::min();
  std::vector<std::int64_t> data(codes_per_edge * num_edges);
  for (std::int32_t e = 0; e < num_edges; ++e)
  {
    std::ranges::copy(std::array<std::int64_t, codes_per_edge>{none, none, 0},
                      edge_row<codes_per_edge>(data, e).begin());
  }
  for (std::int32_t c = 0; c < num_owned_cells; ++c)
  {
    std::span<const std::int32_t> edges = c_to_e.links(c);
    for (std::size_t i = 0; i < edges.size(); ++i)
    {
      const std::int64_t k = code(c, i);
      fold_codes(edge_row<codes_per_edge>(data, edges[i]),
                 std::array<std::int64_t, codes_per_edge>{k, -k, 1});
    }
  }

  const common::Scatterer<> sc(edge_map);
  const std::vector<std::int32_t>& owned = sc.local_indices_block();
  const std::vector<std::int32_t>& ghosts = sc.remote_indices_block();

  // Ghost contributions to the owner
  std::vector<std::int64_t> ghost_buffer(codes_per_edge * ghosts.size());
  for (std::size_t j = 0; j < ghosts.size(); ++j)
  {
    std::ranges::copy(
        edge_row<codes_per_edge>(data, num_owned_edges + ghosts[j]),
        std::next(ghost_buffer.begin(), codes_per_edge * j));
  }
  std::vector<std::int64_t> owned_buffer(codes_per_edge * owned.size());
  MPI_Request request = MPI_REQUEST_NULL;
  sc.scatter_rev_begin(ghost_buffer.data(), owned_buffer.data(), codes_per_edge,
                       request);
  sc.scatter_rev_end(request);
  for (std::size_t j = 0; j < owned.size(); ++j)
  {
    fold_codes(edge_row<codes_per_edge>(data, owned[j]),
               edge_row<codes_per_edge>(owned_buffer, j));
  }

  // Reduction back to the ghosts
  for (std::size_t j = 0; j < owned.size(); ++j)
  {
    std::ranges::copy(edge_row<codes_per_edge>(data, owned[j]),
                      std::next(owned_buffer.begin(), codes_per_edge * j));
  }
  sc.scatter_fwd_begin(owned_buffer.data(), ghost_buffer.data(), codes_per_edge,
                       request);
  sc.scatter_fwd_end(request);
  for (std::size_t j = 0; j < ghosts.size(); ++j)
  {
    std::ranges::copy(
        edge_row<codes_per_edge>(ghost_buffer, j),
        edge_row<codes_per_edge>(data, num_owned_edges + ghosts[j]).begin());
  }

  return data;
}

/// Gather `local` from all processes onto process 0, where `counts`
/// holds the number of values from each process (empty elsewhere).
/// Collective.
std::vector<std::int64_t> gather_on_root(MPI_Comm comm,
                                         std::span<const std::int64_t> local,
                                         std::span<const int> counts)
{
  std::vector<int> disp(counts.size() + 1, 0);
  std::partial_sum(counts.begin(), counts.end(), std::next(disp.begin()));
  std::vector<std::int64_t> all(disp.back());
  MPI_Gatherv(local.data(), local.size(), MPI_INT64_T, all.data(),
              counts.data(), disp.data(), MPI_INT64_T, 0, comm);
  return all;
}

/// Orient the parts joined by `links`, rows `(a, b, r)` meaning that
/// part `b` has `r` times the orientation of part `a`. Each set of
/// joined parts is oriented by its part with the lowest label, which
/// gets `1`. Returns the sorted labels, their orientations, and whether
/// the links are consistent; a contradicting link means the surface is
/// not orientable.
std::tuple<std::vector<std::int64_t>, std::vector<std::int8_t>, bool>
orient_parts(std::span<const std::int64_t> links)
{
  std::vector<std::int64_t> labels;
  labels.reserve(2 * links.size() / 3);
  for (std::size_t k = 0; k < links.size(); k += 3)
    labels.insert(labels.end(), {links[k], links[k + 1]});
  std::ranges::sort(labels);
  auto [last, end] = std::ranges::unique(labels);
  labels.erase(last, end);
  auto index = [&labels](std::int64_t l) -> std::int32_t
  {
    return std::ranges::distance(labels.begin(),
                                 std::ranges::lower_bound(labels, l));
  };

  // Union-find over the parts, numbered by label. rel[a] is the
  // orientation of a relative to parent[a]. A set's root is its lowest
  // label.
  std::vector<std::int32_t> parent(labels.size());
  std::iota(parent.begin(), parent.end(), 0);
  std::vector<std::int8_t> rel(labels.size(), 1);

  // Root of a and the orientation of a relative to it, compressing the
  // path
  auto find = [&parent, &rel](std::int32_t a) -> std::pair<std::int32_t, int>
  {
    std::int32_t root = a;
    int s = 1;
    while (parent[root] != root)
    {
      s *= rel[root];
      root = parent[root];
    }
    for (int sa = s; a != root;)
    {
      const std::int32_t next = parent[a];
      const int r = rel[a];
      parent[a] = root;
      rel[a] = sa;
      sa *= r;
      a = next;
    }
    return {root, s};
  };

  bool consistent = true;
  for (std::size_t k = 0; k < links.size(); k += 3)
  {
    auto [ra, sa] = find(index(links[k]));
    auto [rb, sb] = find(index(links[k + 1]));
    const int r = links[k + 2];
    if (ra == rb)
      consistent = consistent and sb == r * sa;
    else
    {
      // The orientation of either root relative to the other
      const std::int8_t s = r * sa * sb;
      if (ra < rb)
      {
        parent[rb] = ra;
        rel[rb] = s;
      }
      else
      {
        parent[ra] = rb;
        rel[ra] = s;
      }
    }
  }

  std::vector<std::int8_t> orientation(labels.size());
  for (std::size_t a = 0; a < labels.size(); ++a)
    orientation[a] = find(a).second;
  return {std::move(labels), std::move(orientation), consistent};
}
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
  const std::vector<std::uint8_t>& edge_perms
      = topology.get_entity_permutations(1);
  const int edges_per_cell = mesh::cell_num_entities(topology.cell_type(), 1);
  assert(std::cmp_less_equal(edges_per_cell, edge_runs_forward.size()));

  // Code of cell c and its local edge i: 2 * (global index of c) +
  // runs_up(c, i). It packs the cell and its direction along the edge
  // into one integer that is the same on every process. Codes order as
  // global cell indices do, so the larger code is the cell with the
  // larger global index, and code % 2 is its direction.
  std::vector<std::int32_t> local(num_cells);
  std::iota(local.begin(), local.end(), 0);
  std::vector<std::int64_t> global(num_cells);
  cell_map->local_to_global(local, global);
  auto code
      = [&global, &edge_perms, edges_per_cell](std::int32_t c, std::size_t i)
  { return 2 * global[c] + runs_up(edge_perms, edges_per_cell, c, i); };
  std::vector<std::int64_t> edge_data
      = reduce_edge_codes(*c_to_e, num_owned, *edge_map, code);

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
      std::span<const std::int64_t, codes_per_edge> d
          = edge_row<codes_per_edge>(edge_data, edges[i]);
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
  std::shared_ptr<const graph::AdjacencyList<std::int32_t>> c_to_e
      = topology.connectivity(2, 1);
  std::shared_ptr<const graph::AdjacencyList<std::int32_t>> e_to_c
      = topology.connectivity(1, 2);
  if (!c_to_e or !e_to_c)
  {
    throw std::runtime_error("The cell-to-edge and edge-to-cell "
                             "connectivities must be created first.");
  }

  MPI_Comm comm = topology.comm();
  std::shared_ptr<const common::IndexMap> cell_map = topology.index_map(2);
  std::shared_ptr<const common::IndexMap> edge_map = topology.index_map(1);
  const std::int32_t num_owned = cell_map->size_local();
  const std::int32_t num_cells = num_owned + cell_map->num_ghosts();
  const std::vector<std::uint8_t>& edge_perms
      = topology.get_entity_permutations(1);
  const int edges_per_cell = mesh::cell_num_entities(topology.cell_type(), 1);
  assert(std::cmp_less_equal(edges_per_cell, edge_runs_forward.size()));

  // +1 if cell c runs its local edge i up, -1 otherwise. Two cells
  // sharing an edge agree when they run it in opposite directions.
  auto direction = [&edge_perms, edges_per_cell](std::int32_t c,
                                                 std::size_t i) -> std::int8_t
  { return runs_up(edge_perms, edges_per_cell, c, i) ? 1 : -1; };

  // 1. Depth-first walk of each "part", the owned cells connected
  //    through edges whose two cells are both owned, from its "root",
  //    its first cell. sign[c] is the orientation of c relative to the
  //    root. Edges with more than two cells are not crossed. Conflicting
  //    routes are not checked here, but in step 2.
  std::vector<std::int8_t> sign(num_owned, 0);
  std::vector<std::int32_t> part(num_owned, -1);
  std::vector<std::int64_t> label; // Global index of each part's root
  {
    const std::int64_t offset = cell_map->local_range()[0];
    std::vector<std::int32_t> stack;
    for (std::int32_t root = 0; root < num_owned; ++root)
    {
      if (part[root] >= 0)
        continue;
      const std::int32_t p = label.size();
      label.push_back(offset + root);
      part[root] = p;
      sign[root] = 1;
      stack.push_back(root);
      while (!stack.empty())
      {
        const std::int32_t c = stack.back();
        stack.pop_back();
        std::span<const std::int32_t> edges = c_to_e->links(c);
        for (std::size_t i = 0; i < edges.size(); ++i)
        {
          // One cell: a boundary edge, or the other cell is on another
          // process (joined in step 2). More than two: not a manifold,
          // raised in step 2.
          std::span<const std::int32_t> cells = e_to_c->links(edges[i]);
          if (cells.size() != 2)
            continue;
          const std::int32_t d = cells[0] == c ? cells[1] : cells[0];
          if (d >= num_owned or part[d] >= 0)
            continue;
          std::span<const std::int32_t> d_edges = c_to_e->links(d);
          const std::size_t j = std::ranges::distance(
              d_edges.begin(), std::ranges::find(d_edges, edges[i]));
          part[d] = p;
          sign[d] = direction(c, i) == direction(d, j) ? -sign[c] : sign[c];
          stack.push_back(d);
        }
      }
    }
  }

  // 2. Find where parts meet. t = sign[c] * direction(c, i) is the
  //    direction c runs its edge i, oriented relative to its root, so
  //    consistently oriented cells give opposite t. The code 2 * label +
  //    (t > 0) gives both cells of an edge, their parts and their t.
  auto code = [&label, &part, &sign, &direction](std::int32_t c,
                                                 std::size_t i) -> std::int64_t
  { return 2 * label[part[c]] + (sign[c] * direction(c, i) > 0); };
  std::vector<std::int64_t> edge_data
      = reduce_edge_codes(*c_to_e, num_owned, *edge_map, code);

  // The owner of each edge checks it, and records a "link" (a, b, r)
  // where its two cells are in parts a < b: part b has r times the
  // orientation of part a, -1 if the cells run the edge the same way.
  constexpr int not_manifold = 1;
  constexpr int not_orientable = 2;
  int status = 0;
  std::vector<std::int64_t> links;
  for (std::int32_t e = 0; e < edge_map->size_local(); ++e)
  {
    std::span<const std::int64_t, codes_per_edge> d
        = edge_row<codes_per_edge>(edge_data, e);
    const std::int64_t hi = d[0];
    const std::int64_t lo = -d[1];
    if (d[2] > 2)
      status |= not_manifold;
    else if (d[2] == 2 and hi / 2 != lo / 2)
      links.insert(links.end(), {lo / 2, hi / 2, hi % 2 == lo % 2 ? -1 : 1});
    else if (d[2] == 2 and hi == lo) // One part, same t
      status |= not_orientable;
  }

  // Parts of this process with a link, which need their orientation
  // relative to the others
  std::vector<std::int8_t> linked(label.size(), false);
  for (std::int32_t c = 0; c < num_owned; ++c)
  {
    for (std::int32_t e : c_to_e->links(c))
    {
      std::span<const std::int64_t, codes_per_edge> d
          = edge_row<codes_per_edge>(edge_data, e);
      if (d[2] == 2 and d[0] / 2 != -d[1] / 2)
        linked[part[c]] = true;
    }
  }
  std::vector<std::int64_t> linked_labels;
  for (std::size_t p = 0; p < label.size(); ++p)
    if (linked[p])
      linked_labels.push_back(label[p]);

  // 3. Orient the linked parts on process 0, which replies to each
  //    process with the combined status and the orientation of each of
  //    its linked parts. The links are the edges between parts on
  //    different processes, so they are few compared with the cells.
  const int rank = dolfinx::MPI::rank(comm);
  const int size = dolfinx::MPI::size(comm);
  std::array<int, 3> sizes_local = {status, static_cast<int>(links.size()),
                                    static_cast<int>(linked_labels.size())};
  std::vector<int> sizes(rank == 0 ? 3 * size : 0);
  MPI_Gather(sizes_local.data(), 3, MPI_INT, sizes.data(), 3, MPI_INT, 0, comm);
  std::vector<int> link_counts, part_counts;
  for (std::size_t r = 0; r < sizes.size(); r += 3)
  {
    status |= sizes[r];
    link_counts.push_back(sizes[r + 1]);
    part_counts.push_back(sizes[r + 2]);
  }
  std::vector<std::int64_t> all_links
      = gather_on_root(comm, links, link_counts);
  std::vector<std::int64_t> all_labels
      = gather_on_root(comm, linked_labels, part_counts);

  std::vector<std::int8_t> reply;
  std::vector<int> reply_counts(part_counts.size());
  std::vector<int> reply_disp(part_counts.size() + 1, 0);
  if (rank == 0)
  {
    auto [labels, orientation, consistent] = orient_parts(all_links);
    if (!consistent)
      status |= not_orientable;
    std::ranges::transform(part_counts, reply_counts.begin(),
                           [](int n) { return n + 1; });
    std::partial_sum(reply_counts.begin(), reply_counts.end(),
                     std::next(reply_disp.begin()));
    reply.resize(reply_disp.back());
    for (std::size_t r = 0, k = 0; r < part_counts.size(); ++r)
    {
      reply[reply_disp[r]] = status;
      for (int j = 1; j <= part_counts[r]; ++j, ++k)
      {
        // Every linked part is in a link
        auto it = std::ranges::lower_bound(labels, all_labels[k]);
        assert(it != labels.end() and *it == all_labels[k]);
        reply[reply_disp[r] + j]
            = orientation[std::ranges::distance(labels.begin(), it)];
      }
    }
  }
  std::vector<std::int8_t> answer(1 + linked_labels.size());
  MPI_Scatterv(reply.data(), reply_counts.data(), reply_disp.data(), MPI_INT8_T,
               answer.data(), answer.size(), MPI_INT8_T, 0, comm);
  if (answer[0] & not_manifold)
  {
    throw std::runtime_error("Cannot orient the cells: an edge is shared "
                             "by more than two cells.");
  }
  if (answer[0] & not_orientable)
  {
    throw std::runtime_error("Cannot orient the cells: the surface is "
                             "not orientable (e.g. a Möbius strip).");
  }
  std::vector<std::int8_t> part_sign(label.size(), 1);
  for (std::size_t p = 0, k = 1; p < label.size(); ++p)
    if (linked[p])
      part_sign[p] = answer[k++];

  // 4. Orientations of the owned cells, then of the ghost cells from
  //    their owners
  std::vector<std::int8_t> orientations(num_cells);
  for (std::int32_t c = 0; c < num_owned; ++c)
    orientations[c] = part_sign[part[c]] * sign[c];
  const common::Scatterer<> cell_sc(*cell_map);
  const std::vector<std::int32_t>& owned = cell_sc.local_indices_block();
  const std::vector<std::int32_t>& ghosts = cell_sc.remote_indices_block();
  std::vector<std::int8_t> send_buffer(owned.size());
  std::ranges::transform(owned, send_buffer.begin(),
                         [&orientations](std::int32_t c)
                         { return orientations[c]; });
  std::vector<std::int8_t> recv_buffer(ghosts.size());
  MPI_Request request = MPI_REQUEST_NULL;
  cell_sc.scatter_fwd_begin(send_buffer.data(), recv_buffer.data(), 1, request);
  cell_sc.scatter_fwd_end(request);
  for (std::size_t i = 0; i < ghosts.size(); ++i)
    orientations[num_owned + ghosts[i]] = recv_buffer[i];

  return orientations;
}
//-----------------------------------------------------------------------------
