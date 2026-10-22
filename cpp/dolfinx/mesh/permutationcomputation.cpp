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

  // Equivalent to a la::Vector<std::int64_t> over the edges with block
  // size codes_per_edge: a reverse scatter that folds each received row
  // with fold_codes, then a forward scatter. The Scatterer is used
  // directly so that the mesh module does not depend on la.
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
  if (!c_to_e)
  {
    throw std::runtime_error(
        "The cell-to-edge connectivity must be created first.");
  }

  MPI_Comm comm = topology.comm();
  std::shared_ptr<const common::IndexMap> cell_map = topology.index_map(2);
  std::shared_ptr<const common::IndexMap> edge_map = topology.index_map(1);
  const std::int32_t num_owned_cells = cell_map->size_local();
  const std::int32_t num_cells = num_owned_cells + cell_map->num_ghosts();
  const std::vector<std::uint8_t>& edge_perms
      = topology.get_entity_permutations(1);
  const int edges_per_cell = mesh::cell_num_entities(topology.cell_type(), 1);
  assert(std::cmp_less_equal(edges_per_cell, edge_runs_forward.size()));

  // +1 if cell c runs its local edge i up, -1 otherwise. Two cells
  // sharing an edge agree when they run it in opposite directions.
  auto direction = [&edge_perms, edges_per_cell](std::int32_t c,
                                                 std::size_t i) -> std::int8_t
  { return runs_up(edge_perms, edges_per_cell, c, i) ? 1 : -1; };

  // The owned cells of each edge, as "slots" c * edges_per_cell + i for
  // cell c and its local edge i. Built here, rather than requiring the
  // edge-to-cell connectivity, so that nothing is kept on the topology.
  graph::AdjacencyList<std::int32_t> edge_slots(0);
  {
    const std::int32_t num_edges
        = edge_map->size_local() + edge_map->num_ghosts();
    std::vector<std::int32_t> offsets(num_edges + 1, 0);
    for (std::int32_t c = 0; c < num_owned_cells; ++c)
      for (std::int32_t e : c_to_e->links(c))
        ++offsets[e + 1];
    std::partial_sum(offsets.begin(), offsets.end(), offsets.begin());
    std::vector<std::int32_t> slots(offsets.back());
    std::vector<std::int32_t> pos(offsets.begin(), std::prev(offsets.end()));
    for (std::int32_t c = 0; c < num_owned_cells; ++c)
    {
      std::span<const std::int32_t> edges = c_to_e->links(c);
      for (std::size_t i = 0; i < edges.size(); ++i)
        slots[pos[edges[i]]++] = c * edges_per_cell + i;
    }
    edge_slots = graph::AdjacencyList<std::int32_t>(std::move(slots),
                                                    std::move(offsets));
  }

  // 1. Depth-first walk of each "part", the owned cells connected
  //    through edges with exactly two owned cells, from its "root", its
  //    first cell. sign[c] is the orientation of c relative to the root.
  //    Conflicting routes are not checked here, but in step 2.
  std::vector<std::int8_t> sign(num_owned_cells, 0);
  std::vector<std::int32_t> part(num_owned_cells, -1);
  std::int32_t num_parts = 0;
  {
    std::vector<std::int32_t> stack;
    for (std::int32_t root = 0; root < num_owned_cells; ++root)
    {
      if (part[root] >= 0)
        continue;
      const std::int32_t p = num_parts++;
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
          // One owned cell: a boundary edge, or the other cell is
          // owned by another process (joined in step 2). More than two:
          // not a manifold, raised in step 2.
          std::span<const std::int32_t> slots = edge_slots.links(edges[i]);
          if (slots.size() != 2)
            continue;
          const std::int32_t other
              = slots[0] / edges_per_cell == c ? slots[1] : slots[0];
          const std::int32_t d = other / edges_per_cell;
          const std::size_t j = other % edges_per_cell;
          if (part[d] >= 0)
            continue;
          part[d] = p;
          sign[d] = direction(c, i) == direction(d, j) ? -sign[c] : sign[c];
          stack.push_back(d);
        }
      }
    }
  }

  // Number the parts globally, process by process. Parts are numbered
  // in the order of their roots, and processes own increasing ranges of
  // cells, so the lowest part of a set is the one whose root has the
  // lowest global index.
  const common::IndexMap part_map(comm, num_parts);
  const std::int64_t part_offset = part_map.local_range()[0];
  const std::int64_t num_global_parts = part_map.size_global();

  // 2. Find where parts meet. t = sign[c] * direction(c, i) is the
  //    direction cell c runs its edge i once oriented relative to the
  //    root of its part, so consistently oriented cells give opposite t.
  //    Each owned cell contributes one code per edge (see
  //    encode_edge_orientation), and reduce_edge_codes keeps the largest
  //    and the smallest code of each edge. On an edge with two cells
  //    these are the codes of the two cells.
  const int rank = dolfinx::MPI::rank(comm);
  const int size = dolfinx::MPI::size(comm);
  // The code of cell c on its local edge i: its part (by global number),
  // whether its t is positive, and its owning process, packed into
  // (2 * part + (t > 0)) * size + rank
  auto encode_edge_orientation
      = [part_offset, rank, size, &part, &sign,
         &direction](std::int32_t c, std::size_t i) -> std::int64_t
  {
    const std::int64_t global_part = part_offset + part[c];
    const bool t_positive = sign[c] * direction(c, i) > 0;
    return (2 * global_part + t_positive) * size + rank;
  };
  // The part, whether t is positive, and the owning process in a code of
  // encode_edge_orientation:
  //   part = code / size / 2,
  //   t > 0 = (code / size) % 2,
  //   rank = code % size
  auto decode_edge_orientation
      = [size](std::int64_t code) -> std::tuple<std::int64_t, bool, int>
  { return {code / size / 2, (code / size) % 2, code % size}; };
  std::vector<std::int64_t> edge_data = reduce_edge_codes(
      *c_to_e, num_owned_cells, *edge_map, encode_edge_orientation);

  // Each owned cell reads, on each of its edges, the largest code, the
  // smallest code and the number of cells, counted over all processes:
  // - more than two cells: not a manifold;
  // - one cell: a boundary edge, joining nothing;
  // - two cells: one code is this cell's, the other the other cell's.
  //   - Same part: equal t means the part cannot be oriented.
  //   - Different parts: the other cell's part is a neighbour of this
  //     cell's part, with r times its orientation, where r = -1 if the
  //     two cells have equal t and 1 otherwise.
  constexpr int not_manifold = 1;
  constexpr int not_orientable = 2;
  int status = 0;
  // (local part index, global number of the neighbouring part, process
  //  owning the neighbouring part, r). Only r can differ between two
  //  meetings of the same pair of parts. Two owned cells sharing an edge
  //  are in the same part, so there is at most one meeting per edge with
  //  one owned cell, an edge on the boundary of the mesh or of this
  //  process.
  std::vector<std::array<std::int64_t, 4>> meetings;
  {
    // sum(offsets[1:] - offsets[:-1] == 1)
    const std::vector<std::int32_t>& offsets = edge_slots.offsets();
    const std::int32_t edges_with_one_owned_cell = std::transform_reduce(
        std::next(offsets.begin()), offsets.end(), offsets.begin(), 0,
        std::plus{},
        [](std::int32_t next, std::int32_t prev) { return next - prev == 1; });
    meetings.reserve(edges_with_one_owned_cell);
  }
  for (std::int32_t c = 0; c < num_owned_cells; ++c)
  {
    std::span<const std::int32_t> edges = c_to_e->links(c);
    for (std::size_t i = 0; i < edges.size(); ++i)
    {
      std::span<const std::int64_t, codes_per_edge> d
          = edge_row<codes_per_edge>(edge_data, edges[i]);
      const std::int64_t cells_on_edge = d[2];
      if (cells_on_edge > 2)
      {
        status |= not_manifold;
        continue;
      }
      if (cells_on_edge < 2)
        continue;

      const std::int64_t own_code = encode_edge_orientation(c, i);
      const std::int64_t other_code = own_code == d[0] ? -d[1] : d[0];
      const auto [own_part, own_t_positive, own_rank]
          = decode_edge_orientation(own_code);
      const auto [other_part, other_t_positive, other_rank]
          = decode_edge_orientation(other_code);
      const bool same_t = own_t_positive == other_t_positive;
      if (other_part != own_part)
      {
        meetings.push_back({part[c], other_part, other_rank, same_t ? -1 : 1});
      }
      else if (same_t)
        status |= not_orientable;
    }
  }

  // The neighbours of each part, by global part number, with the
  // relative orientation of each in part_signs and its owner in
  // neighbour_owners. A neighbour met with both orientations means the
  // surface cannot be oriented.
  std::ranges::sort(meetings);
  auto [last, end] = std::ranges::unique(meetings);
  meetings.erase(last, end);
  std::vector<std::int32_t> offsets(num_parts + 1, 0);
  std::vector<std::int64_t> neighbours;
  std::vector<std::int8_t> part_signs;
  std::vector<int> neighbour_owners;
  neighbours.reserve(meetings.size());
  part_signs.reserve(meetings.size());
  neighbour_owners.reserve(meetings.size());
  for (std::size_t k = 0; k < meetings.size(); ++k)
  {
    const auto [local_part, neighbour_part, owner, r] = meetings[k];
    // The same pair of parts twice in the list means both a 1 and a -1
    // orientation, so the surface is not orientable. The rest of the
    // list is then not needed: every process raises the error after the
    // first round of step 3, which all processes still enter.
    if (k > 0 and meetings[k - 1][0] == local_part
        and meetings[k - 1][1] == neighbour_part)
    {
      status |= not_orientable;
      break;
    }
    ++offsets[local_part + 1];
    neighbours.push_back(neighbour_part);
    part_signs.push_back(r);
    neighbour_owners.push_back(owner);
  }
  std::partial_sum(offsets.begin(), offsets.end(), offsets.begin());
  const graph::AdjacencyList<std::int64_t> part_neighbours(
      std::move(neighbours), std::move(offsets));

  // 3. Join the parts across processes, over an index map of the parts
  //    with each process's neighbouring parts as ghosts. Each part holds
  //    a row (representative, orientation): the lowest part it is known
  //    to be connected to, initially itself, and its orientation
  //    relative to that part. Each round, a forward scatter brings the
  //    neighbours' rows into the ghost rows, and every part takes the
  //    lowest representative among its neighbours', with the orientation
  //    that implies. Once no representative changes, every part's
  //    representative is the lowest part of its set, which the set is
  //    oriented by, and the rows received in that round are final, so
  //    each link is checked: a neighbour with the other orientation
  //    means the surface cannot be oriented.
  std::span<const std::int64_t> neighbour_parts = part_neighbours.array();
  const std::vector<std::int32_t>& neighbour_offsets
      = part_neighbours.offsets();
  std::vector<std::pair<std::int64_t, int>> ghost_owner_pairs;
  for (std::size_t k = 0; k < neighbour_parts.size(); ++k)
    ghost_owner_pairs.emplace_back(neighbour_parts[k], neighbour_owners[k]);
  std::ranges::sort(ghost_owner_pairs);
  auto [last_pair, end_pair] = std::ranges::unique(ghost_owner_pairs);
  ghost_owner_pairs.erase(last_pair, end_pair);
  std::vector<std::int64_t> ghost_parts;
  std::vector<int> ghost_owners;
  for (const auto& [ghost, owner] : ghost_owner_pairs)
  {
    ghost_parts.push_back(ghost);
    ghost_owners.push_back(owner);
  }
  const common::IndexMap neighbour_map(comm, num_parts, ghost_parts,
                                       ghost_owners);
  std::vector<std::int32_t> neighbour_rows(neighbour_parts.size());
  neighbour_map.global_to_local(neighbour_parts, neighbour_rows);

  // Rows of the owned parts, then of the ghosts
  constexpr int row_size = 2;
  std::vector<std::int64_t> rows(row_size * (num_parts + ghost_parts.size()),
                                 1);
  for (std::int32_t p = 0; p < num_parts; ++p)
    rows[row_size * p] = part_offset + p; // Orientation 1 relative to itself

  // Each round's exchange is equivalent to the forward scatter of a
  // la::Vector<std::int64_t> over neighbour_map with block size
  // row_size. The Scatterer is used directly so that the mesh module
  // does not depend on la.
  const common::Scatterer<> sc(neighbour_map);
  const std::vector<std::int32_t>& owned_rows = sc.local_indices_block();
  const std::vector<std::int32_t>& ghost_rows = sc.remote_indices_block();
  std::vector<std::int64_t> send(row_size * owned_rows.size());
  std::vector<std::int64_t> recv(row_size * ghost_rows.size());

  constexpr int changed = 4;
  constexpr int conflict = 8;

  // Each round moves the lowest representative one part further along
  // every chain of parts, so the representatives stop changing within
  // num_global_parts rounds, and one more round sees no change
  bool converged = false;
  for (std::int64_t round = 0; round <= num_global_parts and !converged;
       ++round)
  {
    for (std::size_t j = 0; j < owned_rows.size(); ++j)
    {
      send[row_size * j] = rows[row_size * owned_rows[j]];
      send[row_size * j + 1] = rows[row_size * owned_rows[j] + 1];
    }
    MPI_Request request = MPI_REQUEST_NULL;
    sc.scatter_fwd_begin(send.data(), recv.data(), row_size, request);
    sc.scatter_fwd_end(request);
    for (std::size_t j = 0; j < ghost_rows.size(); ++j)
    {
      const std::int32_t ghost = num_parts + ghost_rows[j];
      rows[row_size * ghost] = recv[row_size * j];
      rows[row_size * ghost + 1] = recv[row_size * j + 1];
    }

    int round_status = status;
    // The rows after this round, updated from the previous round's rows
    // so that every part sees the same values
    std::vector<std::int64_t> updated_rows = rows;
    for (std::int32_t p = 0; p < num_parts; ++p)
    {
      const std::int64_t representative = rows[row_size * p];
      const std::int64_t orientation = rows[row_size * p + 1];
      std::int64_t& updated_representative = updated_rows[row_size * p];
      std::int64_t& updated_orientation = updated_rows[row_size * p + 1];

      // p adopts the lowest of its neighbours' representatives, if lower
      // than its own, which is how it joins the set of the lowest part
      // it is connected to. A neighbour has r times the orientation of p,
      // so p has r times the neighbour's.
      for (std::int32_t k = neighbour_offsets[p]; k < neighbour_offsets[p + 1];
           ++k)
      {
        const std::int64_t neighbour_representative
            = rows[row_size * neighbour_rows[k]];
        const std::int64_t neighbour_orientation
            = rows[row_size * neighbour_rows[k] + 1];
        const std::int64_t r = part_signs[k];
        if (neighbour_representative < updated_representative)
        {
          updated_representative = neighbour_representative;
          updated_orientation = r * neighbour_orientation;
        }
        if (neighbour_representative == representative
            and neighbour_orientation != r * orientation)
        {
          round_status |= conflict;
        }
      }
    }
    if (updated_rows != rows)
      round_status |= changed;
    rows = std::move(updated_rows);

    MPI_Allreduce(MPI_IN_PLACE, &round_status, 1, MPI_INT, MPI_BOR, comm);
    if (round_status & not_manifold)
    {
      throw std::runtime_error("Cannot orient the cells: an edge is shared "
                               "by more than two cells.");
    }
    // A conflict is final only in a round without changes
    converged = !(round_status & changed);
    if ((round_status & not_orientable)
        or (converged and (round_status & conflict)))
    {
      throw std::runtime_error("Cannot orient the cells: the surface is "
                               "not orientable (e.g. a Möbius strip).");
    }
  }
  assert(converged);

  // 4. Orientations of the owned cells, then of the ghost cells from
  //    their owners
  std::vector<std::int8_t> orientations(num_cells);
  for (std::int32_t c = 0; c < num_owned_cells; ++c)
    orientations[c] = rows[row_size * part[c] + 1] * sign[c];
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
    orientations[num_owned_cells + ghosts[i]] = recv_buffer[i];

  return orientations;
}
//-----------------------------------------------------------------------------
