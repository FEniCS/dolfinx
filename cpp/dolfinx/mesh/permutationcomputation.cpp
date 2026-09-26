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
  auto c_to_e = topology.connectivity(2, 1);
  auto e_to_c = topology.connectivity(1, 2);
  if (!c_to_e or !e_to_c)
  {
    throw std::runtime_error("Edges and the connectivity between edges and "
                             "cells must be created first.");
  }

  MPI_Comm comm = topology.comm();
  std::shared_ptr<const common::IndexMap> cell_map = topology.index_map(2);
  std::shared_ptr<const common::IndexMap> edge_map = topology.index_map(1);
  const std::int32_t num_owned = cell_map->size_local();
  const std::int32_t num_cells = num_owned + cell_map->num_ghosts();

  // For process-local cell c and one of its edges e, whether walking
  // round c in its vertex order (anticlockwise on the reference cell)
  // runs e from its lower to its higher global vertex. The global vertex
  // numbering is shared by all cells and ranks, so values from different
  // cells can be compared. forward[i] is whether the walk runs Basix edge
  // i = [p, q] from p to q, and the reflection bit whether p to q runs
  // from the higher to the lower global vertex. Round a triangle (0-1-2)
  // the walk runs its edges [1, 2], [0, 2], [0, 1] forward, back,
  // forward, and round a quadrilateral (0-1-3-2) its edges [0, 1],
  // [0, 2], [1, 3], [2, 3] forward, back, forward, back.
  const std::vector<std::uint8_t>& edge_perms
      = topology.get_entity_permutations(1);
  const int edges_per_cell = mesh::cell_num_entities(topology.cell_type(), 1);
  auto runs_up = [&c_to_e, &edge_perms, &edges_per_cell](std::int32_t c,
                                                         std::int32_t e) -> bool
  {
    constexpr std::array<bool, 4> forward = {true, false, true, false};
    std::span<const std::int32_t> edges = c_to_e->links(c);
    const std::size_t i
        = std::ranges::distance(edges.begin(), std::ranges::find(edges, e));
    assert(i < edges.size());
    const bool reflected = edge_perms[c * edges_per_cell + i] % 2;
    return forward[i] != reflected;
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

  // 2. Find the cell on the other side of each edge that the walk could
  //    not cross because that cell is owned by another rank. Step 3 joins
  //    the parts across these pairs.
  //
  //    A "half" is an owned cell together with one of its edges that
  //    other ranks hold, as owner or ghost. Only these "sharing ranks"
  //    can own the other cells of the edge. Every half is sent to all
  //    sharing ranks of its edge, so each rank sees, with its own halves,
  //    all cells of its shared edges and pairs its halves itself, without
  //    a reply. The sharing ranks of all edges form one symmetric
  //    neighbourhood, whose communicator step 3 uses again.
  auto [ranks, edge_rank_data, edge_rank_offsets]
      = common::compute_sharing_neighbourhood(*edge_map);
  const graph::AdjacencyList<int> edge_ranks(std::move(edge_rank_data),
                                             std::move(edge_rank_offsets));
  const std::size_t num_neighbours = ranks.size();
  MPI_Comm graph;
  int ierr = MPI_Dist_graph_create_adjacent(
      comm, ranks.size(), ranks.data(), MPI_UNWEIGHTED, ranks.size(),
      ranks.data(), MPI_UNWEIGHTED, MPI_INFO_NULL, false, &graph);
  dolfinx::MPI::check_error(comm, ierr);

  // The halves: owned cell, whether the cell runs the edge up, and the
  // local edge
  std::vector<std::int32_t> half_cell;
  std::vector<std::int8_t> half_up;
  std::vector<std::int32_t> half_edge;
  for (std::int32_t c = 0; c < num_owned; ++c)
  {
    for (std::int32_t e : c_to_e->links(c))
    {
      if (edge_ranks.num_links(e) > 0)
      {
        half_cell.push_back(c);
        half_up.push_back(runs_up(c, e));
        half_edge.push_back(e);
      }
    }
  }
  const std::size_t num_halves = half_cell.size();

  // The other side of each half: the neighbourhood rank that owns it, or
  // -1 if there is none, its half index there, and whether it runs the
  // edge up. Step 3 sends to that rank, addressed to that half, and
  // compares the directions to tell whether the two cells agree.
  std::vector<std::int32_t> partner(num_halves, -1);
  std::vector<std::int64_t> partner_half(num_halves, -1);
  std::vector<std::int8_t> partner_up(num_halves, 0);
  {
    // Send (global edge, direction, half index) to every rank sharing the
    // edge. The receiver finds the edge by its global index and keeps the
    // half index to address the half in step 3.
    std::vector<std::int64_t> global_edge(num_halves);
    edge_map->local_to_global(half_edge, global_edge);
    std::vector<std::vector<std::int64_t>> send_to(num_neighbours);
    for (std::size_t h = 0; h < num_halves; ++h)
    {
      for (int r : edge_ranks.links(half_edge[h]))
      {
        send_to[r].insert(send_to[r].end(), {global_edge[h], half_up[h],
                                             static_cast<std::int64_t>(h)});
      }
    }
    std::vector<std::int64_t> send;
    std::vector<std::int32_t> rows_per_dest;
    for (const std::vector<std::int64_t>& rows : send_to)
    {
      send.insert(send.end(), rows.begin(), rows.end());
      rows_per_dest.push_back(rows.size() / 3);
    }
    auto [recv, rows_per_src] = exchange_rows(graph, std::move(send),
                                              rows_per_dest, num_neighbours, 3);

    // Count on each edge the owned cells, i.e. own halves, and the halves
    // received from other ranks, and keep the sender and row of a
    // received half. A rank that holds an edge only through ghost cells
    // receives its halves but has none of its own, and ignores them.
    const std::int32_t num_edges
        = edge_map->size_local() + edge_map->num_ghosts();
    std::vector<std::int32_t> num_own(num_edges, 0);
    for (std::int32_t e : half_edge)
      ++num_own[e];
    std::vector<std::int32_t> num_remote(num_edges, 0);
    std::vector<std::int32_t> remote_src(num_edges, -1);
    std::vector<std::int32_t> remote_row(num_edges, -1);
    {
      const std::size_t num_rows = recv.size() / 3;
      std::vector<std::int64_t> recv_global(num_rows);
      for (std::size_t r = 0; r < num_rows; ++r)
        recv_global[r] = recv[3 * r];
      std::vector<std::int32_t> recv_edge(num_rows);
      edge_map->global_to_local(recv_global, recv_edge);
      for (std::size_t s = 0, r = 0; s < num_neighbours; ++s)
      {
        for (std::int32_t i = 0; i < rows_per_src[s]; ++i, ++r)
        {
          const std::int32_t e = recv_edge[r];
          assert(e >= 0);
          ++num_remote[e];
          remote_src[e] = s;
          remote_row[e] = r;
        }
      }
    }

    // Classify each half by the number of cells of its edge. More than two
    // means that the surface is not a manifold there, e.g. a T-joint whose
    // cells are split over ranks, so that no rank holds all of them. With
    // exactly one received half, and so one own, the two are partners.
    // Otherwise there is no partner: two own halves were joined by the
    // walk, and a single one lies on the boundary of the surface.
    for (std::size_t h = 0; h < num_halves; ++h)
    {
      const std::int32_t e = half_edge[h];
      if (num_own[e] + num_remote[e] > 2)
        problems |= not_manifold;
      else if (num_remote[e] == 1)
      {
        const std::int32_t r = remote_row[e];
        partner[h] = remote_src[e];
        partner_half[h] = recv[3 * r + 2];
        partner_up[h] = recv[3 * r + 1];
      }
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
      if (part[c] == p)
        label[p++] = offset + c;
  }
  std::vector<std::int8_t> part_sign(num_parts, 1);

  // Halves with a partner, grouped by the partner's neighbourhood rank
  std::vector<std::int32_t> paired;
  std::vector<std::int32_t> rows_per_neighbour(num_neighbours, 0);
  for (std::size_t h = 0; h < num_halves; ++h)
  {
    if (partner[h] >= 0)
    {
      paired.push_back(h);
      ++rows_per_neighbour[partner[h]];
    }
  }
  std::ranges::stable_sort(paired, {},
                           [&partner](std::int32_t h) { return partner[h]; });

  // Send (partner half, label, orientation) for every paired half and
  // return what the partners sent
  auto exchange_state
      = [&paired, &half_cell, &partner_half, &label, &part, &part_sign, &sign,
         &graph, &rows_per_neighbour, &num_neighbours]()
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
                         num_neighbours, 3)
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
