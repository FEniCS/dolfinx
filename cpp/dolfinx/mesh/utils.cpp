// Copyright (C) 2006-2026 Anders Logg and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "utils.h"
#include "Geometry.h"
#include "Mesh.h"
#include "Topology.h"
#include "cell_types.h"
#include "graphbuild.h"
#include <algorithm>
#include <cstdlib>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/log.h>
#include <dolfinx/common/math.h>
#include <dolfinx/common/sort.h>
#include <dolfinx/fem/ElementDofLayout.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/graph/partition.h>
#include <format>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

using namespace dolfinx;

//-----------------------------------------------------------------------------
std::vector<std::int64_t> mesh::impl::reorder_cells(
    const graph::Reorder& reorder_fn, std::span<const double> cell_centroids,
    int gdim, std::optional<std::int32_t> max_facet_to_cell_links,
    const std::vector<CellType>& celltypes,
    const std::vector<fem::ElementDofLayout>& doflayouts,
    const std::vector<std::vector<int>>& ghost_owners,
    std::vector<std::vector<std::int64_t>>& cells,
    std::vector<std::span<std::int64_t>>& cells_v,
    std::vector<std::vector<std::int64_t>>& original_idx, int num_threads)
{
  // Build local dual graph for owned cells to (i) get list of vertices
  // on the process boundary and (ii) apply re-ordering to cells for
  // locality

  spdlog::info("Build local dual graphs, re-order cells, and compute process "
               "boundary vertices.");

  // Unmatched facets (process boundary candidates) per cell type: the
  // flattened, row-major vertex data and the row width (number of
  // vertices per facet), which can differ between cell types.
  std::vector<std::pair<std::vector<std::int64_t>, int>> facets;

  // Build lists of cells (by cell type) that excludes ghosts
  std::vector<std::span<const std::int64_t>> cells1_v_local;
  std::size_t cell_offset = 0;
  for (std::size_t i = 0; i < celltypes.size(); ++i)
  {
    int num_cell_vertices = mesh::num_cell_vertices(celltypes[i]);
    std::size_t num_owned_cells
        = cells_v[i].size() / num_cell_vertices - ghost_owners[i].size();
    cells1_v_local.emplace_back(cells_v[i].data(),
                                num_owned_cells * num_cell_vertices);

    // Build local dual graph for cell type
    auto [graph, unmatched_facets, max_v, _facet_attached_cells]
        = build_local_dual_graph(std::vector{celltypes[i]},
                                 std::vector{cells1_v_local.back()},
                                 max_facet_to_cell_links, num_threads);

    // Store unmatched_facets for current cell type
    facets.emplace_back(std::move(unmatched_facets), max_v);

    // Compute graph reordering.
    const std::vector<std::int32_t> remap = std::visit(
        [cell_centroids, gdim, cell_offset, num_owned_cells,
         &graph](const auto& fn) -> std::vector<std::int32_t>
        {
          using F = std::decay_t<decltype(fn)>;
          if constexpr (std::is_same_v<F, graph::reorder_graph_fn>)
            return fn ? fn(graph) : graph::reorder_rcm(graph);
          else
          {
            if (!fn)
              throw std::invalid_argument(
                  "Geometric cell reordering function is empty.");
            return fn(
                cell_centroids.subspan(cell_offset, gdim * num_owned_cells),
                gdim);
          }
        },
        reorder_fn);

    cell_offset += gdim * num_owned_cells;

    // Update 'original' indices
    const std::vector<std::int64_t>& orig_idx = original_idx[i];
    std::vector<std::int64_t> _original_idx(orig_idx.size());
    std::copy_n(orig_idx.rbegin(), ghost_owners[i].size(),
                _original_idx.rbegin());
    {
      for (std::size_t j = 0; j < remap.size(); ++j)
        _original_idx[remap[j]] = orig_idx[j];
    }
    original_idx[i] = _original_idx;

    // Reorder cells. `cells_v[i]` aliases `cells[i]` for 'P1
    // geometry', where the two reorderings are the same operation on
    // the same buffer.
    impl::reorder_list(
        std::span(cells_v[i].data(), remap.size() * num_cell_vertices), remap);
    if (cells_v[i].data() != cells[i].data())
    {
      impl::reorder_list(
          std::span(cells[i].data(), remap.size() * doflayouts[i].num_dofs()),
          remap);
    }
  }

  // Build list of boundary vertices from unmatched facets across all
  // cell
  if (facets.size() == 1) // Optimisation for single cell type
  {
    std::vector<std::int64_t>& vertices = facets.front().first;

    // Remove duplicated vertex indices
    dolfinx::radix_sort(vertices);
    auto [unique_end, range_end] = std::ranges::unique(vertices);
    vertices.erase(unique_end, range_end);

    // Remove -1 if it appears as first entity. This can happen in
    // mixed topology meshes where '-1' is used to pad facet data when
    // cells facets have differing numbers of vertices.
    if (!vertices.empty() and vertices.front() == -1)
      vertices.erase(vertices.begin());

    return vertices;
  }
  else
  {
    // Pack 'unmatched' facets for all cell types into a single
    // column-major array (facets0): column j holds vertex j across
    // all facets, so the multi-column sort_by_perm() overload can
    // operate directly on contiguous per-column data.
    std::size_t num_facets = std::accumulate(
        facets.begin(), facets.end(), std::size_t(0), [](std::size_t x, auto& y)
        { return x + (y.second > 0 ? y.first.size() / y.second : 0); });
    int max_v = std::ranges::max_element(facets, [](auto& a, auto& b)
                                         { return a.second < b.second; })
                    ->second;

    std::vector<std::int64_t> facets0_b(max_v * num_facets, -1);
    std::vector<std::span<std::int64_t>> facets0(max_v);
    for (int j = 0; j < max_v; ++j)
      facets0[j] = std::span(facets0_b.data() + j * num_facets, num_facets);

    {
      std::size_t row = 0;
      for (const auto& [v_data, num_v] : facets)
      {
        for (auto it = v_data.begin(); it != v_data.end(); it += num_v, ++row)
          for (int j = 0; j < num_v; ++j)
            facets0[j][row] = *std::next(it, j);
      }
    }

    // Compute row permutation
    std::vector<std::span<const std::int64_t>> facets0_view(facets0.begin(),
                                                            facets0.end());
    const std::vector<std::int32_t> perm = dolfinx::sort_by_perm(
        std::span<std::span<const std::int64_t>>(facets0_view));

    // For facets in facets0 that appear only once, store the facet
    // vertices
    std::vector<std::int64_t> vertices;

    // Number of leading valid (non -1 padding) vertices in row
    auto trim_len = [&facets0, max_v](std::int32_t row)
    {
      int n = max_v;
      while (n > 0 and facets0[n - 1][row] < 0)
        --n;
      return n;
    };

    auto it = perm.begin();
    while (it != perm.end())
    {
      std::int32_t row0 = *it;
      int n = trim_len(row0);

      // Find iterator to next facet whose leading n vertices differ
      // from row0
      auto it1 = std::find_if_not(it, perm.end(),
                                  [&facets0, row0, n](std::int32_t row)
                                  {
                                    for (int j = 0; j < n; ++j)
                                      if (facets0[j][row] != facets0[j][row0])
                                        return false;
                                    return true;
                                  });

      // If no repeated facet found, insert row0 vertices
      if (std::ranges::distance(it, it1) == 1)
      {
        for (int j = 0; j < n; ++j)
          vertices.push_back(facets0[j][row0]);
      }
      else if (std::ranges::distance(it, it1) > 2)
        throw std::runtime_error("More than two matching facets found.");

      // Advance iterator
      it = it1;
    }

    // Remove duplicate indices
    dolfinx::radix_sort(vertices);
    auto [unique_end, range_end] = std::ranges::unique(vertices);
    vertices.erase(unique_end, range_end);

    return vertices;
  }
}
//-----------------------------------------------------------------------------
std::vector<std::int64_t>
mesh::extract_topology(CellType cell_type, const fem::ElementDofLayout& layout,
                       std::span<const std::int64_t> cells)
{
  // Use ElementDofLayout to get vertex dof indices (local to a cell)
  const int num_vertices_per_cell = num_cell_vertices(cell_type);
  const int num_node_per_cell = layout.num_dofs();
  std::vector<int> local_vertices(num_vertices_per_cell);
  for (int i = 0; i < num_vertices_per_cell; ++i)
  {
    const std::vector<int>& local_index = layout.entity_dofs(0, i);
    assert(local_index.size() == 1);
    local_vertices[i] = local_index[0];
  }

  // Extract vertices
  std::vector<std::int64_t> topology((cells.size() / num_node_per_cell)
                                     * num_vertices_per_cell);
  for (std::size_t c = 0; c < cells.size() / num_node_per_cell; ++c)
  {
    auto p = cells.subspan(c * num_node_per_cell, num_node_per_cell);
    std::span t(topology.data() + c * num_vertices_per_cell,
                num_vertices_per_cell);
    for (int j = 0; j < num_vertices_per_cell; ++j)
      t[j] = p[local_vertices[j]];
  }

  return topology;
}
//-----------------------------------------------------------------------------
bool mesh::is_vertex_dof_layout(CellType cell_type,
                                const fem::ElementDofLayout& layout)
{
  const int num_vertices_per_cell = num_cell_vertices(cell_type);
  if (layout.num_dofs() != num_vertices_per_cell)
    return false;

  for (int i = 0; i < num_vertices_per_cell; ++i)
  {
    const std::vector<int>& local_index = layout.entity_dofs(0, i);
    if (local_index.size() != 1 or local_index.front() != i)
      return false;
  }

  return true;
}
//-----------------------------------------------------------------------------
std::vector<std::int32_t> mesh::exterior_facet_indices(const Topology& topology,
                                                       int facet_type_idx)
{
  const int tdim = topology.dim();
  auto f_to_c = topology.connectivity(tdim - 1, tdim);
  if (!f_to_c)
  {
    throw std::runtime_error(
        "Facet to cell connectivity has not been computed.");
  }

  // Find all owned facets (not ghost) with only one attached cell
  auto facet_map = topology.index_maps(tdim - 1).at(facet_type_idx);

  std::vector<std::int32_t> facets;
  for (std::int32_t f = 0; f < facet_map->size_local(); ++f)
  {
    if (f_to_c->num_links(f) == 1)
      facets.push_back(f);
  }

  // Remove facets on internal inter-process boundary
  std::vector<std::int32_t> ext_facets;
  std::ranges::set_difference(facets,
                              topology.interprocess_facets(facet_type_idx),
                              std::back_inserter(ext_facets));

  return ext_facets;
}
//------------------------------------------------------------------------------
std::vector<std::int32_t> mesh::exterior_facet_indices(const Topology& topology)
{
  if (topology.entity_types(topology.dim() - 1).size() > 1)
  {
    throw std::runtime_error("Multiple facet types in mesh. Call "
                             "exterior_facet_indices with facet type index.");
  }

  return mesh::exterior_facet_indices(topology, 0);
}
//------------------------------------------------------------------------------
std::vector<std::int32_t>
mesh::compute_incident_entities(const Topology& topology,
                                std::span<const std::int32_t> entities, int d0,
                                int d1)
{
  auto map0 = topology.index_map(d0);
  if (!map0)
  {
    throw std::runtime_error(std::format(
        "Mesh entities of dimension {} have not been created.", d0));
  }

  auto map1 = topology.index_map(d1);
  if (!map1)
  {
    throw std::runtime_error(std::format(
        "Mesh entities of dimension {} have not been created.", d1));
  }

  auto e0_to_e1 = topology.connectivity(d0, d1);
  if (!e0_to_e1)
  {
    throw std::runtime_error(
        std::format("Connectivity missing: ({}, {})", d0, d1));
  }

  std::vector<std::int32_t> entities1;
  for (std::int32_t entity : entities)
  {
    auto e = e0_to_e1->links(entity);
    entities1.insert(entities1.end(), e.begin(), e.end());
  }

  std::ranges::sort(entities1);
  auto [unique_end, range_end] = std::ranges::unique(entities1);
  entities1.erase(unique_end, range_end);

  return entities1;
}
//-----------------------------------------------------------------------------
