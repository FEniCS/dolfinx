// Copyright (C) 2013-2022 Chris Richardson
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "utils.h"
#include <algorithm>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/MPI.h>
#include <dolfinx/common/sort.h>
#include <dolfinx/fem/ElementDofLayout.h>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/MeshTags.h>
#include <dolfinx/mesh/Topology.h>
#include <dolfinx/mesh/graphbuild.h>
#include <dolfinx/mesh/topologycomputation.h>
#include <dolfinx/mesh/utils.h>
#include <map>
#include <memory>
#include <mpi.h>
#include <vector>

using namespace dolfinx;

//-----------------------------------------------------------------------------
std::int64_t refinement::impl::local_to_global(std::int32_t local_index,
                                               const common::IndexMap& map)
{
  assert(local_index >= 0);
  const std::array local_range = map.local_range();
  const std::int32_t local_size = local_range[1] - local_range[0];
  if (local_index < local_size)
  {
    const std::int64_t global_offset = local_range[0];
    return global_offset + local_index;
  }
  else
  {
    std::span ghosts = map.ghosts();
    assert((local_index - local_size) < (int)ghosts.size());
    return ghosts[local_index - local_size];
  }
}
//---------------------------------------------------------------------------------
void refinement::update_logical_edgefunction(
    MPI_Comm comm,
    const std::vector<std::vector<std::int32_t>>& marked_for_update,
    std::span<std::int8_t> marked_edges, const common::IndexMap& map)
{
  std::vector<int> send_sizes;
  std::vector<std::int64_t> data_to_send;
  for (std::size_t i = 0; i < marked_for_update.size(); ++i)
  {
    for (std::int32_t q : marked_for_update[i])
      data_to_send.push_back(impl::local_to_global(q, map));

    send_sizes.push_back(marked_for_update[i].size());
  }

  // Send all shared edges marked for update and receive from other
  // processes
  std::vector<std::int64_t> data_to_recv;
  {
    int indegree(-1), outdegree(-2), weighted(-1);
    MPI_Dist_graph_neighbors_count(comm, &indegree, &outdegree, &weighted);
    assert(indegree == (int)marked_for_update.size());
    assert(indegree == outdegree);

    std::vector<int> recv_sizes(outdegree);
    send_sizes.reserve(1);
    recv_sizes.reserve(1);
    MPI_Neighbor_alltoall(send_sizes.data(), 1, MPI_INT, recv_sizes.data(), 1,
                          MPI_INT, comm);

    // Build displacements
    std::vector<int> send_disp = {0};
    std::partial_sum(send_sizes.begin(), send_sizes.end(),
                     std::back_inserter(send_disp));
    std::vector<int> recv_disp = {0};
    std::partial_sum(recv_sizes.begin(), recv_sizes.end(),
                     std::back_inserter(recv_disp));

    data_to_recv.resize(recv_disp.back());
    MPI_Neighbor_alltoallv(data_to_send.data(), send_sizes.data(),
                           send_disp.data(), MPI_INT64_T, data_to_recv.data(),
                           recv_sizes.data(), recv_disp.data(), MPI_INT64_T,
                           comm);
  }

  // Flatten received values and set marked_edges at each index received
  std::vector<std::int32_t> local_indices(data_to_recv.size());
  map.global_to_local(data_to_recv, local_indices);
  for (std::int32_t local_index : local_indices)
  {
    assert(local_index != -1);
    marked_edges[local_index] = true;
  }
}
//-----------------------------------------------------------------------------
std::vector<std::int64_t>
refinement::adjust_indices(const common::IndexMap& map, std::int32_t n)
{
  // NOTE: Is this effectively concatenating index maps?

  // Get offset for 'n' for this process
  const std::int64_t num_local = n;
  std::int64_t global_offset = 0;
  MPI_Exscan(&num_local, &global_offset, 1, MPI_INT64_T, MPI_SUM, map.comm());

  std::span owners = map.owners();
  std::span src = map.src();
  std::span dest = map.dest();

  MPI_Comm comm;
  MPI_Dist_graph_create_adjacent(map.comm(), src.size(), src.data(),
                                 MPI_UNWEIGHTED, dest.size(), dest.data(),
                                 MPI_UNWEIGHTED, MPI_INFO_NULL, false, &comm);

  // Communicate offset to neighbors
  std::vector<std::int64_t> offsets(src.size(), 0);
  offsets.reserve(1);
  MPI_Neighbor_allgather(&global_offset, 1, MPI_INT64_T, offsets.data(), 1,
                         MPI_INT64_T, comm);

  MPI_Comm_free(&comm);

  int local_size = map.size_local();
  std::vector<std::int64_t> global_indices = map.global_indices();

  // Add new offset to owned indices
  std::transform(global_indices.begin(),
                 std::next(global_indices.begin(), local_size),
                 global_indices.begin(),
                 [global_offset](auto x) { return x + global_offset; });

  // Add offsets to ghost indices
  std::transform(std::next(global_indices.begin(), local_size),
                 global_indices.end(), owners.begin(),
                 std::next(global_indices.begin(), local_size),
                 [&src, &offsets](auto idx, auto r)
                 {
                   auto it = std::ranges::lower_bound(src, r);
                   assert(it != src.end() and *it == r);
                   int rank = std::distance(src.begin(), it);
                   return idx + offsets[rank];
                 });

  return global_indices;
}
//-----------------------------------------------------------------------------
std::array<std::vector<std::int32_t>, 2> refinement::transfer_facet_meshtag(
    const mesh::MeshTags<std::int32_t>& tags0, const mesh::Topology& topology1,
    std::span<const std::int32_t> cell, std::span<const std::int8_t> facet)
{
  auto topology = tags0.topology();
  assert(topology);
  auto values = tags0.values();
  auto indices = tags0.indices();

  int tdim = topology->dim();
  if (topology->index_map(tdim)->num_ghosts() > 0)
    throw std::runtime_error("Ghosted meshes are not supported");

  auto c_to_f = topology->connectivity(tdim, tdim - 1);
  if (!c_to_f)
    throw std::runtime_error("Parent mesh is missing cell-facet connectivity.");

  // Create map parent->child facets
  const std::int32_t num_input_facets
      = topology->index_map(tdim - 1)->size_local()
        + topology->index_map(tdim - 1)->num_ghosts();

  // Get global index for each refined cell, before reordering in Mesh
  // construction
  const std::vector<std::int64_t>& original_cell_index
      = topology1.original_cell_index[0];
  assert(original_cell_index.size() == cell.size());
  std::int64_t global_offset = topology1.index_map(tdim)->local_range()[0];

  // Map cells back to original index
  std::vector<std::int32_t> local_cell_index(original_cell_index.size());
  for (std::size_t i = 0; i < local_cell_index.size(); ++i)
  {
    assert(original_cell_index[i] >= global_offset);
    assert(original_cell_index[i] - global_offset
           < (int)local_cell_index.size());
    local_cell_index[original_cell_index[i] - global_offset] = i;
  }

  // Count number of child facets for each parent facet
  std::vector<int> count_child(num_input_facets, 0);
  for (std::size_t c = 0; c < cell.size(); ++c)
  {
    auto facets = c_to_f->links(cell[c]);
    for (int j = 0; j <= tdim; ++j)
    {
      if (std::int8_t fidx = facet[c * (tdim + 1) + j]; fidx != -1)
        ++count_child[facets[fidx]];
    }
  }

  auto c_to_f_refined = topology1.connectivity(tdim, tdim - 1);
  if (!c_to_f_refined)
  {
    throw std::runtime_error(
        "Refined mesh is missing cell-facet connectivity.");
  }

  // Fill in data for each child facet
  std::vector<int> offset_child(num_input_facets + 1, 0);
  std::partial_sum(count_child.begin(), count_child.end(),
                   std::next(offset_child.begin()));
  std::vector<std::int32_t> child_facet(offset_child.back());
  for (std::size_t c = 0; c < cell.size(); ++c)
  {
    auto facets = c_to_f->links(cell[c]);

    // Use original indexing for child cell
    auto refined_facets = c_to_f_refined->links(local_cell_index[c]);

    // Get child facets for each cell
    for (int j = 0; j <= tdim; ++j)
    {
      if (std::int8_t fidx = facet[c * (tdim + 1) + j]; fidx != -1)
      {
        int offset = offset_child[facets[fidx]];
        child_facet[offset] = refined_facets[j];
        ++offset_child[facets[fidx]];
      }
    }
  }

  // Rebuild offset
  offset_child.front() = 0;
  std::partial_sum(count_child.begin(), count_child.end(),
                   std::next(offset_child.begin()));
  graph::AdjacencyList<std::int32_t> p_to_c_facet(std::move(child_facet),
                                                  std::move(offset_child));

  // Copy facet meshtag from parent to child
  std::vector<std::int32_t> facet_indices, tag_values;
  for (std::size_t i = 0; i < indices.size(); ++i)
  {
    std::int32_t parent_index = indices[i];
    auto pclinks = p_to_c_facet.links(parent_index);

    // Eliminate duplicates
    std::ranges::sort(pclinks);
    auto it_end = std::ranges::unique(pclinks).begin();
    facet_indices.insert(facet_indices.end(), pclinks.begin(), it_end);
    tag_values.insert(tag_values.end(), std::distance(pclinks.begin(), it_end),
                      values[i]);
  }

  // Sort values into order, based on facet indices
  std::vector<std::int32_t> sort_order(tag_values.size());
  std::iota(sort_order.begin(), sort_order.end(), 0);
  dolfinx::radix_sort(sort_order, [&facet_indices](auto index)
                      { return facet_indices[index]; });
  std::vector<std::int32_t> sorted_facet_indices(facet_indices.size());
  std::vector<std::int32_t> sorted_tag_values(tag_values.size());
  for (std::size_t i = 0; i < sort_order.size(); ++i)
  {
    sorted_tag_values[i] = tag_values[sort_order[i]];
    sorted_facet_indices[i] = facet_indices[sort_order[i]];
  }

  return {std::move(sorted_facet_indices), std::move(sorted_tag_values)};
}
//----------------------------------------------------------------------------
std::array<std::vector<std::int32_t>, 2>
refinement::transfer_cell_meshtag(const mesh::MeshTags<std::int32_t>& tags0,
                                  const mesh::Topology& topology1,
                                  std::span<const std::int32_t> cell)
{
  auto topology0 = tags0.topology();
  assert(topology0);
  auto values0 = tags0.values();
  auto indices0 = tags0.indices();

  const int tdim = topology0->dim();
  if (tags0.dim() != tdim)
    throw std::runtime_error("Input meshtag is not cell-based");

  if (topology0->index_map(tdim)->num_ghosts() > 0)
    throw std::runtime_error("Ghosted meshes are not supported");

  // Create map parent->child facets
  const std::int32_t num_input_cells
      = topology0->index_map(tdim)->size_local()
        + topology0->index_map(tdim)->num_ghosts();
  std::vector<int> count_child(num_input_cells, 0);

  // Get global index for each refined cell, before reordering in Mesh
  // construction
  const std::vector<std::int64_t>& original_cell_index
      = topology1.original_cell_index[0];
  assert(original_cell_index.size() == cell.size());
  std::int64_t global_offset = topology1.index_map(tdim)->local_range()[0];

  // Map back to original index
  std::vector<std::int32_t> local_cell_index(original_cell_index.size());
  for (std::size_t i = 0; i < local_cell_index.size(); ++i)
  {
    assert(original_cell_index[i] >= global_offset);
    assert(original_cell_index[i] - global_offset
           < (int)local_cell_index.size());
    local_cell_index[original_cell_index[i] - global_offset] = i;
  }

  // Count number of child cells for each parent cell
  for (std::int32_t c : cell)
    ++count_child[c];

  std::vector<int> offset_child(num_input_cells + 1, 0);
  std::partial_sum(count_child.begin(), count_child.end(),
                   std::next(offset_child.begin()));
  std::vector<std::int32_t> child_cell(offset_child.back());

  // Fill in data for each child cell
  for (std::size_t c = 0; c < cell.size(); ++c)
  {
    std::int32_t pc = cell[c];
    int offset = offset_child[pc];

    // Use original indexing for child cell
    const std::int32_t lc = local_cell_index[c];
    child_cell[offset] = lc;
    ++offset_child[pc];
  }

  // Rebuild offset
  offset_child.front() = 0;
  std::partial_sum(count_child.begin(), count_child.end(),
                   std::next(offset_child.begin()));
  graph::AdjacencyList p_to_c_cell(std::move(child_cell),
                                   std::move(offset_child));

  // Copy cell meshtag from parent to child
  std::vector<std::int32_t> cell_indices, tag_values;
  // std::span<const std::int32_t> in_index = meshtag.indices();
  // std::span<const std::int32_t> in_value = meshtag.values();
  for (std::size_t i = 0; i < indices0.size(); ++i)
  {
    auto pclinks = p_to_c_cell.links(indices0[i]);
    cell_indices.insert(cell_indices.end(), pclinks.begin(), pclinks.end());
    tag_values.insert(tag_values.end(), pclinks.size(), values0[i]);
  }

  // Sort values into order, based on cell indices
  std::vector<std::int32_t> sort_order(tag_values.size());
  std::iota(sort_order.begin(), sort_order.end(), 0);
  dolfinx::radix_sort(sort_order, [&cell_indices](auto index)
                      { return cell_indices[index]; });
  std::vector<std::int32_t> sorted_tag_values(tag_values.size());
  std::vector<std::int32_t> sorted_cell_indices(cell_indices.size());
  for (std::size_t i = 0; i < sort_order.size(); ++i)
  {
    sorted_tag_values[i] = tag_values[sort_order[i]];
    sorted_cell_indices[i] = cell_indices[sort_order[i]];
  }

  return {std::move(sorted_cell_indices), std::move(sorted_tag_values)};
}
//-----------------------------------------------------------------------------
