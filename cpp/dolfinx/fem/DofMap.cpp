// Copyright (C) 2007-2018 Anders Logg and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "DofMap.h"
#include "DofMapBuilder.h"
#include "ElementDofLayout.h"
#include "utils.h"
#include <cstdint>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/MPI.h>
#include <dolfinx/common/types.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/mesh/Topology.h>

using namespace dolfinx;
using namespace dolfinx::fem;

namespace
{
template <typename T>
Eigen::Array<std::int32_t, Eigen::Dynamic, 1>
remap_dofs(const std::vector<std::int32_t>& old_to_new,
           const graph::AdjacencyList<T>& dofs_old)
{
  const Eigen::Array<T, Eigen::Dynamic, 1>& _dofs_old = dofs_old.array();
  Eigen::Array<std::int32_t, Eigen::Dynamic, 1> dofmap(_dofs_old.rows());
  for (Eigen::Index i = 0; i < dofmap.size(); ++i)
    dofmap[i] = old_to_new[_dofs_old[i]];
  return dofmap;
}
//-----------------------------------------------------------------------------
// Build a collapsed DofMap from a dofmap view
fem::DofMap build_collapsed_dofmap(MPI_Comm comm, const DofMap& dofmap_view,
                                   const mesh::Topology& topology)
{
  auto element_dof_layout = std::make_shared<ElementDofLayout>(
      dofmap_view.element_dof_layout->copy());
  assert(element_dof_layout);

  if (dofmap_view.index_map->block_size() == 1
      and element_dof_layout->block_size() > 1)
  {
    throw std::runtime_error(
        "Cannot collapse dofmap with block size greater "
        "than 1 from parent with block size of 1. Create new dofmap first.");
  }

  if (dofmap_view.index_map->block_size() > 1
      and element_dof_layout->block_size() > 1)
  {
    throw std::runtime_error(
        "Cannot (yet) collapse dofmap with block size greater "
        "than 1 from parent with block size greater than 1. Create new dofmap "
        "first.");
  }

  // Get topological dimension
  const int tdim = topology.dim();
  auto cells = topology.connectivity(tdim, 0);
  assert(cells);

  // Build set of dofs that are in the new dofmap
  std::vector<std::int32_t> dofs_view;
  for (int i = 0; i < cells->num_nodes(); ++i)
  {
    auto cell_dofs = dofmap_view.cell_dofs(i);
    for (Eigen::Index dof = 0; dof < cell_dofs.rows(); ++dof)
      dofs_view.push_back(cell_dofs[dof]);
  }
  std::sort(dofs_view.begin(), dofs_view.end());
  dofs_view.erase(std::unique(dofs_view.begin(), dofs_view.end()),
                  dofs_view.end());
  // Get block sizes
  const int bs_view = dofmap_view.index_map->block_size();

  // Compute sizes
  const std::int32_t num_owned_view = dofmap_view.index_map->size_local();
  const auto it_unowned0 = std::lower_bound(dofs_view.begin(), dofs_view.end(),
                                            num_owned_view * bs_view);
  const std::size_t num_owned = std::distance(dofs_view.begin(), it_unowned0);
  const std::size_t num_unowned = std::distance(it_unowned0, dofs_view.end());

  // Get process offset for new dofmap
  const std::int64_t process_offset
      = dolfinx::MPI::global_offset(comm, num_owned, true);

  // For owned dofs, compute new global index
  std::vector<std::int64_t> global_index(dofmap_view.index_map->size_local(),
                                         -1);
  for (auto it = dofs_view.begin(); it != it_unowned0; ++it)
  {
    const std::size_t block = std::distance(dofs_view.begin(), it);
    const std::int32_t block_parent = *it / bs_view;
    global_index[block_parent] = block + process_offset;
  }

  // Send new global indices for owned dofs to non-owning process, and
  // receive new global indices from owner
  std::vector global_index_remote
      = dofmap_view.index_map->scatter_fwd(global_index, 1);
  const Eigen::Array<std::int32_t, Eigen::Dynamic, 1> ghost_owner_old
      = dofmap_view.index_map->ghost_owner_rank();

  // Compute ghosts for collapsed dofmap
  Eigen::Array<std::int64_t, Eigen::Dynamic, 1> ghosts(num_unowned);
  std::vector<int> ghost_owners(num_unowned);
  for (auto it = it_unowned0; it != dofs_view.end(); ++it)
  {
    const std::int32_t index = std::distance(it_unowned0, it);
    const std::int32_t index_old = *it / bs_view - num_owned_view;
    assert(global_index_remote[index_old] >= 0);
    ghosts[index] = global_index_remote[index_old];
    ghost_owners[index] = ghost_owner_old[index_old];
  }

  // Create new index map
  auto index_map = std::make_shared<common::IndexMap>(
      comm, num_owned,
      dolfinx::MPI::compute_graph_edges(
          comm, std::set<int>(ghost_owners.begin(), ghost_owners.end())),
      ghosts, ghost_owners, 1);

  // Create array from dofs in view to new dof indices
  std::vector<std::int32_t> old_to_new(dofs_view.back() + 1, -1);
  std::int32_t count = 0;
  for (auto& dof : dofs_view)
    old_to_new[dof] = count++;

  // Build new dofmap
  const graph::AdjacencyList<std::int32_t>& dof_array_view = dofmap_view.list();
  Eigen::Array<std::int32_t, Eigen::Dynamic, 1> dofmap
      = remap_dofs(old_to_new, dof_array_view);

  // Dimension sanity checks
  assert(element_dof_layout);
  assert(dofmap.rows()
         == (cells->num_nodes() * element_dof_layout->num_dofs()));

  const int cell_dimension = element_dof_layout->num_dofs();
  assert(dofmap.rows() % cell_dimension == 0);
  Eigen::Map<Eigen::Array<std::int32_t, Eigen::Dynamic, Eigen::Dynamic,
                          Eigen::RowMajor>>
      _dofmap(dofmap.data(), dofmap.rows() / cell_dimension, cell_dimension);

  return fem::DofMap(element_dof_layout, index_map,
                     graph::AdjacencyList<std::int32_t>(_dofmap));
}

} // namespace

//-----------------------------------------------------------------------------
graph::AdjacencyList<std::int32_t>
fem::transpose_dofmap(graph::AdjacencyList<std::int32_t>& dofmap,
                      std::int32_t num_cells)
{
  // Count number of cell contributions to each global index
  const std::int32_t max_index
      = dofmap.array().head(dofmap.offsets()(num_cells)).maxCoeff();
  std::vector<int> num_local_contributions(max_index + 1, 0);
  for (int c = 0; c < num_cells; ++c)
  {
    auto dofs = dofmap.links(c);
    for (int i = 0; i < dofs.rows(); ++i)
      num_local_contributions[dofs[i]]++;
  }

  // Compute offset for each global index
  std::vector<int> index_offsets(num_local_contributions.size() + 1, 0);
  std::partial_sum(num_local_contributions.begin(),
                   num_local_contributions.end(), index_offsets.begin() + 1);

  std::vector<std::int32_t> data(index_offsets.back());
  std::vector<int> pos = index_offsets;
  int cell_offset = 0;
  for (int c = 0; c < num_cells; ++c)
  {
    auto dofs = dofmap.links(c);
    for (int i = 0; i < dofs.rows(); ++i)
      data[pos[dofs[i]]++] = cell_offset++;
  }

  // Sort the source indices for each global index
  // This could improve linear memory access
  // FIXME: needs profiling
  for (int index = 0; index < max_index; ++index)
  {
    std::sort(data.begin() + index_offsets[index],
              data.begin() + index_offsets[index + 1]);
  }

  return graph::AdjacencyList<std::int32_t>(data, index_offsets);
}
//-----------------------------------------------------------------------------
DofMap::DofMap(std::shared_ptr<const ElementDofLayout> element_dof_layout,
               std::shared_ptr<const common::IndexMap> index_map,
               const graph::AdjacencyList<std::int32_t>& dofmap)
    : element_dof_layout(element_dof_layout), index_map(index_map),
      _dofmap(dofmap)
{
  // Dofmap data is copied as the types for dofmap and _dofmap may
  // differ, typically 32- vs 64-bit integers
}
//-----------------------------------------------------------------------------
DofMap DofMap::extract_sub_dofmap(const std::vector<int>& component) const
{
  assert(!component.empty());

  // Set element dof layout and cell dimension
  assert(element_dof_layout);
  std::shared_ptr<const ElementDofLayout> sub_element_dof_layout
      = this->element_dof_layout->sub_dofmap(component);

  // Get components in parent map that correspond to sub-dofs
  const std::vector sub_element_map_view
      = this->element_dof_layout->sub_view(component);

  // Build dofmap by extracting from parent
  const int num_cells = this->_dofmap.num_nodes();
  const std::int32_t dofs_per_cell = sub_element_map_view.size();
  Eigen::Array<std::int32_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      dofmap(num_cells, dofs_per_cell);
  for (int c = 0; c < num_cells; ++c)
  {
    auto cell_dmap_parent = this->_dofmap.links(c);
    for (std::int32_t i = 0; i < dofs_per_cell; ++i)
      dofmap(c, i) = cell_dmap_parent[sub_element_map_view[i]];
  }

  return DofMap(sub_element_dof_layout, this->index_map,
                graph::AdjacencyList<std::int32_t>(dofmap));
}
//-----------------------------------------------------------------------------
std::pair<std::unique_ptr<DofMap>, std::vector<std::int32_t>>
DofMap::collapse(MPI_Comm comm, const mesh::Topology& topology) const
{
  assert(element_dof_layout);
  assert(index_map);

  // Create new element dof layout and reset parent

  // Parent does not have block structure but sub-map does, so build
  // new submap to get block structure for collapsed dofmap.
  // Create new dofmap
  std::unique_ptr<DofMap> dofmap_new;
  if (this->index_map->block_size() == 1
      and this->element_dof_layout->block_size() > 1)
  {
    // Create new element dof layout and reset parent
    auto collapsed_dof_layout
        = std::make_shared<ElementDofLayout>(element_dof_layout->copy());

    // Parent does not have block structure but sub-map does, so build
    // new submap to get block structure for collapsed dofmap.
    auto [index_map, dofmap]
        = DofMapBuilder::build(comm, topology, *collapsed_dof_layout);
    dofmap_new = std::make_unique<DofMap>(element_dof_layout, index_map,
                                          std::move(dofmap));
  }
  else
  {
    // Collapse dof map, without build and re-ordering from scratch
    dofmap_new = std::make_unique<DofMap>(
        build_collapsed_dofmap(comm, *this, topology));
  }
  assert(dofmap_new);

  // Build map from collapsed dof index to original dof index
  auto index_map_new = dofmap_new->index_map;
  const std::int32_t size
      = (index_map_new->size_local() + index_map_new->num_ghosts())
        * index_map_new->block_size();
  std::vector<std::int32_t> collapsed_map(size);

  const int tdim = topology.dim();
  auto cells = topology.connectivity(tdim, 0);
  assert(cells);
  for (int c = 0; c < cells->num_nodes(); ++c)
  {
    auto cell_dofs_view = this->cell_dofs(c);
    auto cell_dofs = dofmap_new->cell_dofs(c);
    assert(cell_dofs_view.rows() == cell_dofs.rows());
    for (Eigen::Index i = 0; i < cell_dofs_view.rows(); ++i)
    {
      assert(cell_dofs[i] < (int)collapsed_map.size());
      collapsed_map[cell_dofs[i]] = cell_dofs_view[i];
    }
  }

  return {std::move(dofmap_new), std::move(collapsed_map)};
}
//-----------------------------------------------------------------------------
