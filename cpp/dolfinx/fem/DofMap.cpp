// Copyright (C) 2007-2018 Anders Logg and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "DofMap.h"
#include "ElementDofLayout.h"
#include "dofmapbuilder.h"
#include "utils.h"
#include <cstdint>
#include <dolfinx/common/IndexMapNew.h>
#include <dolfinx/common/MPI.h>
#include <dolfinx/common/sort.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/mesh/Topology.h>
#include <memory>
#include <utility>

using namespace dolfinx;
using namespace dolfinx::fem;

namespace
{
//-----------------------------------------------------------------------------
template <typename T>
std::vector<std::int32_t>
remap_dofs(const std::vector<std::int32_t>& old_to_new,
           const graph::AdjacencyList<T>& dofs_old)
{
  const std::vector<T>& _dofs_old = dofs_old.array();
  std::vector<std::int32_t> dofmap(_dofs_old.size());
  for (std::size_t i = 0; i < dofmap.size(); ++i)
    dofmap[i] = old_to_new[_dofs_old[i]];
  return dofmap;
}
//-----------------------------------------------------------------------------
// Build a collapsed DofMap from a dofmap view. Extracts dofs and
// doesn't build a new re-ordered dofmap
fem::DofMap build_collapsed_dofmap(MPI_Comm /*comm*/, const DofMap& dofmap_view,
                                   const mesh::Topology& topology)
{
  if (dofmap_view.element_dof_layout().block_size() > 1)
  {
    throw std::runtime_error(
        "Cannot collapse a dofmap with block size greater "
        "than 1 from parent with block size of 1. Create new dofmap first.");
  }

  // Get topological dimension
  const int tdim = topology.dim();
  auto cells = topology.connectivity(tdim, 0);
  assert(cells);

  // Build set of dofs that are in the new dofmap (un-blocked)
  std::vector<std::int32_t> dofs_view = dofmap_view.list().array();
  dolfinx::radix_sort(xtl::span(dofs_view));
  dofs_view.erase(std::unique(dofs_view.begin(), dofs_view.end()),
                  dofs_view.end());

  // Get block size
  const int bs_view = dofmap_view.index_map_bs();

  // Compute sizes
  const std::int32_t num_owned_view = dofmap_view.index_map->size_local();
  const auto it_unowned0 = std::lower_bound(dofs_view.begin(), dofs_view.end(),
                                            num_owned_view * bs_view);

  // Create sub-index map
  std::shared_ptr<common::IndexMapNew> index_map;
  {
    std::vector<std::int32_t> indices;
    indices.reserve(std::distance(dofs_view.begin(), it_unowned0));
    std::transform(dofs_view.begin(), it_unowned0, std::back_inserter(indices),
                   [bs = bs_view](auto idx) { return idx / bs; });
    indices.erase(std::unique(indices.begin(), indices.end()), indices.end());

    auto [_index_map, _] = dofmap_view.index_map->create_submap(indices);
    index_map = std::make_shared<common::IndexMapNew>(std::move(_index_map));
  }

  // Create array from dofs in view to new dof indices
  std::vector<std::int32_t> old_to_new(dofs_view.back() + 1, -1);
  std::int32_t count = 0;
  for (auto& dof : dofs_view)
    old_to_new[dof] = count++;

  // Build new dofmap
  const graph::AdjacencyList<std::int32_t>& dof_array_view = dofmap_view.list();
  std::vector<std::int32_t> dofmap = remap_dofs(old_to_new, dof_array_view);

  // Dimension sanity checks
  assert((int)dofmap.size()
         == (cells->num_nodes() * dofmap_view.element_dof_layout().num_dofs()));

  const int cell_dimension = dofmap_view.element_dof_layout().num_dofs();
  assert(dofmap.size() % cell_dimension == 0);

  // Copy dof layout, discarding parent data
  ElementDofLayout element_dof_layout = dofmap_view.element_dof_layout().copy();

  // Create new dofmap and return
  return fem::DofMap(
      std::move(element_dof_layout), index_map, 1,
      graph::regular_adjacency_list(std::move(dofmap), cell_dimension), 1);
}

} // namespace

//-----------------------------------------------------------------------------
graph::AdjacencyList<std::int32_t>
fem::transpose_dofmap(const graph::AdjacencyList<std::int32_t>& dofmap,
                      std::int32_t num_cells)
{
  // Count number of cell contributions to each global index
  const std::int32_t max_index = *std::max_element(
      dofmap.array().begin(),
      std::next(dofmap.array().begin(), dofmap.offsets()[num_cells]));

  std::vector<int> num_local_contributions(max_index + 1, 0);
  for (int c = 0; c < num_cells; ++c)
  {
    for (auto dof : dofmap.links(c))
      num_local_contributions[dof]++;
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
    for (auto dof : dofmap.links(c))
      data[pos[dof]++] = cell_offset++;
  }

  // Sort the source indices for each global index
  // This could improve linear memory access
  // FIXME: needs profiling
  for (int index = 0; index < max_index; ++index)
  {
    std::sort(data.begin() + index_offsets[index],
              data.begin() + index_offsets[index + 1]);
  }

  return graph::AdjacencyList<std::int32_t>(std::move(data),
                                            std::move(index_offsets));
}
//-----------------------------------------------------------------------------
/// Equality operator
bool DofMap::operator==(const DofMap& map) const
{
  return this->_index_map_bs == map._index_map_bs
         and this->_dofmap == map._dofmap and this->_bs == map._bs;
}
//-----------------------------------------------------------------------------
int DofMap::bs() const noexcept { return _bs; }
//-----------------------------------------------------------------------------
DofMap DofMap::extract_sub_dofmap(const std::vector<int>& component) const
{
  assert(!component.empty());

  // Get components in parent map that correspond to sub-dofs
  const std::vector sub_element_map_view
      = this->element_dof_layout().sub_view(component);

  // Build dofmap by extracting from parent
  const int num_cells = this->_dofmap.num_nodes();
  // FIXME X: how does sub_element_map_view hand block sizes?
  const std::int32_t dofs_per_cell = sub_element_map_view.size();
  std::vector<std::int32_t> dofmap(num_cells * dofs_per_cell);
  const int bs_parent = this->bs();
  for (int c = 0; c < num_cells; ++c)
  {
    auto cell_dmap_parent = this->_dofmap.links(c);
    for (std::int32_t i = 0; i < dofs_per_cell; ++i)
    {
      const std::div_t pos = std::div(sub_element_map_view[i], bs_parent);
      dofmap[c * dofs_per_cell + i]
          = bs_parent * cell_dmap_parent[pos.quot] + pos.rem;
    }
  }

  // FIXME X

  // Set element dof layout and cell dimension
  ElementDofLayout sub_element_dof_layout
      = _element_dof_layout.sub_layout(component);
  return DofMap(
      std::move(sub_element_dof_layout), this->index_map, this->index_map_bs(),
      graph::regular_adjacency_list(std::move(dofmap), dofs_per_cell), 1);
}
//-----------------------------------------------------------------------------
std::pair<DofMap, std::vector<std::int32_t>> DofMap::collapse(
    MPI_Comm comm, const mesh::Topology& topology,
    const std::function<std::vector<int>(
        const graph::AdjacencyList<std::int32_t>&)>& reorder_fn) const
{
  // Create new dofmap
  auto create_subdofmap = [](MPI_Comm comm, auto index_map_bs, auto& layout,
                             auto& topology, auto& reorder_fn, auto& dmap)
  {
    if (index_map_bs == 1 and layout.block_size() > 1)
    {
      // Parent does not have block structure but sub-map does, so build
      // new submap to get block structure for collapsed dofmap.

      // Create new element dof layout and reset parent
      ElementDofLayout collapsed_dof_layout = layout.copy();

      auto [_index_map, bs, dofmap] = fem::build_dofmap_data(
          comm, topology, collapsed_dof_layout, reorder_fn);
      auto index_map
          = std::make_shared<common::IndexMapNew>(std::move(_index_map));
      return DofMap(layout, index_map, bs, std::move(dofmap), bs);
    }
    else
    {
      // Collapse dof map, without build and re-ordering from scratch
      return build_collapsed_dofmap(comm, dmap, topology);
    }
  };

  DofMap dofmap_new = create_subdofmap(
      comm, index_map_bs(), _element_dof_layout, topology, reorder_fn, *this);

  // Build map from collapsed dof index to original dof index
  auto index_map_new = dofmap_new.index_map;
  const std::int32_t size
      = (index_map_new->size_local() + index_map_new->num_ghosts())
        * dofmap_new.index_map_bs();
  std::vector<std::int32_t> collapsed_map(size);

  const int tdim = topology.dim();
  auto cells = topology.connectivity(tdim, 0);
  assert(cells);
  const int bs = dofmap_new.bs();
  for (int c = 0; c < cells->num_nodes(); ++c)
  {
    xtl::span<const std::int32_t> cell_dofs_view = this->cell_dofs(c);
    xtl::span<const std::int32_t> cell_dofs = dofmap_new.cell_dofs(c);
    for (std::size_t i = 0; i < cell_dofs.size(); ++i)
    {
      for (int k = 0; k < bs; ++k)
      {
        assert(bs * cell_dofs[i] + k < (int)collapsed_map.size());
        assert(bs * i + k < cell_dofs_view.size());
        collapsed_map[bs * cell_dofs[i] + k] = cell_dofs_view[bs * i + k];
      }
    }
  }

  return {std::move(dofmap_new), std::move(collapsed_map)};
}
//-----------------------------------------------------------------------------
const graph::AdjacencyList<std::int32_t>& DofMap::list() const
{
  return _dofmap;
}
//-----------------------------------------------------------------------------
int DofMap::index_map_bs() const { return _index_map_bs; }
//-----------------------------------------------------------------------------
