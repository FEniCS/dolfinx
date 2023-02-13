// Copyright (C) 2006-2023 Anders Logg and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "Geometry.h"
#include "Topology.h"
#include <common/Scatterer.h>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/sort.h>
#include <dolfinx/fem/ElementDofLayout.h>
#include <dolfinx/fem/dofmapbuilder.h>
#include <dolfinx/graph/partition.h>

using namespace dolfinx;
using namespace dolfinx::mesh;

//-----------------------------------------------------------------------------
int Geometry::dim() const { return _dim; }
//-----------------------------------------------------------------------------
const graph::AdjacencyList<std::int32_t>& Geometry::dofmap() const
{
  return _dofmap;
}
//-----------------------------------------------------------------------------
std::shared_ptr<const common::IndexMap> Geometry::index_map() const
{
  return _index_map;
}
//-----------------------------------------------------------------------------
std::span<double> Geometry::x() { return _x; }
//-----------------------------------------------------------------------------
std::span<const double> Geometry::x() const { return _x; }
//-----------------------------------------------------------------------------
const fem::CoordinateElement& Geometry::cmap() const { return _cmap; }
//-----------------------------------------------------------------------------
const std::vector<std::int64_t>& Geometry::input_global_indices() const
{
  return _input_global_indices;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
mesh::Geometry mesh::create_geometry(
    MPI_Comm comm, const Topology& topology,
    const fem::CoordinateElement& element,
    const graph::AdjacencyList<std::int64_t>& cell_nodes,
    std::span<const double> x, int dim,
    const std::function<std::vector<int>(
        const graph::AdjacencyList<std::int32_t>&)>& reorder_fn)
{
  // TODO: make sure required entities are initialised, or extend
  // fem::build_dofmap_data

  //  Build 'geometry' dofmap on the topology
  auto [_dof_index_map, bs, dofmap] = fem::build_dofmap_data(
      comm, topology, element.create_dof_layout(), reorder_fn);
  auto dof_index_map
      = std::make_shared<common::IndexMap>(std::move(_dof_index_map));

  // If the mesh has higher order geometry, permute the dofmap
  if (element.needs_dof_permutations())
  {
    const int D = topology.dim();
    const int num_cells = topology.connectivity(D, 0)->num_nodes();
    const std::vector<std::uint32_t>& cell_info
        = topology.get_cell_permutation_info();

    for (std::int32_t cell = 0; cell < num_cells; ++cell)
      element.unpermute_dofs(dofmap.links(cell), cell_info[cell]);
  }

  auto remap_data
      = [](auto comm, auto& cell_nodes, auto& x, int dim, auto& dofmap)
  {
    // Build list of unique (global) node indices from adjacency list
    // (geometry nodes)
    std::vector<std::int64_t> indices = cell_nodes.array();
    dolfinx::radix_sort(std::span(indices));
    indices.erase(std::unique(indices.begin(), indices.end()), indices.end());

    //  Distribute  node coordinates by global index from other ranks.
    //  Order of coords matches order of the indices in 'indices'.
    std::vector<double> coords
        = MPI::distribute_data<double>(comm, indices, x, dim);

    // Compute local-to-global map from local indices in dofmap to the
    // corresponding global indices in cell_nodes
    std::vector l2g
        = graph::build::compute_local_to_global_links(cell_nodes, dofmap);

    // Compute local (dof) to local (position in coords) map from (i)
    // local-to-global for dofs and (ii) local-to-global for entries in
    // coords
    std::vector l2l = graph::build::compute_local_to_local(l2g, indices);

    // Allocate space for input global indices and copy data
    std::vector<std::int64_t> igi(indices.size());
    std::transform(l2l.cbegin(), l2l.cend(), igi.begin(),
                   [&indices](auto index) { return indices[index]; });

    return std::tuple(std::move(coords), std::move(l2l), std::move(igi));
  };

  auto [coords, l2l, igi] = remap_data(comm, cell_nodes, x, dim, dofmap);

  // Build coordinate dof array, copying coordinates to correct
  // position
  assert(coords.size() % dim == 0);
  const std::size_t shape0 = coords.size() / dim;
  const std::size_t shape1 = dim;
  std::vector<double> xg(3 * shape0, 0);
  for (std::size_t i = 0; i < shape0; ++i)
  {
    std::copy_n(std::next(coords.cbegin(), shape1 * l2l[i]), shape1,
                std::next(xg.begin(), 3 * i));
  }

  return Geometry(dof_index_map, std::move(dofmap), element, std::move(xg), dim,
                  std::move(igi));
}
//-----------------------------------------------------------------------------
std::pair<mesh::Geometry, std::vector<int32_t>>
mesh::create_subgeometry(const Topology& topology, const Geometry& geometry,
                         int dim,
                         std::span<const std::int32_t> subentity_to_entity,
                         const common::IndexMap& submap)
{
  // Get the geometry dofs in the sub-geometry based on the entities in
  // sub-geometry
  const fem::ElementDofLayout layout = geometry.cmap().create_dof_layout();
  // NOTE: Unclear what this return for prisms
  const std::size_t num_entity_dofs = layout.num_entity_closure_dofs(dim);

  std::vector<std::int32_t> x_indices;
  x_indices.reserve(num_entity_dofs * subentity_to_entity.size());
  std::vector<std::int32_t> sub_x_dofmap_offsets;
  sub_x_dofmap_offsets.reserve(subentity_to_entity.size() + 1);
  sub_x_dofmap_offsets.push_back(0);
  {
    const graph::AdjacencyList<std::int32_t>& xdofs = geometry.dofmap();
    const int tdim = topology.dim();

    // Fetch connectivities required to get entity dofs
    const std::vector<std::vector<std::vector<int>>>& closure_dofs
        = layout.entity_closure_dofs_all();
    auto e_to_c = topology.connectivity(dim, tdim);
    assert(e_to_c);
    auto c_to_e = topology.connectivity(tdim, dim);
    assert(c_to_e);
    for (std::size_t i = 0; i < subentity_to_entity.size(); ++i)
    {
      const std::int32_t idx = subentity_to_entity[i];
      assert(!e_to_c->links(idx).empty());
      // Always pick the last cell to be consistent with the e_to_v connectivity
      const std::int32_t cell = e_to_c->links(idx).back();
      auto cell_entities = c_to_e->links(cell);
      auto it = std::find(cell_entities.begin(), cell_entities.end(), idx);
      assert(it != cell_entities.end());
      std::size_t local_entity = std::distance(cell_entities.begin(), it);

      auto xc = xdofs.links(cell);
      for (std::int32_t entity_dof : closure_dofs[dim][local_entity])
        x_indices.push_back(xc[entity_dof]);
      sub_x_dofmap_offsets.push_back(x_indices.size());
    }
  }

  std::vector<std::int32_t> sub_x_dofs = x_indices;
  std::sort(sub_x_dofs.begin(), sub_x_dofs.end());
  sub_x_dofs.erase(std::unique(sub_x_dofs.begin(), sub_x_dofs.end()),
                   sub_x_dofs.end());

  // Get the sub-geometry dofs owned by this process
  auto x_index_map = geometry.index_map();
  assert(x_index_map);

  auto [subx_to_x_dofmap, map_data] = x_index_map->create_submap(
      common::compute_owned_indices(sub_x_dofs, *x_index_map), sub_x_dofs);

  std::shared_ptr<common::IndexMap> sub_x_dof_index_map
      = std::make_shared<common::IndexMap>(std::move(map_data.first));

  // Create a map from the dofs in the sub-geometry to the geometry
  subx_to_x_dofmap.reserve(sub_x_dof_index_map->size_local()
                           + sub_x_dof_index_map->num_ghosts());
  std::transform(map_data.second.begin(), map_data.second.end(),
                 std::back_inserter(subx_to_x_dofmap),
                 [offset = x_index_map->size_local()](auto x_dof_index)
                 { return offset + x_dof_index; });

  // Create sub-geometry coordinates
  std::span<const double> x = geometry.x();
  std::int32_t sub_num_x_dofs = subx_to_x_dofmap.size();
  std::vector<double> sub_x(3 * sub_num_x_dofs);
  for (int i = 0; i < sub_num_x_dofs; ++i)
  {
    std::copy_n(std::next(x.begin(), 3 * subx_to_x_dofmap[i]), 3,
                std::next(sub_x.begin(), 3 * i));
  }

  // Create geometry to sub-geometry  map
  std::vector<std::int32_t> x_to_subx_dof_map(
      x_index_map->size_local() + x_index_map->num_ghosts(), -1);
  for (std::size_t i = 0; i < subx_to_x_dofmap.size(); ++i)
    x_to_subx_dof_map[subx_to_x_dofmap[i]] = i;

  // Create sub-geometry dofmap
  std::vector<std::int32_t> sub_x_dofmap_vec;
  sub_x_dofmap_vec.reserve(x_indices.size());
  std::transform(x_indices.cbegin(), x_indices.cend(),
                 std::back_inserter(sub_x_dofmap_vec),
                 [&x_to_subx_dof_map](auto x_dof)
                 {
                   assert(x_to_subx_dof_map[x_dof] != -1);
                   return x_to_subx_dof_map[x_dof];
                 });

  // Create sub-geometry coordinate element
  CellType sub_coord_cell
      = cell_entity_type(geometry.cmap().cell_shape(), dim, 0);
  fem::CoordinateElement sub_coord_ele(sub_coord_cell, geometry.cmap().degree(),
                                       geometry.cmap().variant());

  // Same communication as for the topology is also needed for the
  // geometry
  if (!(topology.dim() == dim))
  {
    std::vector<std::int64_t> sub_xdofmap_global_vec(sub_x_dofmap_vec.size(),
                                                     0);
    sub_x_dof_index_map->local_to_global(sub_x_dofmap_vec,
                                         sub_xdofmap_global_vec);

    const int x_dofs_per_entity = sub_coord_ele.dim();

    // FIXME See if this scatter can be done without the subtopology
    // index map
    common::Scatterer scatterer(submap, x_dofs_per_entity);
    std::vector<std::int64_t> ghost_x_dofs(
        x_dofs_per_entity * submap.num_ghosts(), 0);

    scatterer.scatter_fwd(std::span<const std::int64_t>(
                              sub_xdofmap_global_vec.begin(),
                              sub_xdofmap_global_vec.begin()
                                  + x_dofs_per_entity * submap.size_local()),
                          std::span<std::int64_t>(ghost_x_dofs));

    std::span<std::int32_t> ghost_x_dofs_local(
        sub_x_dofmap_vec.begin() + x_dofs_per_entity * submap.size_local(),
        sub_x_dofmap_vec.end());

    sub_x_dof_index_map->global_to_local(ghost_x_dofs, ghost_x_dofs_local);
  }

  graph::AdjacencyList<std::int32_t> sub_x_dofmap(
      std::move(sub_x_dofmap_vec), std::move(sub_x_dofmap_offsets));

  // Sub-geometry input_global_indices
  // TODO: Check this
  const std::vector<std::int64_t>& igi = geometry.input_global_indices();
  std::vector<std::int64_t> sub_igi;
  sub_igi.reserve(subx_to_x_dofmap.size());
  std::transform(subx_to_x_dofmap.begin(), subx_to_x_dofmap.end(),
                 std::back_inserter(sub_igi),
                 [&igi](std::int32_t sub_x_dof) { return igi[sub_x_dof]; });

  // Create geometry
  return {Geometry(sub_x_dof_index_map, std::move(sub_x_dofmap), sub_coord_ele,
                   std::move(sub_x), geometry.dim(), std::move(sub_igi)),
          std::move(subx_to_x_dofmap)};
}
//-----------------------------------------------------------------------------
