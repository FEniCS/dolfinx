// Copyright (C) 2006-2020 Anders Logg and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "Geometry.h"
#include "Topology.h"
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
                         std::span<const std::int32_t> submesh_to_mesh_map)
{
  // Get the geometry dofs in the sub-geometry based on the entities in
  // sub-geometry
  const fem::ElementDofLayout layout = geometry.cmap().create_dof_layout();
  // NOTE: Unclear what this return for prisms
  const std::size_t num_entity_dofs = layout.num_entity_closure_dofs(dim);

  std::vector<std::int32_t> geometry_indices;
  geometry_indices.reserve(num_entity_dofs * submesh_to_mesh_map.size());
  std::vector<std::int32_t> submesh_x_dofmap_offsets;
  submesh_x_dofmap_offsets.reserve(submesh_to_mesh_map.size() + 1);
  submesh_x_dofmap_offsets.push_back(0);
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
    for (std::size_t i = 0; i < submesh_to_mesh_map.size(); ++i)
    {
      const std::int32_t idx = submesh_to_mesh_map[i];
      assert(!e_to_c->links(idx).empty());
      // Always pick the last cell to be consistent with the e_to_v connectivity
      const std::int32_t cell = e_to_c->links(idx).back();
      auto cell_entities = c_to_e->links(cell);
      auto it = std::find(cell_entities.begin(), cell_entities.end(), idx);
      assert(it != cell_entities.end());
      std::size_t local_entity = std::distance(cell_entities.begin(), it);

      auto xc = xdofs.links(cell);
      for (std::int32_t entity_dof : closure_dofs[dim][local_entity])
        geometry_indices.push_back(xc[entity_dof]);
      submesh_x_dofmap_offsets.push_back(geometry_indices.size());
    }
  }

  std::vector<std::int32_t> submesh_x_dofs = geometry_indices;
  std::sort(submesh_x_dofs.begin(), submesh_x_dofs.end());
  submesh_x_dofs.erase(
      std::unique(submesh_x_dofs.begin(), submesh_x_dofs.end()),
      submesh_x_dofs.end());

  // Get the geometry dofs in the submesh owned by this process
  auto mesh_geometry_dof_index_map = geometry.index_map();
  assert(mesh_geometry_dof_index_map);
  auto submesh_owned_x_dofs = common::compute_owned_indices(
      submesh_x_dofs, *mesh_geometry_dof_index_map);

  // Create submesh geometry index map
  std::vector<int32_t> submesh_to_mesh_x_dof_map(submesh_owned_x_dofs.begin(),
                                                 submesh_owned_x_dofs.end());
  std::shared_ptr<common::IndexMap> submesh_x_dof_index_map;
  {
    std::pair<common::IndexMap, std::vector<int32_t>>
        submesh_x_dof_index_map_pair
        = mesh_geometry_dof_index_map->create_submap(submesh_owned_x_dofs);

    submesh_x_dof_index_map = std::make_shared<common::IndexMap>(
        std::move(submesh_x_dof_index_map_pair.first));

    // Create a map from the (local) geometry dofs in the submesh to the
    // (local) geometry dofs in the mesh.
    submesh_to_mesh_x_dof_map.reserve(submesh_x_dof_index_map->size_local()
                                      + submesh_x_dof_index_map->num_ghosts());
    std::transform(submesh_x_dof_index_map_pair.second.begin(),
                   submesh_x_dof_index_map_pair.second.end(),
                   std::back_inserter(submesh_to_mesh_x_dof_map),
                   [size = mesh_geometry_dof_index_map->size_local()](
                       auto x_dof_index) { return size + x_dof_index; });
  }

  // Create submesh geometry coordinates
  std::span<const double> mesh_x = geometry.x();
  const int submesh_num_x_dofs = submesh_to_mesh_x_dof_map.size();
  std::vector<double> submesh_x(3 * submesh_num_x_dofs);
  for (int i = 0; i < submesh_num_x_dofs; ++i)
  {
    std::copy_n(std::next(mesh_x.begin(), 3 * submesh_to_mesh_x_dof_map[i]), 3,
                std::next(submesh_x.begin(), 3 * i));
  }

  // Create mesh to submesh geometry map
  std::vector<std::int32_t> mesh_to_submesh_x_dof_map(
      mesh_geometry_dof_index_map->size_local()
          + mesh_geometry_dof_index_map->num_ghosts(),
      -1);
  for (std::size_t i = 0; i < submesh_to_mesh_x_dof_map.size(); ++i)
    mesh_to_submesh_x_dof_map[submesh_to_mesh_x_dof_map[i]] = i;

  // Create submesh geometry dofmap
  std::vector<std::int32_t> submesh_x_dofmap_vec;
  submesh_x_dofmap_vec.reserve(geometry_indices.size());
  std::transform(geometry_indices.cbegin(), geometry_indices.cend(),
                 std::back_inserter(submesh_x_dofmap_vec),
                 [&mesh_to_submesh_x_dof_map](auto x_dof)
                 {
                   std::int32_t x_dof_submesh
                       = mesh_to_submesh_x_dof_map[x_dof];
                   assert(x_dof_submesh != -1);
                   return x_dof_submesh;
                 });

  graph::AdjacencyList<std::int32_t> submesh_x_dofmap(
      std::move(submesh_x_dofmap_vec), std::move(submesh_x_dofmap_offsets));

  // Create submesh coordinate element
  CellType submesh_coord_cell
      = cell_entity_type(geometry.cmap().cell_shape(), dim, 0);
  fem::CoordinateElement submesh_coord_ele(
      submesh_coord_cell, geometry.cmap().degree(), geometry.cmap().variant());

  // Submesh geometry input_global_indices
  // TODO Check this
  const std::vector<std::int64_t>& mesh_igi = geometry.input_global_indices();
  std::vector<std::int64_t> submesh_igi;
  submesh_igi.reserve(submesh_to_mesh_x_dof_map.size());
  std::transform(submesh_to_mesh_x_dof_map.begin(),
                 submesh_to_mesh_x_dof_map.end(),
                 std::back_inserter(submesh_igi),
                 [&mesh_igi](std::int32_t submesh_x_dof)
                 { return mesh_igi[submesh_x_dof]; });

  // Create geometry
  return {Geometry(submesh_x_dof_index_map, std::move(submesh_x_dofmap),
                   submesh_coord_ele, std::move(submesh_x), geometry.dim(),
                   std::move(submesh_igi)),
          std::move(submesh_to_mesh_x_dof_map)};
}
//-----------------------------------------------------------------------------
