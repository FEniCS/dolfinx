// Copyright (C) 2006-2022 Anders Logg and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "Topology.h"
#include <concepts>
#include <cstdint>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/MPI.h>
#include <dolfinx/common/sort.h>
#include <dolfinx/fem/CoordinateElement.h>
#include <dolfinx/fem/ElementDofLayout.h>
#include <dolfinx/fem/dofmapbuilder.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/graph/partition.h>
#include <functional>
#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace dolfinx::mesh
{

/// @brief Geometry stores the geometry imposed on a mesh.
template <typename T>
class Geometry
{
public:
  /// @brief Constructor of object that holds mesh geometry data.
  ///
  /// @param[in] index_map Index map associated with the geometry dofmap
  /// @param[in] dofmap The geometry (point) dofmap. For a cell, it
  /// gives the position in the point array of each local geometry node
  /// @param[in] element The element that describes the cell geometry map
  /// @param[in] x The point coordinates. The shape is `(num_points, 3)`
  /// and the storage is row-major.
  /// @param[in] dim The geometric dimension (`0 < dim <= 3`)
  /// @param[in] input_global_indices The 'global' input index of each
  /// point, commonly from a mesh input file. The type is
  /// `std:vector<std::int64_t>`.
  template <std::convertible_to<graph::AdjacencyList<std::int32_t>> U,
            std::convertible_to<std::vector<T>> V,
            std::convertible_to<std::vector<std::int64_t>> W>
  Geometry(std::shared_ptr<const common::IndexMap> index_map, U&& dofmap,
           const fem::CoordinateElement& element, V&& x, int dim,
           W&& input_global_indices)
      : _dim(dim), _dofmap(std::forward<U>(dofmap)), _index_map(index_map),
        _cmap(element), _x(std::forward<V>(x)),
        _input_global_indices(std::forward<W>(input_global_indices))
  {
    assert(_x.size() % 3 == 0);
    if (_x.size() / 3 != _input_global_indices.size())
      throw std::runtime_error("Geometry size mis-match");
  }

  /// Copy constructor
  Geometry(const Geometry&) = default;

  /// Move constructor
  Geometry(Geometry&&) = default;

  /// Destructor
  ~Geometry() = default;

  /// Copy Assignment
  Geometry& operator=(const Geometry&) = delete;

  /// Move Assignment
  Geometry& operator=(Geometry&&) = default;

  /// Copy constructor
  template <typename U>
  Geometry<U> astype() const
  {
    return Geometry<U>(_index_map, _dofmap, _cmap,
                       std::vector<U>(_x.begin(), _x.end()), _dim,
                       _input_global_indices);
  }

  /// Return Euclidean dimension of coordinate system
  int dim() const { return _dim; }

  /// DOF map
  const graph::AdjacencyList<std::int32_t>& dofmap() const { return _dofmap; }

  /// Index map
  std::shared_ptr<const common::IndexMap> index_map() const
  {
    return _index_map;
  }

  /// @brief Access geometry degrees-of-freedom data (const version).
  ///
  /// @return The flattened row-major geometry data, where the shape is
  /// (num_points, 3)
  std::span<const T> x() const { return _x; }

  /// @brief Access geometry degrees-of-freedom data (non-const
  /// version).
  ///
  /// @return The flattened row-major geometry data, where the shape is
  /// (num_points, 3)
  std::span<T> x() { return _x; }

  /// @brief The element that describes the geometry map.
  ///
  /// @return The coordinate/geometry element
  const fem::CoordinateElement& cmap() const { return _cmap; }

  /// Global user indices
  const std::vector<std::int64_t>& input_global_indices() const
  {
    return _input_global_indices;
  }

private:
  // Geometric dimension
  int _dim;

  // Map per cell for extracting coordinate data
  graph::AdjacencyList<std::int32_t> _dofmap;

  // IndexMap for geometry 'dofmap'
  std::shared_ptr<const common::IndexMap> _index_map;

  // The coordinate element
  fem::CoordinateElement _cmap;

  // Coordinates for all points stored as a contiguous array (row-major,
  // column size = 3)
  std::vector<T> _x;

  // Global indices as provided on Geometry creation
  std::vector<std::int64_t> _input_global_indices;
};

/// @brief Build Geometry from input data.
///
/// This function should be called after the mesh topology is built. It
/// distributes the 'node' coordinate data to the required MPI process
/// and then creates a mesh::Geometry object.
///
/// @param[in] comm The MPI communicator to build the Geometry on
/// @param[in] topology The mesh topology
/// @param[in] element The element that defines the geometry map for
/// each cell
/// @param[in] cells The mesh cells, including higher-order geometry
/// 'nodes'
/// @param[in] x The node coordinates (row-major, with shape
/// `(num_nodes, dim)`. The global index of each node is `i +
/// rank_offset`, where `i` is the local row index in `x` and
/// `rank_offset` is the sum of `x` rows on all processed with a lower
/// rank than the caller.
/// @param[in] dim The geometric dimension (1, 2, or 3)
/// @param[in] reorder_fn Function for re-ordering the degree-of-freedom
/// map associated with the geometry data
template <typename T>
mesh::Geometry<T>
create_geometry(MPI_Comm comm, const Topology& topology,
                const fem::CoordinateElement& element,
                const graph::AdjacencyList<std::int64_t>& cell_nodes,
                std::span<const T> x, int dim,
                const std::function<std::vector<int>(
                    const graph::AdjacencyList<std::int32_t>&)>& reorder_fn
                = nullptr)
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
    std::vector<T> coords = MPI::distribute_data<T>(comm, indices, x, dim);

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
  std::vector<T> xg(3 * shape0, 0);
  for (std::size_t i = 0; i < shape0; ++i)
  {
    std::copy_n(std::next(coords.cbegin(), shape1 * l2l[i]), shape1,
                std::next(xg.begin(), 3 * i));
  }

  return Geometry<T>(dof_index_map, std::move(dofmap), element, std::move(xg),
                     dim, std::move(igi));
}

/// @brief Create a sub-geometry for a subset of entities.
/// @param topology Full mesh topology
/// @param geometry Full mesh geometry
/// @param dim Topological dimension of the sub-topology
/// @param subentity_to_entity Map from sub-topology entity to the
/// entity in the parent topology
/// @return A sub-geometry and a map from sub-geometry coordinate
/// degree-of-freedom to the coordinate degree-of-freedom in `geometry`.
template <typename T>
std::pair<mesh::Geometry<T>, std::vector<int32_t>>
create_subgeometry(const Topology& topology, const Geometry<T>& geometry,
                   int dim, std::span<const std::int32_t> subentity_to_entity)
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
  auto subx_to_x_dofmap
      = common::compute_owned_indices(sub_x_dofs, *x_index_map);
  std::shared_ptr<common::IndexMap> sub_x_dof_index_map;
  {
    std::pair<common::IndexMap, std::vector<int32_t>> map_data
        = x_index_map->create_submap(subx_to_x_dofmap);
    sub_x_dof_index_map
        = std::make_shared<common::IndexMap>(std::move(map_data.first));

    // Create a map from the dofs in the sub-geometry to the geometry
    subx_to_x_dofmap.reserve(sub_x_dof_index_map->size_local()
                             + sub_x_dof_index_map->num_ghosts());
    std::transform(map_data.second.begin(), map_data.second.end(),
                   std::back_inserter(subx_to_x_dofmap),
                   [offset = x_index_map->size_local()](auto x_dof_index)
                   { return offset + x_dof_index; });
  }

  // Create sub-geometry coordinates
  std::span<const T> x = geometry.x();
  std::int32_t sub_num_x_dofs = subx_to_x_dofmap.size();
  std::vector<T> sub_x(3 * sub_num_x_dofs);
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

  graph::AdjacencyList<std::int32_t> sub_x_dofmap(
      std::move(sub_x_dofmap_vec), std::move(sub_x_dofmap_offsets));

  // Create sub-geometry coordinate element
  CellType sub_coord_cell
      = cell_entity_type(geometry.cmap().cell_shape(), dim, 0);
  fem::CoordinateElement sub_coord_ele(sub_coord_cell, geometry.cmap().degree(),
                                       geometry.cmap().variant());

  // Sub-geometry input_global_indices
  // TODO: Check this
  const std::vector<std::int64_t>& igi = geometry.input_global_indices();
  std::vector<std::int64_t> sub_igi;
  sub_igi.reserve(subx_to_x_dofmap.size());
  std::transform(subx_to_x_dofmap.begin(), subx_to_x_dofmap.end(),
                 std::back_inserter(sub_igi),
                 [&igi](std::int32_t sub_x_dof) { return igi[sub_x_dof]; });

  // Create geometry
  return {Geometry<T>(sub_x_dof_index_map, std::move(sub_x_dofmap),
                      sub_coord_ele, std::move(sub_x), geometry.dim(),
                      std::move(sub_igi)),
          std::move(subx_to_x_dofmap)};
}

} // namespace dolfinx::mesh
