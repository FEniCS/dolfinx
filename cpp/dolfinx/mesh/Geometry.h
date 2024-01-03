// Copyright (C) 2006-2022 Anders Logg and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "Topology.h"
#include <basix/mdspan.hpp>
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
template <std::floating_point T>
class Geometry
{
public:
  /// @brief Value type
  using value_type = T;

  /// @brief Constructor of object that holds mesh geometry data.
  ///
  /// @param[in] index_map Index map associated with the geometry dofmap
  /// @param[in] dofmap The geometry (point) dofmap. For a cell, it
  /// gives the position in the point array of each local geometry node
  /// @param[in] element Element that describes the cell geometry map.
  /// @param[in] x The point coordinates. The shape is `(num_points, 3)`
  /// and the storage is row-major.
  /// @param[in] dim The geometric dimension (`0 < dim <= 3`).
  /// @param[in] input_global_indices The 'global' input index of each
  /// point, commonly from a mesh input file.
  template <typename U, typename V, typename W>
    requires std::is_convertible_v<std::remove_cvref_t<U>,
                                   std::vector<std::int32_t>>
                 and std::is_convertible_v<std::remove_cvref_t<V>,
                                           std::vector<T>>
                 and std::is_convertible_v<std::remove_cvref_t<W>,
                                           std::vector<std::int64_t>>
  Geometry(
      std::shared_ptr<const common::IndexMap> index_map, U&& dofmap,
      const fem::CoordinateElement<
          typename std::remove_reference_t<typename V::value_type>>& element,
      V&& x, int dim, W&& input_global_indices)
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

  /// Return Euclidean dimension of coordinate system
  int dim() const { return _dim; }

  /// DOF map
  MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      const std::int32_t,
      MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>
  dofmap() const
  {
    int ndofs = _cmap.dim();
    return MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        const std::int32_t,
        MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>(
        _dofmap.data(), _dofmap.size() / ndofs, ndofs);
  }

  /// Index map
  std::shared_ptr<const common::IndexMap> index_map() const
  {
    return _index_map;
  }

  /// @brief Access geometry degrees-of-freedom data (const version).
  ///
  /// @return The flattened row-major geometry data, where the shape is
  /// (num_points, 3)
  std::span<const value_type> x() const { return _x; }

  /// @brief Access geometry degrees-of-freedom data (non-const
  /// version).
  ///
  /// @return The flattened row-major geometry data, where the shape is
  /// (num_points, 3)
  std::span<value_type> x() { return _x; }

  /// @brief The elements that describes the geometry maps.
  ///
  /// @return The coordinate/geometry element
  const fem::CoordinateElement<value_type>& cmap() const { return _cmap; }

  /// Global user indices
  const std::vector<std::int64_t>& input_global_indices() const
  {
    return _input_global_indices;
  }

private:
  // Geometric dimension
  int _dim;

  // Map per cell for extracting coordinate data
  std::vector<std::int32_t> _dofmap;

  // IndexMap for geometry 'dofmap'
  std::shared_ptr<const common::IndexMap> _index_map;

  // The coordinate elements
  fem::CoordinateElement<value_type> _cmap;

  // Coordinates for all points stored as a contiguous array (row-major,
  // column size = 3)
  std::vector<value_type> _x;

  // Global indices as provided on Geometry creation
  std::vector<std::int64_t> _input_global_indices;
};

/// @cond
/// Template type deduction
template <typename U, typename V, typename W>
Geometry(std::shared_ptr<const common::IndexMap>, U,
         const fem::CoordinateElement<
             typename std::remove_reference_t<typename V::value_type>>&,
         V, int, W)
    -> Geometry<typename std::remove_cvref_t<typename V::value_type>>;
/// @endcond

/// @brief Build Geometry from input data.
///
/// This function should be called after the mesh topology is built and
/// 'node' coordinate data has been distributed to the processes where
/// it is required.
///
/// @param[in] topology Mesh topology.
/// @param[in] element Element that defines the geometry map for
/// each cell.
/// @param[in] nodes Geometry node global indices for cells on this
/// process. Must be sorted.
/// @param[in] xdofs Geometry degree-of-freedom map (using global
/// indices) for cells on this process. `nodes` is a sorted and unique
/// list of the indices in `xdofs`.
/// @param[in] x The node coordinates (row-major, with shape
/// `(num_nodes, dim)`. The global index of each node is `i +
/// rank_offset`, where `i` is the local row index in `x` and
/// `rank_offset` is the sum of `x` rows on all processed with a lower
/// rank than the caller.
/// @param[in] dim Geometric dimension (1, 2, or 3).
/// @param[in] reorder_fn Function for re-ordering the degree-of-freedom
/// map associated with the geometry data.
/// @return A mesh geometry.
template <typename U>
Geometry<typename std::remove_reference_t<typename U::value_type>>
create_geometry(
    const Topology& topology,
    const fem::CoordinateElement<
        std::remove_reference_t<typename U::value_type>>& element,
    std::span<const std::int64_t> nodes,

    std::span<const std::int64_t> xdofs, const U& x, int dim,
    std::function<std::vector<int>(const graph::AdjacencyList<std::int32_t>&)>
        reorder_fn
    = nullptr)
{
  assert(std::is_sorted(nodes.begin(), nodes.end()));
  using T = typename std::remove_reference_t<typename U::value_type>;

  fem::ElementDofLayout doflayout = element.create_dof_layout();

  //  Build 'geometry' dofmap on the topology
  auto [_dof_index_map, bs, dofmap]
      = fem::build_dofmap_data(topology.index_map(topology.dim())->comm(),
                               topology, doflayout, reorder_fn);
  auto dof_index_map
      = std::make_shared<common::IndexMap>(std::move(_dof_index_map));

  // If the mesh has higher order geometry, permute the dofmap
  if (element.needs_dof_permutations())
  {
    const std::int32_t num_cells
        = topology.connectivity(topology.dim(), 0)->num_nodes();
    const std::vector<std::uint32_t>& cell_info
        = topology.get_cell_permutation_info();
    int d = element.dim();
    for (std::int32_t cell = 0; cell < num_cells; ++cell)
    {
      std::span dofs(dofmap.data() + cell * d, d);
      element.unpermute_dofs(dofs, cell_info[cell]);
    }
  }

  // Compute local-to-global map from local indices in dofmap to the
  // corresponding global indices in cells, and pass to function to
  // compute local (dof) to local (position in coords) map from (i)
  // local-to-global for dofs and (ii) local-to-global for entries in
  // coords
  const std::vector<std::int32_t> l2l = graph::build::compute_local_to_local(
      graph::build::compute_local_to_global(xdofs, dofmap), nodes);

  // Allocate space for input global indices and copy data
  std::vector<std::int64_t> igi(nodes.size());
  std::transform(l2l.cbegin(), l2l.cend(), igi.begin(),
                 [&nodes](auto index) { return nodes[index]; });

  // Build coordinate dof array, copying coordinates to correct position
  assert(x.size() % dim == 0);
  const std::size_t shape0 = x.size() / dim;
  const std::size_t shape1 = dim;
  std::vector<T> xg(3 * shape0, 0);
  for (std::size_t i = 0; i < shape0; ++i)
  {
    std::copy_n(std::next(x.cbegin(), shape1 * l2l[i]), shape1,
                std::next(xg.begin(), 3 * i));
  }

  return Geometry(dof_index_map, std::move(dofmap), element, std::move(xg), dim,
                  std::move(igi));
}

/// @brief Create a sub-geometry for a subset of entities.
/// @param topology Full mesh topology.
/// @param geometry Full mesh geometry.
/// @param dim Topological dimension of the sub-topology.
/// @param subentity_to_entity Map from sub-topology entity to the
/// entity in the parent topology.
/// @return A sub-geometry and a map from sub-geometry coordinate
/// degree-of-freedom to the coordinate degree-of-freedom in `geometry`.
template <std::floating_point T>
std::pair<Geometry<T>, std::vector<int32_t>>
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
  {
    auto xdofs = geometry.dofmap();
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

      auto xc = MDSPAN_IMPL_STANDARD_NAMESPACE::MDSPAN_IMPL_PROPOSED_NAMESPACE::
          submdspan(xdofs, cell, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
      for (std::int32_t entity_dof : closure_dofs[dim][local_entity])
        x_indices.push_back(xc[entity_dof]);
    }
  }

  std::vector<std::int32_t> sub_x_dofs = x_indices;
  std::sort(sub_x_dofs.begin(), sub_x_dofs.end());
  sub_x_dofs.erase(std::unique(sub_x_dofs.begin(), sub_x_dofs.end()),
                   sub_x_dofs.end());

  // Get the sub-geometry dofs owned by this process
  auto x_index_map = geometry.index_map();
  assert(x_index_map);

  std::shared_ptr<common::IndexMap> sub_x_dof_index_map;
  std::vector<std::int32_t> subx_to_x_dofmap;
  {
    auto [map, new_to_old]
        = common::create_sub_index_map(*x_index_map, sub_x_dofs, true);
    sub_x_dof_index_map = std::make_shared<common::IndexMap>(std::move(map));
    subx_to_x_dofmap = std::move(new_to_old);
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
  std::vector<std::int32_t> sub_x_dofmap;
  sub_x_dofmap.reserve(x_indices.size());
  std::transform(x_indices.cbegin(), x_indices.cend(),
                 std::back_inserter(sub_x_dofmap),
                 [&x_to_subx_dof_map](auto x_dof)
                 {
                   assert(x_to_subx_dof_map[x_dof] != -1);
                   return x_to_subx_dof_map[x_dof];
                 });

  // Create sub-geometry coordinate element
  CellType sub_coord_cell
      = cell_entity_type(geometry.cmap().cell_shape(), dim, 0);
  fem::CoordinateElement<T> sub_cmap(sub_coord_cell, geometry.cmap().degree(),
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
  return {Geometry(sub_x_dof_index_map, std::move(sub_x_dofmap), sub_cmap,
                   std::move(sub_x), geometry.dim(), std::move(sub_igi)),
          std::move(subx_to_x_dofmap)};
}

} // namespace dolfinx::mesh
