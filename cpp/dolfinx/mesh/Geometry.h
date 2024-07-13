// Copyright (C) 2006-2022 Anders Logg and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "Topology.h"
#include <algorithm>
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
      : _dim(dim), _dofmaps({dofmap}), _index_map(index_map), _cmaps({element}),
        _x(std::forward<V>(x)),
        _input_global_indices(std::forward<W>(input_global_indices))
  {
    assert(_x.size() % 3 == 0);
    if (_x.size() / 3 != _input_global_indices.size())
      throw std::runtime_error("Geometry size mis-match");
  }

  /// @brief Constructor of object that holds mesh geometry data.
  ///
  /// @param[in] index_map Index map associated with the geometry dofmap
  /// @param[in] dofmaps The geometry (point) dofmaps. For a cell, it
  /// gives the position in the point array of each local geometry node
  /// @param[in] elements Elements that describes the cell geometry maps.
  /// @param[in] x The point coordinates. The shape is `(num_points, 3)`
  /// and the storage is row-major.
  /// @param[in] dim The geometric dimension (`0 < dim <= 3`).
  /// @param[in] input_global_indices The 'global' input index of each
  /// point, commonly from a mesh input file.
  template <typename V, typename W>
    requires std::is_convertible_v<std::remove_cvref_t<V>, std::vector<T>>
                 and std::is_convertible_v<std::remove_cvref_t<W>,
                                           std::vector<std::int64_t>>
  Geometry(
      std::shared_ptr<const common::IndexMap> index_map,
      const std::vector<std::vector<std::int32_t>>& dofmaps,
      const std::vector<fem::CoordinateElement<
          typename std::remove_reference_t<typename V::value_type>>>& elements,
      V&& x, int dim, W&& input_global_indices)
      : _dim(dim), _dofmaps(dofmaps), _index_map(index_map), _cmaps(elements),
        _x(std::forward<V>(x)),
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

  /// @brief  DofMap for the geometry
  /// @return A 2D array with shape [num_cells, dofs_per_cell]
  MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      const std::int32_t,
      MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>
  dofmap() const
  {
    if (_dofmaps.size() != 1)
      throw std::runtime_error("Multiple dofmaps");

    int ndofs = _cmaps.front().dim();
    return MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        const std::int32_t,
        MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>(
        _dofmaps.front().data(), _dofmaps.front().size() / ndofs, ndofs);
  }

  /// @brief The dofmap associated with the `i`th coordinate map in the
  /// geometry.
  /// @param i Index
  /// @return A 2D array with shape [num_cells, dofs_per_cell]
  MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      const std::int32_t,
      MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>
  dofmap(std::int32_t i) const
  {
    if (i < 0 or i >= (int)_dofmaps.size())
    {
      throw std::out_of_range("Cannot get dofmap:" + std::to_string(i)
                              + " out of range");
    }
    int ndofs = _cmaps[i].dim();

    return MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        const std::int32_t,
        MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>(
        _dofmaps[i].data(), _dofmaps[i].size() / ndofs, ndofs);
  }

  /// @brief Index map
  /// @return The index map for the geometry dofs
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

  /// @brief The element that describes the geometry map.
  ///
  /// @return The coordinate/geometry element
  const fem::CoordinateElement<value_type>& cmap() const
  {
    if (_cmaps.size() != 1)
      throw std::runtime_error("Multiple cmaps.");
    return _cmaps.front();
  }

  /// @brief The element that describe the `i`th geometry map
  /// @param i Index of the coordinate element
  /// @return Coordinate element
  const fem::CoordinateElement<value_type>& cmap(std::int32_t i) const
  {
    if (i < 0 or i >= (int)_cmaps.size())
    {
      throw std::out_of_range("Cannot get cmap:" + std::to_string(i)
                              + " out of range");
    }
    return _cmaps[i];
  }

  /// Global user indices
  const std::vector<std::int64_t>& input_global_indices() const
  {
    return _input_global_indices;
  }

private:
  // Geometric dimension
  int _dim;

  // Map per cell for extracting coordinate data for each cmap
  std::vector<std::vector<std::int32_t>> _dofmaps;

  // IndexMap for geometry 'dofmap'
  std::shared_ptr<const common::IndexMap> _index_map;

  // The coordinate elements
  std::vector<fem::CoordinateElement<value_type>> _cmaps;

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
         const std::vector<fem::CoordinateElement<
             typename std::remove_reference_t<typename V::value_type>>>&,
         V, int,
         W) -> Geometry<typename std::remove_cvref_t<typename V::value_type>>;
/// @endcond

/// @brief Build Geometry from input data.
///
/// This function should be called after the mesh topology is built and
/// 'node' coordinate data has been distributed to the processes where
/// it is required.
///
/// @param[in] topology Mesh topology.
/// @param[in] elements List of elements that defines the geometry map for
/// each cell type.
/// @param[in] nodes Geometry node global indices for cells on this
/// process. @pre Must be sorted.
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
/// @note Experimental new interface for multiple cmap/dofmap
/// @return A mesh geometry.
template <typename U>
Geometry<typename std::remove_reference_t<typename U::value_type>>
create_geometry(
    const Topology& topology,
    const std::vector<fem::CoordinateElement<
        std::remove_reference_t<typename U::value_type>>>& elements,
    std::span<const std::int64_t> nodes, std::span<const std::int64_t> xdofs,
    const U& x, int dim,
    std::function<std::vector<int>(const graph::AdjacencyList<std::int32_t>&)>
        reorder_fn
    = nullptr)
{
  spdlog::info("Create Geometry (multiple)");

  assert(std::ranges::is_sorted(nodes));
  using T = typename std::remove_reference_t<typename U::value_type>;

  // Check elements match cell types in topology
  const int tdim = topology.dim();
  const std::size_t num_cell_types = topology.entity_types(tdim).size();
  if (elements.size() != num_cell_types)
    throw std::runtime_error("Mismatch between topology and geometry.");

  std::vector<fem::ElementDofLayout> dof_layouts;
  for (const auto& el : elements)
    dof_layouts.push_back(el.create_dof_layout());

  spdlog::info("Got {} dof layouts", dof_layouts.size());

  //  Build 'geometry' dofmap on the topology
  auto [_dof_index_map, bs, dofmaps]
      = fem::build_dofmap_data(topology.index_map(topology.dim())->comm(),
                               topology, dof_layouts, reorder_fn);
  auto dof_index_map
      = std::make_shared<common::IndexMap>(std::move(_dof_index_map));

  // If the mesh has higher order geometry, permute the dofmap
  if (elements.front().needs_dof_permutations())
  {
    const std::int32_t num_cells
        = topology.connectivity(topology.dim(), 0)->num_nodes();
    const std::vector<std::uint32_t>& cell_info
        = topology.get_cell_permutation_info();
    int d = elements.front().dim();
    for (std::int32_t cell = 0; cell < num_cells; ++cell)
    {
      std::span dofs(dofmaps.front().data() + cell * d, d);
      elements.front().permute_inv(dofs, cell_info[cell]);
    }
  }

  spdlog::info("Calling compute_local_to_global");
  // Compute local-to-global map from local indices in dofmap to the
  // corresponding global indices in cells, and pass to function to
  // compute local (dof) to local (position in coords) map from (i)
  // local-to-global for dofs and (ii) local-to-global for entries in
  // coords

  spdlog::info("xdofs.size = {}", xdofs.size());
  std::vector<std::int32_t> all_dofmaps;
  std::stringstream s;
  for (auto q : dofmaps)
  {
    s << q.size() << " ";
    all_dofmaps.insert(all_dofmaps.end(), q.begin(), q.end());
  }
  spdlog::info("dofmap sizes = {}", s.str());
  spdlog::info("all_dofmaps.size = {}", all_dofmaps.size());
  spdlog::info("nodes.size = {}", nodes.size());

  const std::vector<std::int32_t> l2l = graph::build::compute_local_to_local(
      graph::build::compute_local_to_global(xdofs, all_dofmaps), nodes);

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
    std::copy_n(std::next(x.begin(), shape1 * l2l[i]), shape1,
                std::next(xg.begin(), 3 * i));
  }

  spdlog::info("Creating geometry with {} dofmaps", dof_layouts.size());

  return Geometry(dof_index_map, std::move(dofmaps), elements, std::move(xg),
                  dim, std::move(igi));
}

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
    std::span<const std::int64_t> nodes, std::span<const std::int64_t> xdofs,
    const U& x, int dim,
    std::function<std::vector<int>(const graph::AdjacencyList<std::int32_t>&)>
        reorder_fn
    = nullptr)
{
  assert(std::ranges::is_sorted(nodes));
  using T = typename std::remove_reference_t<typename U::value_type>;

  fem::ElementDofLayout dof_layout = element.create_dof_layout();

  //  Build 'geometry' dofmap on the topology
  auto [_dof_index_map, bs, dofmaps]
      = fem::build_dofmap_data(topology.index_map(topology.dim())->comm(),
                               topology, {dof_layout}, reorder_fn);
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
      std::span dofs(dofmaps.front().data() + cell * d, d);
      element.permute_inv(dofs, cell_info[cell]);
    }
  }

  // Compute local-to-global map from local indices in dofmap to the
  // corresponding global indices in cells, and pass to function to
  // compute local (dof) to local (position in coords) map from (i)
  // local-to-global for dofs and (ii) local-to-global for entries in
  // coords
  const std::vector<std::int32_t> l2l = graph::build::compute_local_to_local(
      graph::build::compute_local_to_global(xdofs, dofmaps.front()), nodes);

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

  return Geometry(dof_index_map, std::move(dofmaps.front()), {element},
                  std::move(xg), dim, std::move(igi));
}

} // namespace dolfinx::mesh
