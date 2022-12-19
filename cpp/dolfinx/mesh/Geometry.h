// Copyright (C) 2006-2020 Anders Logg and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <concepts>
#include <dolfinx/common/MPI.h>
#include <dolfinx/fem/CoordinateElement.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <functional>
#include <memory>
#include <span>
#include <vector>

namespace dolfinx::common
{
class IndexMap;
}

namespace dolfinx::mesh
{
class Topology;

/// @brief Geometry stores the geometry imposed on a mesh.
class Geometry
{
public:
  /// @brief Constructor of object that holds mesh geometry data.
  ///
  /// @param[in] index_map Index map associated with the geometry dofmap
  /// @param[in] dofmap The geometry (point) dofmap. For a cell, it
  /// gives the position in the point array of each local geometry node
  /// @param[in] element The element that describes the cell geometry map
  /// @param[in] x The point coordinates. It is a `std::vector<double>`
  /// and uses row-major storage. The shape is `(num_points, 3)`.
  /// @param[in] dim The geometric dimension (`0 < dim <= 3`)
  /// @param[in] input_global_indices The 'global' input index of each
  /// point, commonly from a mesh input file. The type is
  /// `std:vector<std::int64_t>`.
  template <std::convertible_to<graph::AdjacencyList<std::int32_t>> U,
            std::convertible_to<std::vector<double>> V,
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

  /// Return Euclidean dimension of coordinate system
  int dim() const;

  /// DOF map
  const graph::AdjacencyList<std::int32_t>& dofmap() const;

  /// Index map
  std::shared_ptr<const common::IndexMap> index_map() const;

  /// @brief Access geometry degrees-of-freedom data (const version).
  ///
  /// @return The flattened row-major geometry data, where the shape is
  /// (num_points, 3)
  std::span<const double> x() const;

  /// @brief Access geometry degrees-of-freedom data (non-const
  /// version).
  ///
  /// @return The flattened row-major geometry data, where the shape is
  /// (num_points, 3)
  std::span<double> x();

  /// @brief The element that describes the geometry map.
  ///
  /// @return The coordinate/geometry element
  const fem::CoordinateElement& cmap() const;

  /// Global user indices
  const std::vector<std::int64_t>& input_global_indices() const;

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
  std::vector<double> _x;

  // Global indices as provided on Geometry creation
  std::vector<std::int64_t> _input_global_indices;
};

/// @brief Build Geometry from input data.
///
/// This function should be called after the mesh topology is built. It
/// distributes the 'node' coordinate data to the required MPI process and then
/// creates a mesh::Geometry object.
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
mesh::Geometry
create_geometry(MPI_Comm comm, const Topology& topology,
                const fem::CoordinateElement& element,
                const graph::AdjacencyList<std::int64_t>& cells,
                std::span<const double> x, int dim,
                const std::function<std::vector<int>(
                    const graph::AdjacencyList<std::int32_t>&)>& reorder_fn
                = nullptr);

} // namespace dolfinx::mesh
