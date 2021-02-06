// Copyright (C) 2006-2020 Anders Logg and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfinx/common/MPI.h>
#include <dolfinx/common/ndVector.h>
#include <dolfinx/fem/CoordinateElement.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <memory>
#include <vector>

namespace dolfinx
{
namespace common
{
class IndexMap;
}

namespace fem
{
class CoordinateElement;
} // namespace fem

namespace mesh
{
class Topology;

/// Geometry stores the geometry imposed on a mesh.
///
/// Currently, the geometry is represented by the set of coordinates for
/// the vertices of a mesh, but other representations are possible.

class Geometry
{
public:
  /// Constructor
  template <typename AdjacencyList32, typename Vector64>
  Geometry(const std::shared_ptr<const common::IndexMap>& index_map,
           AdjacencyList32&& dofmap, const fem::CoordinateElement& element,
           const common::ndVector<double>& x, Vector64&& input_global_indices)
      : _dim(x.cols()), _dofmap(std::forward<AdjacencyList32>(dofmap)),
        _index_map(index_map), _cmap(element),
        _input_global_indices(std::forward<Vector64>(input_global_indices))
  {
    if (x.rows() != _input_global_indices.size())
      throw std::runtime_error("Size mis-match");

    // Make all geometry 3D
    if (_dim == 3)
      _x = x;
    else if (_dim != 3)
    {
      _x.resize(x.rows(), 3, 0);
      _x.copy(x);
    }
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

  /// Geometry degrees-of-freedom
  common::ndVector<double>& x();

  /// Geometry degrees-of-freedom
  const common::ndVector<double>& x() const;

  /// The element that describes the geometry map
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

  // Coordinates for all points stored as a contiguous array
  common::ndVector<double> _x;

  // Global indices as provided on Geometry creation
  std::vector<std::int64_t> _input_global_indices;
};

/// Build Geometry
/// FIXME: document
mesh::Geometry create_geometry(MPI_Comm comm, const Topology& topology,
                               const fem::CoordinateElement& coordinate_element,
                               const graph::AdjacencyList<std::int64_t>& cells,
                               const common::ndVector<double>& x);

} // namespace mesh
} // namespace dolfinx
