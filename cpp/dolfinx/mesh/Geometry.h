// Copyright (C) 2006-2020 Anders Logg and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <Eigen/Dense>
#include <dolfinx/common/MPI.h>
#include <dolfinx/fem/ElementDofLayout.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <memory>
#include <string>
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
  Geometry(std::shared_ptr<const common::IndexMap> index_map,
           const graph::AdjacencyList<std::int32_t>& dofmap,
           const fem::ElementDofLayout& layout,
           const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                              Eigen::RowMajor>& x,
           const std::vector<std::int64_t>& global_indices);

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

  /// @todo Remove this non-const version. Just here for mesh::Ordering
  ///
  /// DOF map
  graph::AdjacencyList<std::int32_t>& dofmap();

  /// DOF map
  const graph::AdjacencyList<std::int32_t>& dofmap() const;

  /// Index map
  std::shared_ptr<const common::IndexMap> index_map() const;

  /// Geometry degrees-of-freedom
  Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>& x();

  /// Geometry degrees-of-freedom
  const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>& x() const;

  /// Return coordinate array for node n (index is local to the process)
  Eigen::Vector3d node(int n) const;

  /// Global input indices for points (const)
  const std::vector<std::int64_t>& global_indices() const;

  /// @warning Experimental. Needs revision
  ///
  /// Put ElementDofLayout here for now
  const fem::ElementDofLayout& dof_layout() const;

  /// Hash of coordinate values
  /// @return A tree-hashed value of the coordinates over all MPI
  ///   processes
  std::size_t hash() const;

  /// @warning Experimental. Needs revision
  ///
  /// Put CoordinateElement here for now
  std::shared_ptr<const fem::CoordinateElement> coord_mapping;

private:
  // Geometric dimension
  int _dim;

  // Map per cell for extracting coordinate data
  graph::AdjacencyList<std::int32_t> _dofmap;

  // IndexMap for geometry 'dofmap'
  std::shared_ptr<const common::IndexMap> _index_map;

  // The dof layout on the cell
  fem::ElementDofLayout _layout;

  // Coordinates for all points stored as a contiguous array
  Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor> _x;

  // Global indices for points
  std::vector<std::int64_t> _global_indices;
};

/// Build Geometry
/// FIXME: document
mesh::Geometry create_geometry(
    MPI_Comm comm, const Topology& topology,
    const fem::ElementDofLayout& layout,
    const graph::AdjacencyList<std::int64_t>& cells,
    const Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                        Eigen::RowMajor>>& x);

} // namespace mesh
} // namespace dolfinx
