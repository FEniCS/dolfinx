// Copyright (C) 2006-2018 Anders Logg and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfin/common/types.h>
#include <dolfin/geometry/Point.h>
#include <memory>
#include <string>
#include <vector>

namespace dolfin
{
namespace fem
{
class CoordinateMapping;
}

namespace mesh
{

/// MeshGeometry stores the geometry imposed on a mesh.

/// Currently, the geometry is represented by the set of coordinates for the
/// vertices of a mesh, but other representations are possible.
class MeshGeometry
{
public:
  /// Create set of coordinates
  MeshGeometry(const Eigen::Ref<const EigenRowArrayXXd>& points);

  /// Copy constructor
  MeshGeometry(const MeshGeometry&) = default;

  /// Move constructor
  MeshGeometry(MeshGeometry&&) = default;

  /// Destructor
  ~MeshGeometry() = default;

  /// Copy Assignment
  MeshGeometry& operator=(const MeshGeometry&) = default;

  /// Move Assignment
  MeshGeometry& operator=(MeshGeometry&&) = default;

  /// Return Euclidean dimension of coordinate system
  std::size_t dim() const { return _coordinates.cols(); }

  /// Return the number of local points in the geometry
  std::size_t num_points() const { return _coordinates.rows(); }

  /// Return the number of global points in the geometry
  std::size_t num_points_global() const { return _num_points_global; }

  /// Return coordinate array for point with local index n
  Eigen::Ref<const EigenRowArrayXd> x(std::size_t n) const
  {
    return _coordinates.row(n);
  }

  /// Return coordinate with local index n as a 3D point value
  geometry::Point point(std::size_t n) const;

  // Should this return an Eigen::Ref?
  /// Return array of coordinates for all points
  EigenRowArrayXXd& points() { return _coordinates; }

  // Should this return an Eigen::Ref?
  /// Return array of coordinates for all points (const version)
  const EigenRowArrayXXd& points() const { return _coordinates; }

  /// Global indices for points (const)
  const std::vector<std::int64_t>& global_indices() const
  {
    return _global_indices;
  }

  /// Initialise MeshGeometry data
  void init(std::uint64_t num_points_global, const EigenRowArrayXXd& coordinates,
            const std::vector<std::int64_t>& global_indices)
  {
    _num_points_global = num_points_global;
    _coordinates = coordinates;
    _global_indices = global_indices;
  }

  /// Hash of coordinate values
  ///
  /// @returns std::size_t
  ///    A tree-hashed value of the coordinates over all MPI processes
  ///
  std::size_t hash() const;

  /// Return informal string representation (pretty-print)
  std::string str(bool verbose) const;

  /// Put CoordinateMapping for now. Experimental.
  std::shared_ptr<const fem::CoordinateMapping> coord_mapping;

private:
  // Coordinates for all points stored as a contiguous array
  EigenRowArrayXXd _coordinates;

  // Global indices for points
  std::vector<std::int64_t> _global_indices;

  // Global number of points (taking account of shared points)
  std::uint64_t _num_points_global;
};
} // namespace mesh
} // namespace dolfin
