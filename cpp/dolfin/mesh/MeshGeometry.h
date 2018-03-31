// Copyright (C) 2006 Anders Logg
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

namespace ufc
{
class coordinate_mapping;
}

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

  /// Return the total number of points in the geometry, located on
  /// any entity
  std::size_t num_points() const { return _coordinates.rows(); }

  /// Get point coordinates
  const double* point_coordinates(std::size_t point_index)
  {
    return _coordinates.row(point_index).data();
  }

  /// Return value of coordinate with local index n in direction i
  double x(std::size_t n, std::size_t i) const { return _coordinates(n, i); }

  /// Return array of values for coordinate with local index n
  const double* x(std::size_t n) const { return _coordinates.row(n).data(); }

  // Should this return an Eigen::Ref?
  /// Return array of coordinates for all points
  EigenRowArrayXXd& points() { return _coordinates; }

  // Should this return an Eigen::Ref?
  /// Return array of coordinates for all points (const version)
  const EigenRowArrayXXd& points() const { return _coordinates; }

  /// Return coordinate with local index n as a 3D point value
  geometry::Point point(std::size_t n) const;

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
};
} // namespace mesh
} // namespace dolfin
