// Copyright (C) 2006-2018 Anders Logg and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <Eigen/Dense>
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

/// Geometry stores the geometry imposed on a mesh.

/// Currently, the geometry is represented by the set of coordinates for
/// the vertices of a mesh, but other representations are possible.

class Geometry
{
public:
  Geometry(std::int64_t num_points_global,
           const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                              Eigen::RowMajor>& coordinates,
           const std::vector<std::int64_t>& global_indices);

  /// Copy constructor
  Geometry(const Geometry&) = default;

  /// Move constructor
  Geometry(Geometry&&) = default;

  /// Destructor
  ~Geometry() = default;

  /// Copy Assignment
  Geometry& operator=(const Geometry&) = default;

  /// Move Assignment
  Geometry& operator=(Geometry&&) = default;

  /// Return Euclidean dimension of coordinate system
  std::size_t dim() const;

  /// Return the number of local points in the geometry
  std::size_t num_points() const;

  /// Return the number of global points in the geometry
  std::size_t num_points_global() const;

  /// Return coordinate array for point with local index n
  Eigen::Ref<const Eigen::Vector3d>
  x(std::size_t n) const;

  // Should this return an Eigen::Ref?
  /// Return array of coordinates for all points
  Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>&
  points();

  // Should this return an Eigen::Ref?
  /// Return array of coordinates for all points (const version)
  const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>&
  points() const;

  /// Global indices for points (const)
  const std::vector<std::int64_t>& global_indices() const;

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
  Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>
      _coordinates;

  // Geometric dimension
  int _dim;

  // Global indices for points
  std::vector<std::int64_t> _global_indices;

  // Global number of points (taking account of shared points)
  std::uint64_t _num_points_global;
};
} // namespace mesh
} // namespace dolfin
