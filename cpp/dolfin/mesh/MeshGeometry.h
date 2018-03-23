// Copyright (C) 2006 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfin/common/types.h>
#include <dolfin/geometry/Point.h>
#include <dolfin/log/log.h>
#include <string>
#include <vector>

namespace dolfin
{
namespace mesh
{

/// MeshGeometry stores the geometry imposed on a mesh.

/// Currently, the geometry is represented by the set of coordinates for the
/// vertices of a mesh, but other representations are possible.
class MeshGeometry
{
public:
  /// Create empty set of coordinates
  MeshGeometry(Eigen::Ref<const EigenRowArrayXXd> points);

  /// Copy constructor
  MeshGeometry(const MeshGeometry&) = default;

  /// Move constructor
  MeshGeometry(MeshGeometry&&) = default;

  /// Destructor
  ~MeshGeometry() = default;

  /// Assignment
  MeshGeometry& operator=(const MeshGeometry&) = default;

  /// Return Euclidean dimension of coordinate system
  std::size_t dim() const { return _dim; }

  /// Return polynomial degree of coordinate field
  std::size_t degree() const { return _degree; }

  /// Return the number of vertex coordinates
  std::size_t num_vertices() const
  {
    dolfin_assert(coordinates.size() % _dim == 0);
    return coordinates.size() / _dim;
  }

  /// Return the total number of points in the geometry, located on
  /// any entity
  std::size_t num_points() const
  {
    dolfin_assert(coordinates.size() % _dim == 0);
    return coordinates.size() / _dim;
  }

  /// Get vertex coordinates
  const double* vertex_coordinates(std::size_t point_index)
  {
    dolfin_assert(point_index < num_vertices());
    return &coordinates[point_index * _dim];
  }

  /// Get vertex coordinates
  const double* point_coordinates(std::size_t point_index)
  {
    dolfin_assert(point_index * _dim < coordinates.size());
    return &coordinates[point_index * _dim];
  }

  /// Return value of coordinate with local index n in direction i
  double x(std::size_t n, std::size_t i) const
  {
    dolfin_assert((n * _dim + i) < coordinates.size());
    dolfin_assert(i < _dim);
    return coordinates[n * _dim + i];
  }

  /// Return array of values for coordinate with local index n
  const double* x(std::size_t n) const
  {
    dolfin_assert(n * _dim < coordinates.size());
    return &coordinates[n * _dim];
  }

  /// Return array of values for all coordinates
  std::vector<double>& x() { return coordinates; }

  /// Return array of values for all coordinates
  const std::vector<double>& x() const { return coordinates; }

  /// Return coordinate with local index n as a 3D point value
  geometry::Point point(std::size_t n) const;

  /// Set value of coordinate
  void set(std::size_t local_index, const double* x);

  /// Hash of coordinate values
  ///
  /// @returns std::size_t
  ///    A tree-hashed value of the coordinates over all MPI processes
  ///
  std::size_t hash() const;

  /// Return informal string representation (pretty-print)
  std::string str(bool verbose) const;

private:
  // Euclidean dimension
  std::size_t _dim;

  // Polynomial degree (1 = linear, 2 = quadratic etc.)
  std::size_t _degree;

  // Coordinates for all points stored as a contiguous array
  std::vector<double> coordinates;
};
} // namespace mesh
} // namespace dolfin
